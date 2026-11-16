#!/usr/bin/env bash
# Push the backend (only) to a Hugging Face Docker Space.
#
#   scripts/deploy_hf.sh <hf-username>/<space-name>
#
# Uses huggingface_hub (handles Xet storage for the .pth weights, which plain git
# pushes are rejected for). Auth: `hf auth login` once, or export HF_TOKEN.
set -euo pipefail

SPACE="${1:?usage: scripts/deploy_hf.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$ROOT/requirements.txt" "$STAGE/"
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"
for d in backend src config; do
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/$d" "$STAGE/"
done

# Call upload_folder directly: `hf upload` first tries to create the repo on
# free hardware, which Hugging Face refuses (402) for Docker Spaces.
# delete_patterns removes files from the Space that are no longer in the snapshot.
python3 - "$SPACE" "$STAGE" "Deploy $(git -C "$ROOT" rev-parse --short HEAD)" <<'PY'
import sys
from huggingface_hub import HfApi

space, folder, message = sys.argv[1:4]
HfApi().upload_folder(
    repo_id=space, repo_type="space", folder_path=folder, commit_message=message,
    delete_patterns=["backend/*", "src/*", "uav_model/*", "config/*"],  # uav_model/ removes the old LSTM files
)
PY
echo "Uploaded. Build logs: https://huggingface.co/spaces/${SPACE}"
