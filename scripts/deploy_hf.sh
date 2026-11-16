#!/usr/bin/env bash
# Push the backend (only) to a Hugging Face Docker Space.
#
#   scripts/deploy_hf.sh <hf-username>/<space-name>
#
# Requires a Hugging Face write token; git will prompt for it as the password
# (username = your HF username) unless a credential helper already has it.
set -euo pipefail

SPACE="${1:?usage: scripts/deploy_hf.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

git clone "https://huggingface.co/spaces/${SPACE}" "$STAGE"

# Replace the Space contents with the current backend snapshot.
find "$STAGE" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
cp "$ROOT/Dockerfile" "$ROOT/.dockerignore" "$ROOT/requirements.txt" "$STAGE/"
cp "$ROOT/deploy/huggingface/README.md" "$STAGE/README.md"
for d in backend src uav_model config; do
  rsync -a --exclude '__pycache__' --exclude '*.pyc' "$ROOT/$d" "$STAGE/"
done

cd "$STAGE"
git add -A
if git diff --cached --quiet; then
  echo "Space is already up to date."
  exit 0
fi
git commit -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
git push
echo "Pushed. Build logs: https://huggingface.co/spaces/${SPACE}"
