# UAV model artifacts

Place the **Kaggle export** from `/kaggle/working/uav_model/` here:

| File | Required |
|------|----------|
| `lstm_autoencoder.pth` | **Yes** — trained conservative LSTM weights |
| `scaler.json` | **Yes** — mean/std fit on train flights |
| `model_config.json` | Present (verify matches your export) |
| `feature_schema.json` | Present |
| `threshold_config.json` | Present — do not edit `lstm_ae` |
| `preprocessing.py` | Present (session-aware v3.0) |
| `inference.py` | Optional — app uses `src/` wrappers |

Copy command example:

```bash
cp /path/to/kaggle/uav_model/lstm_autoencoder.pth ./uav_model/
cp /path/to/kaggle/uav_model/scaler.json ./uav_model/
# Prefer overwriting configs with your export if they differ
```

### Checkpoint note (macOS)

PyTorch `.pth` files are zip archives. If Finder expands them into an
`lstm_autoencoder/` folder, keep that folder — `src/artifact_loader.py` will
rebuild a Torch-readable `lstm_autoencoder.pth` automatically (entries must be
prefixed with `lstm_autoencoder/`).

Until `lstm_autoencoder.pth` (or a repairable extracted folder) and `scaler.json`
are present, the app will refuse to run analysis and show a clear error. Tests
use a temporary fixture checkpoint under `tests/fixtures/`.

### What is deployed

See [`docs/DEPLOYED_MODEL.md`](../docs/DEPLOYED_MODEL.md) — the generated, authoritative
description of the feature list, scaler, input shape, window parameters and threshold
(including SHA-256 of each artifact). `scripts/describe_deployment.py --check` fails if it
drifts from this directory.
