# Remote Sensing Time-Series Classification with LSTM

This repository contains a PyTorch LSTM pipeline for binary classification on remote sensing time-series data stored as NumPy arrays.

The current project setup is designed for large `.npy` files and uses memory-mapped loading so the full dataset does not need to fit into RAM at once.

## Dataset Assumptions

The code expects the following files under `data/`:

- `x_sr_s.npy`: time-series reflectance features with shape `[N, T, C]`
- `label.npy`: binary labels with shape `[N]`

For the current dataset:

- `x_sr_s.npy` shape: `25000000 x 22 x 10`
- `label.npy` values: `0/1`

`data/` is ignored by Git and is not included in this repository.

## Environment

Recommended environment used in this project:

- Conda env: `pyt20`
- PyTorch: `2.5.1+cu121`
- GPU: `RTX 3090 24GB`
- RAM: `96GB`

Example activation:

```powershell
conda run -n pyt20 python train.py --config configs/train/smoke.json
```

## Project Structure

```text
.
|-- configs/
|   |-- train/
|   |   |-- best_800k.json
|   |   `-- smoke.json
|   `-- eval/
|       `-- holdout_800k.json
|-- models/
|   `-- lstm_classifier.py
|-- utils/
|   |-- checkpoint.py
|   |-- config.py
|   |-- dataset.py
|   |-- metrics.py
|   `-- runtime.py
|-- train.py
|-- evaluate_holdout.py
|-- train_lstm_classifier.py
|-- eval_test_holdout.py
`-- .gitignore
```

## Main Entry Points

Training:

- `train.py`: main training entrypoint
- `train_lstm_classifier.py`: compatibility wrapper that forwards to `train.py`

Evaluation:

- `evaluate_holdout.py`: evaluate a saved checkpoint on a stratified holdout set
- `eval_test_holdout.py`: compatibility wrapper that forwards to `evaluate_holdout.py`

## Configuration Style

The project supports JSON config files through `--config`.

Example training config:

- [configs/train/best_800k.json](configs/train/best_800k.json)

Example evaluation config:

- [configs/eval/holdout_800k.json](configs/eval/holdout_800k.json)

CLI flags can still override config values.

Example:

```powershell
conda run -n pyt20 python train.py --config configs/train/best_800k.json --epochs 12
```

## Training

### Smoke Test

```powershell
conda run -n pyt20 python train.py --config configs/train/smoke.json
```

### Best Current Training Configuration

```powershell
conda run -n pyt20 python train.py --config configs/train/best_800k.json
```

This configuration corresponds to:

- `sample_size=800000`
- `batch_size=512`
- `epochs=8`
- `hidden_size=128`
- `num_layers=2`
- `bidirectional=true`
- `dropout=0.2`
- `lr=5e-4`
- `standardize=true`

## Evaluation

Run holdout evaluation for a trained checkpoint:

```powershell
conda run -n pyt20 python evaluate_holdout.py --config configs/eval/holdout_800k.json
```

This evaluates the checkpoint on a stratified holdout set sampled from data not used in the corresponding training run.

## Current Best Result

Using the `best_800k` training configuration, the current holdout result is approximately:

- Accuracy: `0.972385`
- Precision: `0.963717`
- Recall: `0.967550`
- F1: `0.965630`

## Notes

- `num_workers=0` is used because multi-process `DataLoader` caused Windows permission issues in the current environment.
- `artifacts/` is ignored by Git and is intended for checkpoints, logs, and evaluation outputs.
- The utility code includes compatibility handling for the local Python/PyTorch package environment.

## Git Ignore

The following are intentionally excluded from version control:

- `data/`
- `artifacts/`
- `.vscode/`
- Python cache files
