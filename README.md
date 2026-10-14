# Real-Time Cough Detection and Dry/Wet Classification

COMP6822001 Speech Recognition final project (BINUS). The app listens to a microphone, detects coughs in
real time and estimates whether each cough sounds dry or wet. It is a screening aid, not a diagnosis.

```
Model Testing/   data preparation, features, CNN, training, experiments, evaluation, export
Final Project/   the real-time app (receives the exported models)
```

All audio processing, the model, metrics and training are written in PyTorch only.
FFmpeg is used once, to convert the dataset to WAV.

## Setup (Windows)

1. Install Python 3.11 or 3.12, and FFmpeg (`winget install Gyan.FFmpeg`; check with `ffmpeg -version`).
2. In `Model Testing`:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch numpy matplotlib
   ```
   (matplotlib is only used to draw figures.)

## Option A: use the trained models (no dataset needed)

The three trained checkpoints are included in `Model Testing/checkpoints/` (~400 KB each).

```
cd "Model Testing"
python src\export.py --to-app
```

This copies the models, `config.json`, `features.py` and `model.py` into `Final Project/app/`.

## Option B: reproduce everything

1. Download COUGHVID v3 (2021-10-12) from Zenodo: https://doi.org/10.5281/zenodo.4048311
   and extract it so this folder exists: `Model Testing/coughvid/coughvid_20211012/`
   (it contains `metadata_compiled.csv` plus the audio files).
2. In `Model Testing`, venv active:
   ```
   python scripts\prepare_dataset.py          # filter, split 70/15/15, convert to 16 kHz WAV (~9 GB)
   python src\train.py pretrain               # cough vs non-cough
   python src\train.py finetune               # dry vs wet, from the pretrained model
   python src\train.py finetune --scratch     # comparison: no pretraining
   python src\experiments.py                  # 5-fold cross-validation (optional, ~25 min on GPU)
   python src\evaluate.py                     # final test, robustness, latency
   python src\export.py --to-app              # copy models into the app
   ```

## Results (test set)

| Model | ROC-AUC | Notes |
| --- | --- | --- |
| Cough detector | 0.978 | catches 97.7% of coughs |
| Dry/wet classifier | 0.654 (95% CI 0.586-0.718) | weak but above chance; doctors only partly agree on dry vs wet |

Both run in about 3.4 ms per 1-second window on a CPU. See `Model Testing/README.md` and `Model Testing/results/`.
