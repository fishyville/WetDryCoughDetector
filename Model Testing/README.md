# Model Testing: Training the Dry/Wet Cough Model

This folder is the "lab": data preparation, feature extraction, training, evaluation and export.
The finished model is exported to `exports/` and copied to `..\Final Project\app\models\`.

## Setup (Windows)

1. **Python 3.11 or 3.12.** Download it from python.org and tick "Add python.exe to PATH".
2. **FFmpeg.** Run `winget install Gyan.FFmpeg`, open a *new* terminal, and check `ffmpeg -version`.
3. **Virtual environment.** In this `Model Testing` folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install torch numpy
   ```
   With an NVIDIA GPU, use the CUDA install command from pytorch.org instead.
4. **Prepare the data:**
   ```
   python scripts\prepare_dataset.py --only drywet   # about 2k files, a few minutes
   python scripts\prepare_dataset.py                 # all about 27k files, needs ~9 GB free
   ```

## Folder layout

```
Model Testing/
  coughvid/                    raw COUGHVID dataset (already here)
  Respiratory_Sound_Database/  ICBHI (not used for the main model)
  scripts/prepare_dataset.py   step 1: filter, split, convert (done)
  src/features.py              step 2: STFT -> log-mel/MFCC, own code   <- shared with app
  src/model.py                 step 3: small CNN                         <- shared with app
  src/metrics.py               own accuracy/precision/recall/F1/confusion matrix
  src/train.py                 step 4: pretrain on cough detection, fine-tune dry/wet
  src/evaluate.py              test set results, noise/microphone experiments
  src/export.py                step 5: save weights + config to exports/
  data/wav/                    converted audio (generated)
  data/manifests/              train/val/test lists (generated)
  exports/                     cough_model.pt + config.json -> copy to the app
```

## Data (COUGHVID v3)

| Task | Labels | Recordings |
|---|---|---|
| Dry/wet (main) | expert majority vote; cough score >= 0.8; no "poor" or "no_cough" quality | 1,983 (1,526 dry / 457 wet) |
| Cough detection (pretraining) | cough_detected >= 0.8 is cough, <= 0.2 is non-cough | about 27,000 (18.5k / 8.5k) |

Split: 70/15/15, stratified, seed 42. Dry/wet val and test recordings are excluded from detector training.

winget install Python.Python.3.11
