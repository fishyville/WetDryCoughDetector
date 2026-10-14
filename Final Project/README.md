# Real-Time Dry/Wet Cough Classifier: App

COMP6822001 Speech Recognition, Final Project.

This folder holds only the application. The model is trained in `..\Model Testing\` and copied here.

## Folder layout

```
Final Project/
  app/
    main.py          real-time loop: microphone -> chunks -> cough detection -> dry/wet -> display
    features.py      COPY of Model Testing/src/features.py (must be identical)
    model.py         COPY of Model Testing/src/model.py    (must be identical)
    models/
      cough_model.pt   trained weights (from Model Testing/exports/)
      config.json      sample rate, window/hop size, n_mels, class names, thresholds
  requirements.txt
```

## How the model gets here

1. Train and evaluate in `Model Testing`.
2. Run `python src\export.py` there, which writes `exports\cough_model.pt` and `exports\config.json`.
3. Copy those two files and `src\features.py` and `src\model.py` into `app\`.

The app must compute features exactly as training did (same sample rate, window, hop and mel settings).
`config.json` stores those settings so they can't drift apart.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
