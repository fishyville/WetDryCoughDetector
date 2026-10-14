"""
Step 5b - Export the trained models for the app.

Writes to exports/ (and, with --to-app, also into ../Final Project/app/):
  models/drywet.pt      dry/wet model weights only (state_dict)
  models/detector.pt    cough-detector weights only (state_dict)
  models/config.json    everything the app needs to reproduce training exactly:
                        feature settings, thresholds, class names, streaming settings,
                        and the test results (for the about-box / report)
  features.py, model.py exact copies of the training code (the app must not re-implement them)

Usage (from "Model Testing", after evaluate.py):
  python src/export.py
  python src/export.py --to-app
"""

import argparse
import datetime
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import CoughNet   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drywet", default=os.path.join("checkpoints", "drywet_finetune_best.pt"))
    ap.add_argument("--detector", default=os.path.join("checkpoints", "pretrain_best.pt"))
    ap.add_argument("--results", default=os.path.join("results", "test_results.json"))
    ap.add_argument("--out", default="exports")
    ap.add_argument("--to-app", action="store_true",
                    help="also copy into ../Final Project/app/")
    ap.add_argument("--app-dir", default=os.path.join("..", "Final Project", "app"))
    ap.add_argument("--hop-seconds", type=float, default=0.25,
                    help="streaming: how often the app evaluates a new 1-second window")
    a = ap.parse_args()

    dw = torch.load(a.drywet, map_location="cpu", weights_only=True)
    det = torch.load(a.detector, map_location="cpu", weights_only=True)
    if dw["feature_config"] != det["feature_config"]:
        sys.exit("The two models were trained with different feature settings - cannot export together.")

    # sanity check: the weights load into a fresh model
    for ck in (dw, det):
        m = CoughNet(2, tuple(ck.get("channels", [16, 32, 64, 128])))
        m.load_state_dict(ck["model"])

    test = None
    if os.path.exists(a.results):
        with open(a.results) as f:
            r = json.load(f)
        test = {"drywet_auc": r["drywet_test"]["roc_auc"],
                "drywet_balanced_accuracy": r["drywet_test"]["balanced_accuracy"],
                "drywet_wet_recall": r["drywet_test"]["recall"],
                "detector_auc": r["detector_test"]["roc_auc"],
                "detector_cough_recall": r["detector_test"]["recall"]}
    else:
        print(f"note: {a.results} not found (run evaluate.py first to include test results)")

    config = {
        "exported": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_config": dw["feature_config"],
        "channels": dw.get("channels", [16, 32, 64, 128]),
        "drywet": {"file": "drywet.pt", "class_names": dw["class_names"],
                   "threshold": dw["threshold"], "best_epoch": dw["epoch"],
                   "val_auc": dw["val_report"]["roc_auc"]},
        "detector": {"file": "detector.pt", "class_names": det["class_names"],
                     "threshold": det["threshold"], "best_epoch": det["epoch"],
                     "val_auc": det["val_report"]["roc_auc"]},
        "streaming": {"window_seconds": dw["feature_config"]["clip_seconds"],
                      "hop_seconds": a.hop_seconds},
        "test_results": test,
        "disclaimer": "Screening aid only, not a medical diagnosis.",
    }

    src = os.path.dirname(os.path.abspath(__file__))
    targets = [a.out] + ([a.app_dir] if a.to_app else [])
    for t in targets:
        os.makedirs(os.path.join(t, "models"), exist_ok=True)
        torch.save(dw["model"], os.path.join(t, "models", "drywet.pt"))
        torch.save(det["model"], os.path.join(t, "models", "detector.pt"))
        with open(os.path.join(t, "models", "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        for name in ("features.py", "model.py"):
            shutil.copy2(os.path.join(src, name), os.path.join(t, name))
        print(f"exported to {t}/  (models/drywet.pt, models/detector.pt, models/config.json, "
              f"features.py, model.py)")


if __name__ == "__main__":
    main()
