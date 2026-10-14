"""
Step 4b - Experiments: which setup gives the best dry/wet model?  (5-fold cross-validation)

Questions answered:
  Q1  Does pretraining on cough detection help?        init = scratch | pretrained
  Q2  Does a longer window help (wet "tail")?          clip = 1.0 s | 1.5 s | 2.0 s
  Q3  Does averaging over several coughs help?         score 1 window vs the mean of the K loudest windows

Why cross-validation?
  The single validation set has only 68 wet coughs, so one lucky or unlucky split can move the
  scores a lot. Here the train + val recordings (1,683) are split into 5 equal parts ("folds").
  Each fold takes one turn as the held-out set while the model trains on the other four, so every
  recording is scored exactly once by a model that never saw it. We report mean +/- std over folds.
  Inside each turn, 15% of the training part is set aside for early stopping, so the held-out fold
  is used ONLY for scoring (no peeking). The TEST set (300 recordings) is not touched here at all.

Usage (from "Model Testing", venv active; ~20-30 min on a GPU for all 6 setups):
  python src/experiments.py
  python src/experiments.py --clips 1.0 1.5 --inits pretrained     (a subset)

Outputs (experiments/):
  cv_folds.csv     one row per setup x fold
  cv_summary.csv   mean and std per setup  (the table for the report)
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureConfig, FeatureExtractor, load_wav, peak_normalize  # noqa: E402
from model import CoughNet                                                      # noqa: E402
from metrics import binary_report, roc_auc                                       # noqa: E402
from train import (build_cache, read_manifest, run_epoch, train_phase,          # noqa: E402
                   set_seed)

NAMES = ("dry", "wet")


# ---------------------------------------------------------------- folds
def stratified_folds(labels, k, seed):
    """Return k lists of indices; each fold keeps the same dry/wet ratio."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for cls in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % k].append(i)
    return folds


def split_inner(idx, labels, frac, seed):
    """Stratified split of the training part: (train, early-stopping) indices."""
    rng = random.Random(seed)
    tr, es = [], []
    for cls in sorted(set(labels[i] for i in idx)):
        group = [i for i in idx if labels[i] == cls]
        rng.shuffle(group)
        n_es = max(1, int(len(group) * frac))
        es += group[:n_es]
        tr += group[n_es:]
    return tr, es


def subset(cache, idx):
    t = torch.tensor(idx, dtype=torch.long)
    return {"X": cache["X"][t], "y": cache["y"][t]}


# ---------------------------------------------------------------- multi-window scoring
def top_windows(x, cfg, k):
    """
    The k loudest NON-overlapping windows of clip_seconds in a recording.
    Repeatedly takes the loudest remaining window and blocks out its span, so with 3 coughs
    in a recording we get one window per cough instead of three copies of the loudest one.
    """
    n = int(cfg.clip_seconds * cfg.sample_rate)
    if x.numel() <= n:
        return [torch.nn.functional.pad(x, (0, n - x.numel()))]
    hop = cfg.hop_length
    frames = x.unfold(0, cfg.win_length, hop).pow(2).mean(dim=1)
    w = max(1, n // hop)
    c = torch.cat([torch.zeros(1), torch.cumsum(frames, 0)])
    sums = (c[w:] - c[:-w]).clone() if frames.numel() > w else c[-1:].clone()
    out, peak = [], sums.max().item()
    for _ in range(k):
        j = int(torch.argmax(sums).item())
        if sums[j].item() <= 0.05 * peak and out:      # remaining windows are near-silent
            break
        start = min(j * hop, x.numel() - n)
        out.append(x[start:start + n])
        lo, hi = max(0, j - w + 1), min(sums.numel(), j + w)
        sums[lo:hi] = -1.0
    return out


@torch.no_grad()
def score_recordings(model, fx, cfg, paths, k, device):
    """P(wet) per recording = mean over its k loudest windows."""
    model.eval()
    probs = []
    for p in paths:
        x, _ = load_wav(p)
        wins = torch.stack([peak_normalize(wi) for wi in top_windows(x, cfg, k)]).to(device)
        pr = torch.softmax(model(fx.log_mel(wins)), dim=1)[:, 1]
        probs.append(pr.mean().item())
    return torch.tensor(probs)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--clips", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--inits", nargs="+", default=["scratch", "pretrained"],
                    choices=["scratch", "pretrained"])
    ap.add_argument("--pretrained", default=os.path.join("checkpoints", "pretrain_best.pt"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k-windows", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    if "pretrained" in a.inits and not os.path.exists(a.pretrained):
        sys.exit(f"{a.pretrained} not found: run `python src/train.py pretrain` first.")

    # train + val recordings together; the test set stays untouched
    items = []
    for split in ("train", "val"):
        items += read_manifest(os.path.join(a.data, "manifests", f"drywet_{split}.csv"))
    labels = [l for _, _, l in items]
    paths = [os.path.join(a.data, rel) for _, rel, _ in items]
    folds = stratified_folds(labels, a.folds, a.seed)
    print(f"device: {device} | {len(items)} recordings ({labels.count(1)} wet) | "
          f"{a.folds} folds | setups: clips {a.clips} x {a.inits}")

    rows = []
    fold_csv = os.path.join(a.out, "cv_folds.csv")
    with open(fold_csv, "w", newline="") as f:
        fw = csv.writer(f)
        fw.writerow(["clip_s", "init", "fold", "best_epoch", "auc_1win", f"auc_{a.k_windows}win",
                     "bal_acc_1win", f"bal_acc_{a.k_windows}win", "f1_wet_1win",
                     f"f1_wet_{a.k_windows}win", "recall_wet_1win", "precision_wet_1win",
                     "threshold", "seconds"])

        for clip in a.clips:
            cfg = FeatureConfig(clip_seconds=clip)
            fx = FeatureExtractor(cfg).to(device)
            cache_s = round(clip + 0.2, 2)
            parts = [build_cache(os.path.join(a.data, "manifests", f"drywet_{s}.csv"), a.data, cfg,
                                 os.path.join(a.data, "cache", f"drywet_{s}_{cache_s}s.pt"),
                                 cache_seconds=cache_s, verbose=False) for s in ("train", "val")]
            full = {"X": torch.cat([p["X"] for p in parts]), "y": torch.cat([p["y"] for p in parts])}

            for init in a.inits:
                for k in range(a.folds):
                    t0 = time.time()
                    set_seed(a.seed + k)
                    held = folds[k]
                    rest = [i for j, fo in enumerate(folds) if j != k for i in fo]
                    tr_idx, es_idx = split_inner(rest, labels, 0.15, a.seed + k)

                    model = CoughNet(2).to(device)
                    if init == "pretrained":
                        ck = torch.load(a.pretrained, map_location=device, weights_only=True)
                        model.load_state_dict(ck["model"])
                        model.replace_head(2)
                        model.to(device)
                        lr, freeze = 3e-4, 3
                    else:
                        lr, freeze = 1e-3, 0
                    ns = SimpleNamespace(out=os.path.join(a.out, "runs"), batch_size=a.batch_size,
                                         weight_decay=1e-4, patience=a.patience,
                                         channels=[16, 32, 64, 128])
                    run = f"clip{clip}_{init}_fold{k + 1}"
                    best = train_phase(model, fx, cfg, subset(full, tr_idx), subset(full, es_idx),
                                       device, ns, run, NAMES, epochs=a.epochs, lr=lr,
                                       freeze_epochs=freeze, verbose=False)
                    ck = torch.load(best, map_location=device, weights_only=True)
                    model.load_state_dict(ck["model"])
                    thr = ck["threshold"]                       # chosen on the early-stopping set

                    # score the held-out fold: (a) one loudest window, (b) mean of K windows
                    _, hy, hp1 = run_epoch(model, fx, subset(full, held), cfg, device, a.batch_size,
                                           loss_fn=torch.nn.CrossEntropyLoss())
                    hpk = score_recordings(model, fx, cfg, [paths[i] for i in held],
                                           a.k_windows, device)
                    r1 = binary_report(hy, hp1, thr, NAMES)
                    rk = binary_report(hy, hpk, thr, NAMES)
                    secs = time.time() - t0
                    row = [clip, init, k + 1, ck["epoch"], r1["roc_auc"], rk["roc_auc"],
                           r1["balanced_accuracy"], rk["balanced_accuracy"], r1["f1"], rk["f1"],
                           r1["recall"], r1["precision"], thr, secs]
                    fw.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])
                    f.flush()
                    rows.append(row)
                    print(f"  clip {clip:.1f}s {init:10s} fold {k + 1}: AUC 1-win {r1['roc_auc']:.3f}"
                          f" | {a.k_windows}-win {rk['roc_auc']:.3f}  bal.acc {r1['balanced_accuracy']:.3f}"
                          f"  (best ep {ck['epoch']}, {secs:.0f}s)")

    # summary: mean +/- std per setup
    sum_csv = os.path.join(a.out, "cv_summary.csv")
    cols = {"auc_1win": 4, f"auc_{a.k_windows}win": 5, "bal_acc_1win": 6,
            f"bal_acc_{a.k_windows}win": 7, "f1_wet_1win": 8, f"f1_wet_{a.k_windows}win": 9}
    print("\n=== 5-fold cross-validation summary (mean +/- std) ===")
    print(f"{'clip':>5} {'init':>11} | " + " | ".join(f"{c:>17}" for c in cols))
    with open(sum_csv, "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["clip_s", "init"] + [f"{c}_{s}" for c in cols for s in ("mean", "std")])
        for clip in a.clips:
            for init in a.inits:
                rs = [r for r in rows if r[0] == clip and r[1] == init]
                stats = []
                for c, j in cols.items():
                    v = [r[j] for r in rs]
                    stats.append((statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0))
                sw.writerow([clip, init] + [f"{x:.4f}" for m, s in stats for x in (m, s)])
                print(f"{clip:>5.1f} {init:>11} | " +
                      " | ".join(f"{m:.3f} +/- {s:.3f}".rjust(17) for m, s in stats))
    print(f"\nWritten: {fold_csv}, {sum_csv}")


if __name__ == "__main__":
    main()
