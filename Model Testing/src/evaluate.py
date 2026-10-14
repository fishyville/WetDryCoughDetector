"""
Step 5a - Final evaluation of the chosen models. Run ONCE, after all choices are made.

  dry/wet model : checkpoints/drywet_finetune_best.pt   (threshold chosen on validation)
  cough detector: checkpoints/pretrain_best.pt          (threshold chosen on validation)

What it measures (all on the TEST set, which no training or selection step has seen):
  1. Headline results      ROC-AUC, balanced accuracy, recall/precision, confusion matrix,
                           with 95% bootstrap confidence intervals (how much the numbers could
                           move with a different 300 recordings).
  2. Robustness            the brief's "baseline vs experimental condition":
                             clean (baseline)
                             white noise           at 30 / 20 / 10 / 5 / 0 dB SNR
                             real background noise at 20 / 10 / 5 dB SNR (non-cough recordings
                                                   from the test set mixed in: speech, rooms, hum)
                             phone-quality audio   300-3400 Hz band only (telephone band)
                           for both the dry/wet model and the cough detector.
  3. Multi-cough summary   dry/wet decided from ONLY the windows the cough detector confirms,
                           averaged (how the app could summarise several coughs).
  4. Latency               time per 1-second window: features + detector + dry/wet model,
                           on CPU and (if present) GPU; real-time factor.

Usage (from "Model Testing", venv active):
  python src/evaluate.py
Outputs: results/test_results.json, results/robustness.csv, results/latency.json,
         results/summary.txt, results/figures/*.png (if matplotlib is installed)
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureConfig, FeatureExtractor, load_wav, peak_normalize   # noqa: E402
from model import CoughNet                                                       # noqa: E402
from metrics import binary_report, roc_auc, format_report                        # noqa: E402
from train import build_cache, read_manifest, center_crop                        # noqa: E402
from experiments import top_windows                                              # noqa: E402

SEED = 1234


# ---------------------------------------------------------------- helpers
def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
    m = CoughNet(2, tuple(ck.get("channels", [16, 32, 64, 128]))).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


@torch.no_grad()
def predict(model, fx, waves, device, batch=128):
    """waves: [N, samples] float -> P(class 1) for each."""
    out = []
    for i in range(0, len(waves), batch):
        w = waves[i:i + batch].to(device)
        out.append(torch.softmax(model(fx.log_mel(w)), dim=1)[:, 1].cpu())
    return torch.cat(out)


def clean_waves(cache, cfg):
    return center_crop(cache["X"].float() / 32767.0, cfg)


def mix_at_snr(signal, noise, snr_db):
    """Scale `noise` so that signal power / noise power = snr_db, then add."""
    ps = signal.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-10)
    pn = noise.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-10)
    return signal + noise * torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def phone_band(x, sr, lo=300.0, hi=3400.0):
    """Keep only 300-3400 Hz (telephone band) with an FFT mask - our own band-pass filter."""
    X = torch.fft.rfft(x, dim=1)
    f = torch.fft.rfftfreq(x.shape[1], d=1.0 / sr)
    X[:, (f < lo) | (f > hi)] = 0
    return torch.fft.irfft(X, n=x.shape[1], dim=1)


def bootstrap_ci(y, p, thr, n=1000, seed=SEED):
    """95% interval for AUC and balanced accuracy by resampling the test set with replacement."""
    g = torch.Generator().manual_seed(seed)
    y, p = torch.as_tensor(y), torch.as_tensor(p)
    aucs, bals = [], []
    for _ in range(n):
        idx = torch.randint(0, len(y), (len(y),), generator=g)
        yy, pp = y[idx], p[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aucs.append(roc_auc(yy, pp))
        bals.append(binary_report(yy, pp, thr)["balanced_accuracy"])
    q = lambda v: (sorted(v)[int(0.025 * len(v))], sorted(v)[int(0.975 * len(v)) - 1])
    return {"auc_95ci": q(aucs), "balanced_acc_95ci": q(bals)}


def roc_points(y, p):
    """(false-positive rate, true-positive rate) for every threshold, for the ROC figure."""
    y, p = torch.as_tensor(y), torch.as_tensor(p)
    order = torch.argsort(p, descending=True)
    ys = y[order].float()
    tp, fp = torch.cumsum(ys, 0), torch.cumsum(1 - ys, 0)
    tpr = torch.cat([torch.zeros(1), tp / ys.sum()])
    fpr = torch.cat([torch.zeros(1), fp / (1 - ys).sum()])
    return fpr.tolist(), tpr.tolist()


@torch.no_grad()
def latency(det, dw, cfg, device, runs=200):
    fx = FeatureExtractor(cfg).to(device)
    x = torch.randn(1, int(cfg.clip_seconds * cfg.sample_rate), device=device) * 0.1
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    for _ in range(20):                                     # warm-up
        dw(fx.log_mel(x)); det(fx.log_mel(x))
    parts = {"features_ms": [], "detector_ms": [], "drywet_ms": [], "total_ms": []}
    for _ in range(runs):
        sync(); t0 = time.perf_counter()
        f = fx.log_mel(x); sync(); t1 = time.perf_counter()
        det(f); sync(); t2 = time.perf_counter()
        dw(f); sync(); t3 = time.perf_counter()
        parts["features_ms"].append((t1 - t0) * 1e3)
        parts["detector_ms"].append((t2 - t1) * 1e3)
        parts["drywet_ms"].append((t3 - t2) * 1e3)
        parts["total_ms"].append((t3 - t0) * 1e3)
    res = {k: {"median": statistics.median(v), "p95": sorted(v)[int(0.95 * len(v)) - 1]}
           for k, v in parts.items()}
    res["real_time_factor"] = res["total_ms"]["median"] / (cfg.clip_seconds * 1000)
    return res


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--drywet", default=os.path.join("checkpoints", "drywet_finetune_best.pt"))
    ap.add_argument("--detector", default=os.path.join("checkpoints", "pretrain_best.pt"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--k-windows", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    random.seed(SEED); torch.manual_seed(SEED)

    dw, dw_ck = load_model(a.drywet, device)
    det, det_ck = load_model(a.detector, device)
    cfg = FeatureConfig.from_dict(dw_ck["feature_config"])
    fx = FeatureExtractor(cfg).to(device)
    thr_dw, thr_det = dw_ck["threshold"], det_ck["threshold"]
    print(f"device {device} | dry/wet threshold {thr_dw:.2f} (from validation) | "
          f"detector threshold {thr_det:.2f} (from validation)")

    man = lambda s: os.path.join(a.data, "manifests", s)
    cache = lambda s: os.path.join(a.data, "cache", s.replace(".csv", ".pt"))
    dw_test = build_cache(man("drywet_test.csv"), a.data, cfg, cache("drywet_test.csv"))
    det_test = build_cache(man("detect_test.csv"), a.data, cfg, cache("detect_test.csv"))
    y_dw, y_det = dw_test["y"], det_test["y"]
    w_dw, w_det = clean_waves(dw_test, cfg), clean_waves(det_test, cfg)

    # ---- 1. headline results on the clean test set
    p_dw = predict(dw, fx, w_dw, device)
    p_det = predict(det, fx, w_det, device)
    r_dw = binary_report(y_dw, p_dw, thr_dw, ("dry", "wet"))
    r_det = binary_report(y_det, p_det, thr_det, ("non-cough", "cough"))
    ci_dw = bootstrap_ci(y_dw, p_dw, thr_dw)
    ci_det = bootstrap_ci(y_det, p_det, thr_det)
    always_dry = float((y_dw == 0).float().mean())
    print(f"\n=== TEST SET: dry vs wet ({len(y_dw)} recordings) ===\n" + format_report(r_dw))
    print(f"95% CI: AUC {ci_dw['auc_95ci'][0]:.3f}-{ci_dw['auc_95ci'][1]:.3f} | "
          f"balanced acc {ci_dw['balanced_acc_95ci'][0]:.3f}-{ci_dw['balanced_acc_95ci'][1]:.3f} | "
          f"'always dry' accuracy would be {always_dry:.3f}")
    print(f"\n=== TEST SET: cough detector ({len(y_det)} recordings) ===\n" + format_report(r_det))

    # ---- 2. robustness
    g = torch.Generator().manual_seed(SEED)
    bg = w_det[y_det == 0]                                   # real non-cough recordings as noise
    pick = lambda n: bg[torch.randint(0, len(bg), (n,), generator=g)]
    conditions = [("clean (baseline)", None, None)]
    conditions += [(f"white noise {s} dB", "white", s) for s in (30, 20, 10, 5, 0)]
    conditions += [(f"background noise {s} dB", "background", s) for s in (20, 10, 5)]
    conditions += [("phone band 300-3400 Hz", "phone", None)]
    rows = []
    print("\n=== ROBUSTNESS (test set) ===")
    print(f"{'condition':26s} | dry/wet AUC  bal.acc  wet recall | detector AUC  cough recall  non-cough recall")
    for name, kind, snr in conditions:
        def corrupt(w):
            if kind == "white":
                return mix_at_snr(w, torch.randn(w.shape, generator=g), snr)
            if kind == "background":
                return mix_at_snr(w, pick(len(w)), snr)
            if kind == "phone":
                return phone_band(w, cfg.sample_rate)
            return w
        pd = predict(dw, fx, corrupt(w_dw), device)
        pc = predict(det, fx, corrupt(w_det), device)
        rd = binary_report(y_dw, pd, thr_dw)
        rc = binary_report(y_det, pc, thr_det)
        row = {"condition": name, "drywet_auc": rd["roc_auc"], "drywet_bal_acc": rd["balanced_accuracy"],
               "drywet_wet_recall": rd["recall"], "drywet_dry_recall": rd["specificity"],
               "detector_auc": rc["roc_auc"], "detector_cough_recall": rc["recall"],
               "detector_noncough_recall": rc["specificity"]}
        rows.append(row)
        print(f"{name:26s} | {rd['roc_auc']:.3f}        {rd['balanced_accuracy']:.3f}    "
              f"{rd['recall']:.3f}      | {rc['roc_auc']:.3f}         {rc['recall']:.3f}         "
              f"{rc['specificity']:.3f}")
    with open(os.path.join(a.out, "robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows])

    # ---- 3. multi-cough summary: average only detector-confirmed windows
    items = read_manifest(man("drywet_test.csv"))
    p_multi, n_used = [], []
    with torch.no_grad():
        for _, rel, _ in items:
            x, _ = load_wav(os.path.join(a.data, rel))
            wins = torch.stack([peak_normalize(v) for v in top_windows(x, cfg, a.k_windows)]).to(device)
            feats = fx.log_mel(wins)
            pc = torch.softmax(det(feats), dim=1)[:, 1]
            pw = torch.softmax(dw(feats), dim=1)[:, 1]
            keep = pc >= thr_det
            if keep.sum() == 0:
                keep = torch.zeros_like(keep); keep[0] = True       # fall back to the loudest
            p_multi.append(pw[keep].mean().item())
            n_used.append(int(keep.sum()))
    r_multi = binary_report(y_dw, p_multi, thr_dw)
    print(f"\n=== MULTI-COUGH SUMMARY (mean of detector-confirmed windows, up to {a.k_windows}) ===")
    print(f"AUC {r_multi['roc_auc']:.3f} (single loudest window: {r_dw['roc_auc']:.3f}) | "
          f"bal.acc {r_multi['balanced_accuracy']:.3f} | windows used per recording: "
          f"mean {statistics.mean(n_used):.1f}")

    # ---- 4. latency
    lat = {"cpu": latency(det.cpu(), dw.cpu(), cfg, torch.device("cpu"))}
    if torch.cuda.is_available():
        lat["cuda"] = latency(det.to("cuda"), dw.to("cuda"), cfg, torch.device("cuda"))
    print("\n=== LATENCY per 1-second window (median / 95th percentile, ms) ===")
    for dev, r in lat.items():
        print(f"{dev:4s}: features {r['features_ms']['median']:.2f}/{r['features_ms']['p95']:.2f}  "
              f"detector {r['detector_ms']['median']:.2f}/{r['detector_ms']['p95']:.2f}  "
              f"dry/wet {r['drywet_ms']['median']:.2f}/{r['drywet_ms']['p95']:.2f}  "
              f"total {r['total_ms']['median']:.2f}/{r['total_ms']['p95']:.2f}  "
              f"real-time factor {r['real_time_factor']:.4f}")

    # ---- save
    results = {"drywet_test": r_dw, "drywet_ci": ci_dw, "always_dry_accuracy": always_dry,
               "detector_test": r_det, "detector_ci": ci_det, "multi_cough": r_multi,
               "multi_cough_mean_windows": statistics.mean(n_used),
               "thresholds": {"drywet": thr_dw, "detector": thr_det}}
    with open(os.path.join(a.out, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(a.out, "latency.json"), "w") as f:
        json.dump(lat, f, indent=2)
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("DRY/WET TEST\n" + format_report(r_dw) + f"\n95% CI {ci_dw}\n\nDETECTOR TEST\n"
                + format_report(r_det) + "\n")

    # ---- figures (optional, matplotlib only draws)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for (y, p, lab, col) in ((y_dw, p_dw, f"dry vs wet (AUC {r_dw['roc_auc']:.3f})", "#2a6fb0"),
                                 (y_det, p_det, f"cough detector (AUC {r_det['roc_auc']:.3f})", "#9aa4ae")):
            fpr, tpr = roc_points(y, p)
            ax.plot(fpr, tpr, color=col, lw=2, label=lab)
        ax.plot([0, 1], [0, 1], ls="--", color="#cccccc", lw=1, label="guessing (AUC 0.5)")
        ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate")
        ax.set_title("ROC curves on the test set", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "figures", "roc_test.png"), dpi=150)
        plt.close(fig)
        print(f"\nFigures written to {os.path.join(a.out, 'figures')}")
    except ImportError:
        print("\n(matplotlib not installed: figures skipped)")
    print(f"Results written to {a.out}/")


if __name__ == "__main__":
    main()
