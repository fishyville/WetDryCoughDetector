"""
Visualise what features.py does, for understanding and for the report's figures.

NOT part of the system: this script only draws pictures. It uses matplotlib for plotting,
while all audio processing still comes from our own src/features.py.

Figures written to analysis/figures/:
  fig1_pipeline.png            one cough through every stage of features.py
  fig2_dry_vs_wet.png          log-mel of dry vs wet coughs (what the CNN sees)
  fig3_cough_vs_noncough.png   log-mel of coughs vs non-coughs (the pretraining task)
  fig4_class_average.png       average dry vs average wet log-mel + their difference

Usage (from the "Model Testing" folder, venv active):
  pip install matplotlib
  python analysis/visualize_features.py
"""

import argparse
import csv
import os
import random
import sys

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from features import (FeatureConfig, FeatureExtractor, load_wav, loudest_window,  # noqa: E402
                      peak_normalize, mel_to_hz)

SEQ = "magma"      # sequential colour map: dark = quiet, bright = loud
DIV = "RdBu_r"     # diverging colour map for differences (red = more in wet, blue = more in dry)
LINE = "#2a6fb0"
HILITE = "#e8a33d"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.spines.top": False, "axes.spines.right": False})


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["uuid"], os.path.join(os.path.dirname(os.path.dirname(path)), r["path"]),
                 int(r["label"])) for r in csv.DictReader(f)]


def clip_of(path, cfg):
    x, sr = load_wav(path)
    return x, peak_normalize(loudest_window(x, cfg))


def normalised_logmel(fx, clip):
    """What the CNN actually sees: log-mel standardised to mean 0, std 1 (as in model.py)."""
    m = fx.log_mel(clip)
    return (m - m.mean()) / (m.std() + 1e-5)


def mel_axis(ax, cfg, n_ticks=5):
    """Label the mel-band axis with real frequencies in Hz."""
    from features import hz_to_mel
    mels = torch.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_mels + 2)[1:-1]
    idx = [int(i) for i in torch.linspace(0, cfg.n_mels - 1, n_ticks)]
    ax.set_yticks(idx)
    ax.set_yticklabels([f"{mel_to_hz(mels[i].item()) / 1000:.1f}k" for i in idx])


def show_spec(ax, spec, cfg, title, vmin=None, vmax=None, cmap=SEQ, mel=True):
    t_end = spec.shape[1] * cfg.hop_length / cfg.sample_rate
    im = ax.imshow(spec.numpy(), origin="lower", aspect="auto", cmap=cmap,
                   extent=[0, t_end, -0.5, spec.shape[0] - 0.5], vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left")
    ax.set_xlabel("time (s)")
    if mel:
        mel_axis(ax, cfg)
        ax.set_ylabel("frequency (Hz, mel-spaced)")
    return im


# ---------------------------------------------------------------- figure 1
def fig_pipeline(path, fx, out):
    cfg = fx.cfg
    full, clip = clip_of(path, cfg)
    sr = cfg.sample_rate

    # where was the loudest window taken from?
    n = int(cfg.clip_seconds * sr)
    best, start = -1.0, 0
    for s in range(0, max(1, full.numel() - n + 1), cfg.hop_length):
        e = (full[s:s + n] ** 2).sum().item()
        if e > best:
            best, start = e, s

    pre = fx.pre_emphasize(clip)
    power = fx.power_spectrogram(pre)
    power_db = 10 * torch.log10(power + 1e-10)
    logmel = fx.log_mel(clip)
    mfcc = fx.mfcc(clip)

    fig, axs = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
    fig.suptitle("features.py, step by step, on one cough", fontsize=13, fontweight="bold")

    a = axs[0, 0]
    t = torch.arange(full.numel()) / sr
    a.plot(t, full, color=LINE, lw=0.5)
    a.axvspan(start / sr, (start + n) / sr, color=HILITE, alpha=0.3, lw=0)
    a.set_title(f"① Full recording ({full.numel() / sr:.1f} s): loudest 1 s highlighted", loc="left")
    a.set_xlabel("time (s)"); a.set_ylabel("amplitude")

    a = axs[0, 1]
    tc = torch.arange(clip.numel()) / sr
    a.plot(tc, clip, color="#9aa4ae", lw=0.6, label="after peak-normalise")
    a.plot(tc, pre, color=LINE, lw=0.6, label="after pre-emphasis")
    a.set_title("② The 1-second clip, before and after pre-emphasis", loc="left")
    a.set_xlabel("time (s)"); a.set_ylabel("amplitude"); a.legend(loc="upper right", frameon=False)

    a = axs[1, 0]
    t_end = power.shape[1] * cfg.hop_length / sr
    im = a.imshow(power_db.numpy(), origin="lower", aspect="auto", cmap=SEQ,
                  extent=[0, t_end, 0, sr / 2], vmin=power_db.max().item() - 80)
    a.set_title(f"③ STFT power spectrogram: {power.shape[0]} bins × {power.shape[1]} frames",
                loc="left")
    a.set_xlabel("time (s)"); a.set_ylabel("frequency (Hz, linear)")
    fig.colorbar(im, ax=a, label="power (dB)")

    a = axs[1, 1]
    freqs = torch.linspace(0, sr / 2, cfg.n_fft // 2 + 1)
    for i in range(cfg.n_mels):
        a.plot(freqs, fx.mel_fb[i], color=LINE, lw=0.8, alpha=0.9 if i % 2 == 0 else 0.45)
    a.set_title(f"④ Mel filterbank: {cfg.n_mels} triangles, narrow at low Hz, wide at high Hz",
                loc="left")
    a.set_xlabel("frequency (Hz)"); a.set_ylabel("filter weight")

    a = axs[2, 0]
    im = show_spec(a, logmel, cfg, f"⑤ Log-mel spectrogram: {logmel.shape[0]} × {logmel.shape[1]}"
                                   "  ← the CNN input")
    fig.colorbar(im, ax=a, label="log energy")

    a = axs[2, 1]
    im = show_spec(a, mfcc, cfg, f"⑥ MFCC: {mfcc.shape[0]} coefficients × {mfcc.shape[1]} frames",
                   cmap=DIV, mel=False,
                   vmin=-mfcc[1:].abs().max().item(), vmax=mfcc[1:].abs().max().item())
    a.set_ylabel("coefficient")
    fig.colorbar(im, ax=a, label="value (c0 clipped)")

    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figures 2 and 3
def fig_compare(groups, fx, out, title):
    """groups: [(row_name, [paths])]. Same colour scale everywhere so pictures are comparable."""
    cols = max(len(p) for _, p in groups)
    fig, axs = plt.subplots(len(groups), cols, figsize=(3.2 * cols, 2.9 * len(groups)),
                            constrained_layout=True, squeeze=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    im = None
    for r, (name, paths) in enumerate(groups):
        for c in range(cols):
            a = axs[r, c]
            if c >= len(paths):
                a.axis("off"); continue
            _, clip = clip_of(paths[c], fx.cfg)
            im = show_spec(a, normalised_logmel(fx, clip), fx.cfg,
                           f"{name} #{c + 1}", vmin=-2.5, vmax=3.0)
            if c > 0:
                a.set_ylabel(""); a.set_yticklabels([])
            if r < len(groups) - 1:
                a.set_xlabel("")
    fig.colorbar(im, ax=axs, shrink=0.8, label="normalised log energy (what the CNN sees)")
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_class_average(items, fx, out, limit=None):
    sums, counts = {}, {}
    for _, path, label in items[:limit] if limit else items:
        if not os.path.exists(path):
            continue
        _, clip = clip_of(path, fx.cfg)
        m = normalised_logmel(fx, clip)
        sums[label] = sums.get(label, 0) + m
        counts[label] = counts.get(label, 0) + 1
    if 0 not in counts or 1 not in counts:
        print("fig4 skipped: need both classes"); return
    dry, wet = sums[0] / counts[0], sums[1] / counts[1]
    diff = wet - dry
    lim = diff.abs().max().item()

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    fig.suptitle("Average spectrogram per class (training set)", fontsize=13, fontweight="bold")
    im = show_spec(axs[0], dry, fx.cfg, f"Average DRY (n = {counts[0]})", vmin=-1.5, vmax=2)
    show_spec(axs[1], wet, fx.cfg, f"Average WET (n = {counts[1]})", vmin=-1.5, vmax=2)
    fig.colorbar(im, ax=axs[:2], shrink=0.85, label="normalised log energy")
    im2 = show_spec(axs[2], diff, fx.cfg, "WET minus DRY", cmap=DIV, vmin=-lim, vmax=lim)
    fig.colorbar(im2, ax=axs[2], label="red = stronger in wet, blue = stronger in dry")
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default=os.path.join("analysis", "figures"))
    ap.add_argument("--n", type=int, default=3, help="examples per class in fig 2 and 3")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry", nargs="*", help="uuids to use as dry examples (optional)")
    ap.add_argument("--wet", nargs="*", help="uuids to use as wet examples (optional)")
    ap.add_argument("--noncough", nargs="*", help="uuids to use as non-cough examples (optional)")
    ap.add_argument("--average-limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fx = FeatureExtractor(FeatureConfig())
    rng = random.Random(a.seed)
    wav = lambda u: os.path.join(a.data, "wav", u + ".wav")

    drywet = read_manifest(os.path.join(a.data, "manifests", "drywet_train.csv"))
    dry = [wav(u) for u in a.dry] if a.dry else \
        [p for _, p, l in rng.sample([x for x in drywet if x[2] == 0], a.n)]
    wet = [wav(u) for u in a.wet] if a.wet else \
        [p for _, p, l in rng.sample([x for x in drywet if x[2] == 1], a.n)]

    fig_pipeline(wet[0], fx, os.path.join(a.out, "fig1_pipeline.png"))
    fig_compare([("Dry", dry), ("Wet", wet)], fx, os.path.join(a.out, "fig2_dry_vs_wet.png"),
                "Dry vs wet coughs: log-mel spectrograms (same colour scale)")

    if a.noncough:
        non = [wav(u) for u in a.noncough]
    else:
        det_path = os.path.join(a.data, "manifests", "detect_train.csv")
        non = [p for _, p, l in rng.sample([x for x in read_manifest(det_path) if x[2] == 0], a.n)] \
            if os.path.exists(det_path) else []
    if non:
        fig_compare([("Cough", (dry[:2] + wet[:1])[:len(non)]), ("Not a cough", non)], fx,
                    os.path.join(a.out, "fig3_cough_vs_noncough.png"),
                    "Cough vs not a cough: the pretraining task")

    fig_class_average(drywet, fx, os.path.join(a.out, "fig4_class_average.png"), a.average_limit)
    print("Figures written to", a.out)


if __name__ == "__main__":
    main()
