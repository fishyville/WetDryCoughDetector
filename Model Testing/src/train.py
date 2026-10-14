"""
Step 4 - Training.

Three commands (run from the "Model Testing" folder, venv active):

  python src/train.py pretrain                 # phase 1: cough vs non-cough (~18.5k clips)
  python src/train.py finetune                 # phase 2: dry vs wet, starting from phase 1
  python src/train.py finetune --scratch       # experiment: dry vs wet WITHOUT pretraining

What happens:
  1. CACHE   each clip's loudest 1.2 s is cut out once and stored as 16-bit numbers in
             data/cache/<name>.pt, so every epoch doesn't re-read thousands of WAV files.
  2. AUGMENT (training only, a new random version every epoch)
             - time shift : take a random 1.0 s window inside the 1.2 s
             - noise      : with p = 0.5, add white noise at 10-40 dB SNR
             - SpecAugment: with p = 0.5, blank one random frequency strip and one time strip
  3. TRAIN   AdamW optimiser, cosine learning-rate schedule, class-weighted cross-entropy
             (the rarer class counts more, so the model can't win by always saying "dry").
  4. SELECT  after every epoch the model is scored on the VALIDATION set; the best epoch
             (highest ROC-AUC) is saved; training stops early if it hasn't improved for a while.

Outputs:
  checkpoints/<run>_best.pt     model weights + feature config + chosen threshold + val metrics
  checkpoints/<run>_history.csv one row per epoch (for the training-curve figure in Step 5)

Only PyTorch + Python standard library + our own features.py / model.py / metrics.py.
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureConfig, FeatureExtractor, load_wav, loudest_window  # noqa: E402
from model import CoughNet, count_parameters                                    # noqa: E402
from metrics import binary_report, best_threshold, format_report                 # noqa: E402

CACHE_SECONDS = 1.2


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(r["uuid"], r["path"], int(r["label"])) for r in csv.DictReader(f)]


def build_cache(manifest_path, data_root, cfg, cache_path, cache_seconds=CACHE_SECONDS, verbose=True):
    """Cut the loudest `cache_seconds` of every clip once; store as int16 to save memory."""
    if os.path.exists(cache_path):
        c = torch.load(cache_path, weights_only=True)
        if verbose:
            print(f"  cache loaded: {cache_path} ({len(c['y'])} clips)")
        return c
    items = read_manifest(manifest_path)
    long_cfg = FeatureConfig(**{**cfg.to_dict(), "clip_seconds": cache_seconds})
    n = int(cache_seconds * cfg.sample_rate)
    X = torch.zeros(len(items), n, dtype=torch.int16)
    y = torch.zeros(len(items), dtype=torch.long)
    t0 = time.time()
    for i, (_, rel, label) in enumerate(items):
        x, sr = load_wav(os.path.join(data_root, rel))
        assert sr == cfg.sample_rate, f"{rel}: {sr} Hz"
        seg = loudest_window(x, long_cfg)
        peak = seg.abs().max()
        if peak > 1e-8:
            seg = seg * (0.9 / peak)                      # peak-normalise (as features.peak_normalize)
        X[i] = (seg * 32767).round().clamp(-32768, 32767).to(torch.int16)
        y[i] = label
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(items)}  ({time.time() - t0:.0f} s)")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    c = {"X": X, "y": y, "uuids": [u for u, _, _ in items]}
    torch.save(c, cache_path)
    print(f"  cache built: {cache_path} ({len(items)} clips, {time.time() - t0:.0f} s)")
    return c


def batches(n, batch_size, shuffle):
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        yield idx[i:i + batch_size]


# ---------------------------------------------------------------- augmentation
def augment_wave(x, cfg, noise_p=0.5, snr_db=(10.0, 40.0)):
    """x: [B, cache_len] float -> [B, clip_len]. Random 1 s crop + optional white noise."""
    B, L = x.shape
    n = int(cfg.clip_seconds * cfg.sample_rate)
    starts = torch.randint(0, L - n + 1, (B,))
    out = torch.stack([x[b, s:s + n] for b, s in enumerate(starts.tolist())])
    use = torch.rand(B, device=x.device) < noise_p
    if use.any():
        snr = torch.empty(B, device=x.device).uniform_(*snr_db)
        sig_pow = out.pow(2).mean(dim=1).clamp_min(1e-10)
        noise_pow = sig_pow / (10 ** (snr / 10))
        noise = torch.randn_like(out) * noise_pow.sqrt().unsqueeze(1)
        out = torch.where(use.unsqueeze(1), out + noise, out)
    return out


def center_crop(x, cfg):
    n = int(cfg.clip_seconds * cfg.sample_rate)
    s = (x.shape[1] - n) // 2
    return x[:, s:s + n]


def spec_augment(m, p=0.5, max_f=8, max_t=10):
    """m: [B, n_mels, T]. Blank one random mel strip and one time strip (set to the clip's mean)."""
    B, Fr, T = m.shape
    m = m.clone()
    for b in range(B):
        if random.random() < p:
            fill = m[b].mean()
            f = random.randint(1, max_f); f0 = random.randint(0, Fr - f)
            t = random.randint(1, max_t); t0 = random.randint(0, T - t)
            m[b, f0:f0 + f, :] = fill
            m[b, :, t0:t0 + t] = fill
    return m


# ---------------------------------------------------------------- one epoch
def run_epoch(model, fx, cache, cfg, device, batch_size, optimizer=None, scheduler=None,
              loss_fn=None, augment=False):
    """Train (if optimizer given) or evaluate. Returns (mean loss, true labels, P(class 1))."""
    training = optimizer is not None
    model.train(training)
    X, y = cache["X"], cache["y"]
    total, count, ys, ps = 0.0, 0, [], []
    with torch.set_grad_enabled(training):
        for idx in batches(len(y), batch_size, shuffle=training):
            wav = X[idx].to(device).float() / 32767.0
            wav = augment_wave(wav, cfg) if augment else center_crop(wav, cfg)
            feats = fx.log_mel(wav)
            if augment:
                feats = spec_augment(feats)
            target = y[idx].to(device)
            logits = model(feats)
            loss = loss_fn(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            total += loss.item() * len(idx)
            count += len(idx)
            ys.append(target.cpu())
            ps.append(torch.softmax(logits.detach(), dim=1)[:, 1].cpu())
    return total / count, torch.cat(ys), torch.cat(ps)


def class_weights(y):
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = torch.bincount(y, minlength=2).float().clamp_min(1)
    w = counts.sum() / (2 * counts)
    return w


def cosine_schedule(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ---------------------------------------------------------------- a training phase
def train_phase(model, fx, cfg, train_c, val_c, device, a, run_name, names, epochs, lr,
                freeze_epochs=0, verbose=True):
    os.makedirs(a.out, exist_ok=True)
    w = class_weights(train_c["y"]).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    if verbose:
        print(f"  class weights: {names[0]}={w[0]:.2f}, {names[1]}={w[1]:.2f}")

    steps_per_epoch = math.ceil(len(train_c["y"]) / a.batch_size)
    hist_path = os.path.join(a.out, f"{run_name}_history.csv")
    best_path = os.path.join(a.out, f"{run_name}_best.pt")
    best_auc, bad_epochs = -1.0, 0

    with open(hist_path, "w", newline="") as hf:
        hw = csv.writer(hf)
        hw.writerow(["epoch", "phase", "lr", "train_loss", "val_loss", "val_auc", "val_f1",
                     "val_balanced_acc", "val_recall", "val_precision", "threshold", "seconds"])

        # optional warm-up: train only the new head while the backbone stays fixed
        stages = []
        if freeze_epochs > 0:
            stages.append(("head-only", freeze_epochs, lr * 3, True))
        stages.append(("full", epochs, lr, False))

        epoch_no = 0
        for stage, n_ep, stage_lr, frozen in stages:
            model.freeze_backbone(frozen)
            params = [p for p in model.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(params, lr=stage_lr, weight_decay=a.weight_decay)
            sched = cosine_schedule(opt, n_ep * steps_per_epoch,
                                    warmup_steps=0 if frozen else steps_per_epoch)
            for _ in range(n_ep):
                epoch_no += 1
                t0 = time.time()
                cur_lr = opt.param_groups[0]["lr"]          # learning rate at the start of this epoch
                tr_loss, _, _ = run_epoch(model, fx, train_c, cfg, device, a.batch_size,
                                          opt, sched, loss_fn, augment=True)
                va_loss, vy, vp = run_epoch(model, fx, val_c, cfg, device, a.batch_size,
                                            loss_fn=loss_fn)
                thr, _ = best_threshold(vy, vp, "f1")
                r = binary_report(vy, vp, thr, names)
                secs = time.time() - t0
                hw.writerow([epoch_no, stage, f"{cur_lr:.2e}", f"{tr_loss:.4f}", f"{va_loss:.4f}",
                             f"{r['roc_auc']:.4f}", f"{r['f1']:.4f}", f"{r['balanced_accuracy']:.4f}",
                             f"{r['recall']:.4f}", f"{r['precision']:.4f}", f"{thr:.2f}",
                             f"{secs:.1f}"])
                hf.flush()
                mark = ""
                if r["roc_auc"] > best_auc:
                    best_auc, bad_epochs, mark = r["roc_auc"], 0, "  * best"
                    torch.save({"model": model.state_dict(), "feature_config": cfg.to_dict(),
                                "threshold": thr, "class_names": list(names),
                                "epoch": epoch_no, "val_report": r,
                                "channels": a.channels}, best_path)
                elif stage == "full":
                    bad_epochs += 1
                if verbose:
                    print(f"  ep {epoch_no:2d} [{stage:9s}] loss {tr_loss:.3f}/{va_loss:.3f}  "
                          f"AUC {r['roc_auc']:.3f}  F1 {r['f1']:.3f}  bal.acc "
                          f"{r['balanced_accuracy']:.3f}  thr {thr:.2f}  {secs:.0f}s{mark}")
                if stage == "full" and bad_epochs >= a.patience:
                    if verbose:
                        print(f"  early stop: no improvement for {a.patience} epochs")
                    break

    ck = torch.load(best_path, weights_only=True)
    if verbose:
        print(f"\n  best epoch {ck['epoch']} -> {best_path}")
        print("  validation, best epoch:\n    " + format_report(ck["val_report"]).replace("\n", "\n    "))
    return best_path


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["pretrain", "finetune"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--init", default=os.path.join("checkpoints", "pretrain_best.pt"),
                    help="finetune: checkpoint to start from")
    ap.add_argument("--scratch", action="store_true", help="finetune without pretraining")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--freeze-epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--channels", type=int, nargs=4, default=[16, 32, 64, 128])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device(a.device)
    cfg = FeatureConfig()
    fx = FeatureExtractor(cfg).to(device)
    man = lambda s: os.path.join(a.data, "manifests", s)
    cache = lambda s: os.path.join(a.data, "cache", s.replace(".csv", ".pt"))
    print(f"device: {device}")

    if a.phase == "pretrain":
        print("Phase 1: cough vs non-cough")
        tr = build_cache(man("detect_train.csv"), a.data, cfg, cache("detect_train.csv"))
        va = build_cache(man("detect_val.csv"), a.data, cfg, cache("detect_val.csv"))
        model = CoughNet(2, tuple(a.channels)).to(device)
        print(f"  model parameters: {count_parameters(model):,}")
        train_phase(model, fx, cfg, tr, va, device, a, "pretrain", ("non-cough", "cough"),
                    epochs=a.epochs or 20, lr=a.lr or 1e-3, freeze_epochs=0)

    else:
        print("Phase 2: dry vs wet" + (" (from scratch)" if a.scratch else " (fine-tune)"))
        tr = build_cache(man("drywet_train.csv"), a.data, cfg, cache("drywet_train.csv"))
        va = build_cache(man("drywet_val.csv"), a.data, cfg, cache("drywet_val.csv"))
        model = CoughNet(2, tuple(a.channels)).to(device)
        if a.scratch:
            run, freeze, lr = "drywet_scratch", 0, a.lr or 1e-3
        else:
            if not os.path.exists(a.init):
                sys.exit(f"{a.init} not found: run `python src/train.py pretrain` first "
                         "(or use --scratch).")
            ck = torch.load(a.init, map_location=device, weights_only=True)
            model.load_state_dict(ck["model"])
            model.replace_head(2)
            model.to(device)
            print(f"  loaded backbone from {a.init} (epoch {ck['epoch']}), new head")
            run, freeze, lr = "drywet_finetune", a.freeze_epochs, a.lr or 3e-4
        train_phase(model, fx, cfg, tr, va, device, a, run, ("dry", "wet"),
                    epochs=a.epochs or 40, lr=lr, freeze_epochs=freeze)


if __name__ == "__main__":
    main()
