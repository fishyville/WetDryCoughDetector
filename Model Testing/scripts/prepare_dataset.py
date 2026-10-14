"""
Step 1 - Prepare the COUGHVID dataset.

What it does:
  1. Reads metadata_compiled.csv (Python standard library only, no pandas).
  2. Builds two labelled lists:
       - drywet : clean expert-labelled coughs, label = dry (0) / wet (1)
       - detect : cough (1) vs non-cough (0), from the cough_detected score
  3. Splits each list into train / val / test (70 / 15 / 15), stratified, fixed seed.
     Recordings in the dry/wet val+test sets are removed from the detector
     train set, so the pretraining step never sees your final test audio.
  4. Converts the needed .webm / .ogg files to 16 kHz mono 16-bit .wav with FFmpeg.
  5. Writes manifests:  data/manifests/{drywet,detect}_{train,val,test}.csv

Usage (from the "Model Testing" folder):
  python scripts/prepare_dataset.py --only drywet      # ~2k files, quick, do this first
  python scripts/prepare_dataset.py                    # everything (~27k files, several GB)

Only external tool used: FFmpeg (one-time data conversion).
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import wave
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

SAMPLE_RATE = 16000
COUGH_THRESHOLD = 0.8      # cough_detected >= 0.8 -> cough
NONCOUGH_THRESHOLD = 0.2   # cough_detected <= 0.2 -> not a cough
SPLITS = (0.70, 0.15, 0.15)
SEED = 42


# ---------------------------------------------------------------- labels
def majority_cough_type(row):
    """Majority vote over the 4 experts, ignoring 'unknown'. Returns 'dry', 'wet' or None."""
    votes = [row.get(f"cough_type_{i}", "") for i in range(1, 5)]
    votes = [v for v in votes if v in ("dry", "wet")]
    if not votes:
        return None
    counts = Counter(votes).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None  # tie between experts -> unreliable, skip
    return counts[0][0]


def has_bad_quality(row):
    """True if any expert marked the recording as poor quality or containing no cough."""
    return any(row.get(f"quality_{i}", "") in ("poor", "no_cough") for i in range(1, 5))


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_lists(metadata_path):
    drywet, detect = [], []
    with open(metadata_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uuid = row["uuid"]
            score = to_float(row.get("cough_detected"))
            if score is None:
                continue

            # dry / wet task
            ctype = majority_cough_type(row)
            if ctype and score >= COUGH_THRESHOLD and not has_bad_quality(row):
                drywet.append((uuid, 1 if ctype == "wet" else 0))

            # cough detection task
            if score >= COUGH_THRESHOLD:
                detect.append((uuid, 1))
            elif score <= NONCOUGH_THRESHOLD:
                detect.append((uuid, 0))
    return drywet, detect


def stratified_split(items, seed=SEED):
    """items: list of (uuid, label). Returns dict split_name -> list, same class ratio in each split."""
    rng = random.Random(seed)
    by_label = {}
    for uuid, label in items:
        by_label.setdefault(label, []).append((uuid, label))
    out = {"train": [], "val": [], "test": []}
    for label, group in by_label.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLITS[0])
        n_val = int(n * SPLITS[1])
        out["train"] += group[:n_train]
        out["val"] += group[n_train:n_train + n_val]
        out["test"] += group[n_train + n_val:]
    for k in out:
        rng.shuffle(out[k])
    return out


# ---------------------------------------------------------------- audio
def index_audio(coughvid_dir):
    """Map uuid -> audio file path, for ANY audio extension (.webm, .ogg, .wav, ...)."""
    index = {}
    for name in os.listdir(coughvid_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() in (".json", ".csv", ".txt", ""):
            continue
        index.setdefault(stem, os.path.join(coughvid_dir, name))
    return index


def convert_one(args):
    """Convert one file with FFmpeg. Returns (uuid, status)."""
    ffmpeg, src, dst, uuid = args
    if os.path.exists(dst) and os.path.getsize(dst) > 44:
        return uuid, "skipped"
    if src is None:
        return uuid, "missing"
    cmd = [ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", src,
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", dst]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst):
        return uuid, "failed"
    return uuid, "ok"


def wav_duration(path):
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coughvid-dir",
                    default=os.path.join("coughvid", "coughvid_20211012"))
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--only", choices=["drywet", "detect", "all"], default="all")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    a = ap.parse_args()

    meta = os.path.join(a.coughvid_dir, "metadata_compiled.csv")
    if not os.path.exists(meta):
        sys.exit(f"Cannot find {meta}. Check --coughvid-dir.")
    if not shutil.which(a.ffmpeg) and not os.path.exists(a.ffmpeg):
        sys.exit("FFmpeg not found. Install it and make sure `ffmpeg -version` works in this terminal.")

    wav_dir = os.path.join(a.out_dir, "wav")
    man_dir = os.path.join(a.out_dir, "manifests")
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)

    drywet, detect = build_lists(meta)
    tasks = {}
    if a.only in ("drywet", "all"):
        tasks["drywet"] = stratified_split(drywet)
    if a.only in ("detect", "all"):
        dw_split = stratified_split(drywet)
        held_out = {u for u, _ in dw_split["val"] + dw_split["test"]}
        tasks["detect"] = stratified_split([x for x in detect if x[0] not in held_out])

    # convert every needed file once
    needed = sorted({u for t in tasks.values() for s in t.values() for u, _ in s})
    print(f"Converting {len(needed)} files to 16 kHz mono WAV with {a.workers} workers...")
    audio = index_audio(a.coughvid_dir)
    print("Audio files found by extension:",
          dict(Counter(os.path.splitext(p)[1].lower() for p in audio.values())))
    jobs = [(a.ffmpeg, audio.get(u), os.path.join(wav_dir, u + ".wav"), u) for u in needed]
    status = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (u, s) in enumerate(ex.map(convert_one, jobs), 1):
            status[u] = s
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    print("Conversion:", dict(Counter(status.values())))
    missing = [u for u, st in status.items() if st in ("missing", "failed")]
    if missing:
        with open(os.path.join(a.out_dir, "missing_or_failed.txt"), "w") as f:
            f.write("\n".join(missing))
        print(f"  {len(missing)} uuids listed in {a.out_dir}/missing_or_failed.txt")

    # write manifests (only files that converted fine and are not empty)
    for task, splits in tasks.items():
        for split, items in splits.items():
            path = os.path.join(man_dir, f"{task}_{split}.csv")
            kept = Counter()
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["uuid", "path", "label", "duration_s"])
                for u, label in items:
                    if status.get(u) not in ("ok", "skipped"):
                        continue
                    wp = os.path.join("wav", u + ".wav")
                    dur = wav_duration(os.path.join(a.out_dir, wp))
                    if dur < 0.3:
                        continue
                    w.writerow([u, wp, label, f"{dur:.2f}"])
                    kept[label] += 1
            names = {"drywet": {0: "dry", 1: "wet"}, "detect": {0: "non-cough", 1: "cough"}}[task]
            print(f"{task:7s} {split:5s}: " + ", ".join(f"{names[k]}={v}" for k, v in sorted(kept.items())))

    print(f"\nDone. Manifests in {man_dir}")


if __name__ == "__main__":
    main()
