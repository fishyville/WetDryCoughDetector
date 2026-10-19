
import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch

from features import FeatureConfig, FeatureExtractor, peak_normalize
from model import CoughNet


# ============================================================
# 1. KONFIGURASI
# ============================================================

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "models"
CONFIG_PATH = MODEL_DIR / "config.json"

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Config tidak ditemukan: {CONFIG_PATH}\n"
        "Jalankan export.py --to-app terlebih dahulu."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

feature_config = FeatureConfig.from_dict(
    config["feature_config"]
)

SAMPLE_RATE = feature_config.sample_rate
WINDOW_SECONDS = float(
    config["streaming"]["window_seconds"]
)
HOP_SECONDS = float(
    config["streaming"]["hop_seconds"]
)

WINDOW_SAMPLES = int(WINDOW_SECONDS * SAMPLE_RATE)
HOP_SAMPLES = int(HOP_SECONDS * SAMPLE_RATE)

DEVICE = torch.device("cpu")


# ============================================================
# 2. MEMUAT MODEL
# ============================================================

def load_model(model_key):
    model_info = config[model_key]
    model_path = MODEL_DIR / model_info["file"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {model_path}"
        )

    model = CoughNet(
        n_classes=2,
        channels=tuple(config["channels"])
    )

    weights = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(weights)
    model.to(DEVICE)
    model.eval()

    return model


detector = load_model("detector")
classifier = load_model("drywet")
extractor = FeatureExtractor(feature_config).to(DEVICE)

detector_threshold = float(
    config["detector"]["threshold"]
)
classifier_threshold = float(
    config["drywet"]["threshold"]
)

detector_classes = config["detector"]["class_names"]
drywet_classes = config["drywet"]["class_names"]


# ============================================================
# 3. PREDIKSI AUDIO
# ============================================================

@torch.inference_mode()
def predict_cough(audio):
    """
    audio: NumPy array mono, float32, 16 kHz.
    """

    waveform = torch.from_numpy(
        np.asarray(audio, dtype=np.float32).copy()
    )

    # Normalisasi volume
    waveform = peak_normalize(waveform)

    # Ekstraksi fitur harus sama dengan saat training
    features = extractor.log_mel(waveform)
    features = features.unsqueeze(0).to(DEVICE)

    # Deteksi batuk
    detector_probs = detector.predict_proba(features)[0]
    cough_index = detector_classes.index("cough")

    cough_probability = float(
        detector_probs[cough_index].item()
    )

    if cough_probability < detector_threshold:
        return {
            "cough": False,
            "cough_probability": cough_probability,
            "type": None,
            "type_probability": None,
        }

    # Klasifikasi batuk kering atau basah
    drywet_probs = classifier.predict_proba(features)[0]

    # Menggunakan threshold yang dipilih saat training
    wet_index = drywet_classes.index("wet")
    dry_index = drywet_classes.index("dry")

    wet_probability = float(
        drywet_probs[wet_index].item()
    )

    if wet_probability >= classifier_threshold:
        cough_type = "Basah (wet)"
        type_probability = wet_probability
    else:
        cough_type = "Kering (dry)"
        type_probability = float(
            drywet_probs[dry_index].item()
        )

    return {
        "cough": True,
        "cough_probability": cough_probability,
        "type": cough_type,
        "type_probability": type_probability,
    }


# ============================================================
# 4. MENJALANKAN MIKROFON SECARA REAL-TIME
# ============================================================

def main():
    print("=" * 55)
    print("     WET/DRY COUGH DETECTOR")
    print("=" * 55)
    print(f"Sample rate : {SAMPLE_RATE} Hz")
    print(f"Window      : {WINDOW_SECONDS:.2f} detik")
    print(f"Interval    : {HOP_SECONDS:.2f} detik")
    print(f"Device      : {DEVICE}")
    print("\nModel berhasil dimuat.")
    print("Tekan Ctrl+C untuk berhenti.\n")

    audio_buffer = np.zeros(0, dtype=np.float32)
    last_report = 0.0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=HOP_SAMPLES,
        ) as stream:

            print("Mikrofon aktif. Silakan berbicara atau batuk.")

            while True:
                chunk, overflowed = stream.read(HOP_SAMPLES)

                if overflowed:
                    print(
                        "Peringatan: sebagian audio mungkin terlewat."
                    )

                chunk = chunk[:, 0].copy()

                audio_buffer = np.concatenate(
                    (audio_buffer, chunk)
                )

                # Simpan hanya satu window terakhir
                if len(audio_buffer) > WINDOW_SAMPLES:
                    audio_buffer = audio_buffer[-WINDOW_SAMPLES:]

                if len(audio_buffer) < WINDOW_SAMPLES:
                    continue

                now = time.monotonic()

                # Hindari pencetakan berulang terlalu cepat
                if now - last_report < HOP_SECONDS * 0.8:
                    continue

                last_report = now

                result = predict_cough(audio_buffer)

                if result["cough"]:
                    print(
                        f"Batuk terdeteksi | "
                        f"P(deteksi): "
                        f"{result['cough_probability']:.1%} | "
                        f"Prediksi: {result['type']} | "
                        f"P(kelas): "
                        f"{result['type_probability']:.1%}"
                    )
                else:
                    print(
                        f"Tidak terdeteksi batuk | "
                        f"P(batuk): "
                        f"{result['cough_probability']:.1%}"
                    )

    except KeyboardInterrupt:
        print("\nAplikasi dihentikan.")

    except Exception as exc:
        print(f"\nTerjadi kesalahan: {exc}")
        raise


if __name__ == "__main__":
    main()
