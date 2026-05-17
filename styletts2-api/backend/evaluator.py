# evaluator.py

import numpy as np
import librosa
import torch
import torchaudio

# =========================
# UTMOS MODEL
# =========================

utmos_model = torch.hub.load(
    "tarepan/SpeechMOS:v1.2.0",
    "utmos22_strong",
    trust_repo=True,
)

utmos_model.eval()

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
utmos_model.to(DEVICE)

SAMPLE_RATE = 24000


# =========================
# MAIN EVALUATION
# =========================

def evaluate_audio(audio: np.ndarray, text: str = "") -> dict:

    metrics = {}

    # =========================================================
    # UTMOS SCORE
    # =========================================================

    wav_tensor = torch.tensor(audio).float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        mos_score = utmos_model(wav_tensor, SAMPLE_RATE)

    metrics["mos"] = round(float(mos_score), 3)

    # =========================================================
    # SILENCE RATIO
    # =========================================================

    rms = librosa.feature.rms(y=audio)[0]

    silence_frames = np.sum(rms < 0.01)

    silence_ratio = silence_frames / len(rms)

    metrics["silence_ratio"] = round(float(silence_ratio), 3)

    # =========================================================
    # CLIPPING RATIO
    # =========================================================

    clipping = np.sum(np.abs(audio) >= 0.99)

    clipping_ratio = clipping / len(audio)

    metrics["clipping_ratio"] = round(float(clipping_ratio), 5)

    # =========================================================
    # NOISE ESTIMATION
    # =========================================================

    spectral_flatness = librosa.feature.spectral_flatness(y=audio)[0]

    noise_ratio = np.mean(spectral_flatness)

    metrics["noise_ratio"] = round(float(noise_ratio), 5)

    # =========================================================
    # SPEAKING RATE
    # =========================================================

    duration = len(audio) / SAMPLE_RATE

    words = len(text.split()) if text else 0

    speaking_rate = words / duration if duration > 0 else 0

    metrics["speaking_rate"] = round(float(speaking_rate), 3)

    return metrics