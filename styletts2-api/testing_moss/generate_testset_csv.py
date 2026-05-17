import os
import csv
import time
import uuid
import requests
import pandas as pd

# =========================================================
# CONFIG
# =========================================================

API_URL = "http://localhost:8001/synthesize"

INPUT_TEXT_FILE = "testset.txt"

OUTPUT_DIR = "generated_audio"

OUTPUT_CSV = "utmos_test_results.csv"

REFERENCE_AUDIO = None
# Example:
# "/workspace/rizoan/reference.wav"

# =========================================================
# CREATE OUTPUT DIR
# =========================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# LANGUAGE DETECTION
# =========================================================

def detect_language(text: str) -> str:
    if any("\u0980" <= ch <= "\u09FF" for ch in text):
        return "bn"
    return "en"

# =========================================================
# READ TEST SENTENCES
# =========================================================

with open(INPUT_TEXT_FILE, "r", encoding="utf-8") as f:
    sentences = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(sentences)} sentences")

# =========================================================
# RESULTS
# =========================================================

results = []

# =========================================================
# INFERENCE LOOP
# =========================================================

for idx, text in enumerate(sentences, start=1):

    print("=" * 60)
    print(f"[{idx}/{len(sentences)}]")
    print(text)

    request_id = str(uuid.uuid4())[:8]

    audio_filename = f"sample_{idx:03d}.wav"

    audio_path = os.path.join(
        OUTPUT_DIR,
        audio_filename
    )

    # =====================================================
    # REQUEST DATA
    # =====================================================

    data = {
        "text": text,
        "alpha": 0.3,
        "beta": 0.7,
        "diffusion_steps": 5,
        "embedding_scale": 1.0,
    }

    # =====================================================
    # OPTIONAL REFERENCE AUDIO
    # =====================================================

    files = {}

    if REFERENCE_AUDIO is not None:

        files["reference_audio"] = (
            os.path.basename(REFERENCE_AUDIO),
            open(REFERENCE_AUDIO, "rb"),
            "audio/wav",
        )

    # =====================================================
    # SEND REQUEST
    # =====================================================

    try:

        t0 = time.time()

        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=300,
        )

        request_time = time.time() - t0

        response.raise_for_status()

    except Exception as e:

        print(f"FAILED: {e}")

        results.append({
            "id": idx,
            "language": detect_language(text),
            "text": text,
            "generated_audio": "",
            "utmos_score": "",
            "audio_duration_sec": "",
            "inference_time_sec": "",
            "rtf": "",
            "request_status": "failed",
            "error": str(e),
        })

        continue

    # =====================================================
    # SAVE AUDIO
    # =====================================================

    with open(audio_path, "wb") as f:
        f.write(response.content)

    # =====================================================
    # READ HEADERS
    # =====================================================

    headers = response.headers

    utmos_score = headers.get("X-MOS", "")
    duration = headers.get("X-Duration-Seconds", "")
    inference_time = headers.get("X-Inference-Time", "")
    rtf = headers.get("X-RTF", "")

    # =====================================================
    # STORE RESULT
    # =====================================================

    results.append({
        "id": idx,
        "language": detect_language(text),
        "text": text,
        "generated_audio": audio_path,
        "utmos_score": utmos_score,
        "audio_duration_sec": duration,
        "inference_time_sec": inference_time,
        "rtf": rtf,
        "request_status": "success",
        "error": "",
    })

    print(f"UTMOS: {utmos_score}")
    print(f"Saved: {audio_path}")

# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

print("=" * 60)
print("DONE")
print(f"CSV saved to: {OUTPUT_CSV}")
print(f"Generated audio dir: {OUTPUT_DIR}")