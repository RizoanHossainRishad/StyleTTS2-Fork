"""
main.py — FastAPI server for StyleTTS2.

Endpoints
---------
POST /synthesize          — Basic TTS (inference)
POST /synthesize/longform — Long-form / chained TTS (LFinference)
POST /synthesize/style    — Style-transfer TTS (STinference)
GET  /health              — Liveness check
GET  /docs                — Swagger UI (auto-generated)
"""

import io
import logging
import os
import re
import time
import uuid
from typing import Annotated, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from evaluator import evaluate_audio

# ── Default reference audio (used when caller does not upload one) ────────────
DEFAULT_REF_AUDIO = (
    #"/workspace/rizoan/StyleTTS2-Fork/styletts2-api/backend/sample_refs/"
    "/home/user01/Desktop/TTS/Backup/StyleTTS2-Fork/styletts2-api/backend/sample_refs/"
    #"Record (online-voice-recorder.com).wav"
    #"teacher_000_20260104_080449.wav"
    "train_bengalimale_03195.wav"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Engine (model loading happens at import time) ─────────────────────────────
logger.info("Importing StyleTTS2 engine …")
from engine import compute_style, inference, LFinference, STinference  # noqa: E402
#from engine2 import compute_style, inference, LFinference, STinference  # noqa: E402

logger.info("Engine ready.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="StyleTTS2 API",
    description=(
        "A REST API wrapping StyleTTS2 for high-quality, style-controllable "
        "text-to-speech synthesis.  Supports English and Bengali input text.\n\n"
        "Reference audio is **optional** — if omitted, a built-in default voice is used."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_RATE = 24_000


# ── Shared helpers ────────────────────────────────────────────────────────────

def _wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Encode a NumPy float32 array as a WAV byte-stream."""
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


""" def _wav_response(audio: np.ndarray, request_id: str, used_default: bool) -> Response:
    headers = {
        "X-Request-Id": request_id,
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Num-Samples": str(len(audio)),
        "X-Duration-Seconds": f"{len(audio) / SAMPLE_RATE:.3f}",
        "X-Used-Default-Reference": "true" if used_default else "false",
    }
    return Response(
        content=_wav_bytes(audio),
        media_type="audio/wav",
        headers=headers,
    ) """

def _wav_response(
    audio,
    request_id,
    used_default,
    inference_time=None,
    rtf=None,
    metrics=None
):
    headers = {
        "X-Request-Id": request_id,
        "X-Sample-Rate": str(SAMPLE_RATE),
        "X-Num-Samples": str(len(audio)),
        "X-Duration-Seconds": f"{len(audio) / SAMPLE_RATE:.3f}",
        "X-Used-Default-Reference": "true" if used_default else "false",
    }

    if inference_time is not None:
        headers["X-Inference-Time"] = f"{inference_time:.3f}"

    if rtf is not None:
        headers["X-RTF"] = f"{rtf:.4f}"

    if metrics is not None:

        mos = metrics.get("mos")
        silence_ratio = metrics.get("silence_ratio")
        noise_ratio = metrics.get("noise_ratio")
        clipping_ratio = metrics.get("clipping_ratio")
        speaking_rate = metrics.get("speaking_rate")

        headers["X-MOS"] = f"{mos:.3f}" if mos is not None else ""
        headers["X-Silence-Ratio"] = f"{silence_ratio:.4f}" if silence_ratio is not None else ""
        headers["X-Noise-Ratio"] = f"{noise_ratio:.4f}" if noise_ratio is not None else ""
        headers["X-Clipping-Ratio"] = f"{clipping_ratio:.4f}" if clipping_ratio is not None else ""
        headers["X-Speaking-Rate"] = f"{speaking_rate:.4f}" if speaking_rate is not None else ""

    return Response(
        content=_wav_bytes(audio),
        media_type="audio/wav",
        headers=headers,
    )


async def _resolve_ref_audio(
    reference_audio: Optional[UploadFile],
    request_id: str,
) -> tuple[str, bool, bool]:
    """
    Resolve the reference audio to a file path.

    Returns
    -------
    path : str
        Absolute path to a WAV file the engine can read.
    is_tmp : bool
        True if *path* is a temp file that should be deleted after use.
    used_default : bool
        True if the built-in default was used (no upload provided).
    """
    if reference_audio is not None and reference_audio.filename:
        tmp_path = f"/tmp/ref_{request_id}.wav"
        content = await reference_audio.read()
        with open(tmp_path, "wb") as fh:
            fh.write(content)
        logger.info("[%s] Using uploaded reference audio (%s)", request_id, reference_audio.filename)
        return tmp_path, True, False

    # Fall back to the default reference
    if not os.path.exists(DEFAULT_REF_AUDIO):
        raise HTTPException(
            status_code=500,
            detail=(
                "No reference audio uploaded and the default reference file "
                f"was not found at: {DEFAULT_REF_AUDIO}"
            ),
        )
    logger.info("[%s] No upload — using default reference audio", request_id)
    return DEFAULT_REF_AUDIO, False, True


def _cleanup(path: str, is_tmp: bool) -> None:
    if is_tmp and os.path.exists(path):
        os.remove(path)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe — returns 200 when the server (and models) are ready."""
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sample_rate": SAMPLE_RATE,
        "default_reference_audio_present": os.path.exists(DEFAULT_REF_AUDIO),
    }


# ── /synthesize ───────────────────────────────────────────────────────────────

@app.post(
    "/synthesize",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
    tags=["Synthesis"],
    summary="Basic TTS inference",
)
async def synthesize(
    text: Annotated[str, Form(description="Text to synthesise (English or Bengali).")],
    reference_audio: UploadFile | None = File(
        default=None,
        description="Reference WAV providing the voice style. Optional."
    ),
    alpha: Annotated[
        float,
        Form(description="Reference prosody weight (0–1). Higher = more like reference."),
    ] = 0.3,
    beta: Annotated[
        float,
        Form(description="Reference style weight (0–1). Higher = more like reference."),
    ] = 0.7,
    diffusion_steps: Annotated[
        int,
        Form(description="Diffusion denoising steps. More steps → higher quality, slower."),
    ] = 5,
    embedding_scale: Annotated[
        float,
        Form(description="Classifier-free guidance scale. Higher → stronger style signal."),
    ] = 1.0,
):
    """
    Synthesise *text* in the voice style extracted from *reference_audio*.

    **reference_audio is optional.** When omitted, a built-in default voice is
    used automatically.  The response header ``X-Used-Default-Reference`` will
    be ``true`` in that case.

    Returns a 24 kHz mono PCM-16 WAV file.
    """
    request_id = str(uuid.uuid4())
    logger.info("[%s] /synthesize  text=%r  steps=%d", request_id, text[:60], diffusion_steps)

    if not text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    ref_path, is_tmp, used_default = await _resolve_ref_audio(reference_audio, request_id)
    try:
        ref_s = compute_style(ref_path)
        t0 = time.time()
        audio = inference(
            text=text,
            ref_s=ref_s,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )
        metrics = evaluate_audio(audio, text)
        inference_time = time.time() - t0
        rtf = (time.time() - t0) / (len(audio) / SAMPLE_RATE)
        logger.info("[%s] done  RTF=%.4f  dur=%.2fs", request_id, rtf, len(audio) / SAMPLE_RATE)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] Inference error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup(ref_path, is_tmp)

    return _wav_response(
    audio,
    request_id,
    used_default,
    inference_time=inference_time,
    rtf=rtf,
    metrics=metrics
    )


# ── /synthesize/longform ──────────────────────────────────────────────────────

@app.post(
    "/synthesize/longform",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
    tags=["Synthesis"],
    summary="Long-form TTS (sentence-chained style)",
)
async def synthesize_longform(
    text: Annotated[
        str,
        Form(
            description=(
                "Full text to synthesise. The server splits it into sentences and "
                "chains style embeddings for natural prosody continuity."
            )
        ),
    ],
    reference_audio: UploadFile | None = File(
        default=None,
        description="Reference WAV providing the voice style. Optional."
    ),
    alpha: Annotated[float, Form()] = 0.3,
    beta: Annotated[float, Form()] = 0.7,
    t: Annotated[
        float,
        Form(description="Style continuity weight between sentences (0–1)."),
    ] = 0.7,
    diffusion_steps: Annotated[int, Form()] = 5,
    embedding_scale: Annotated[float, Form()] = 1.0,
):
    """
    Long-form synthesis that chains style embeddings across sentences to
    produce consistent, natural-sounding speech over multi-sentence input.

    **reference_audio is optional.** Omit to use the built-in default voice.

    Returns a 24 kHz mono PCM-16 WAV file (all sentences concatenated).
    """
    request_id = str(uuid.uuid4())
    logger.info("[%s] /synthesize/longform  text=%r", request_id, text[:80])

    if not text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")

    sentences = [s.strip() for s in re.split(r"(?<=[।.!?])\s+", text) if s.strip()]
    if not sentences:
        raise HTTPException(status_code=422, detail="Could not extract any sentences from text.")

    ref_path, is_tmp, used_default = await _resolve_ref_audio(reference_audio, request_id)
    try:
        ref_s = compute_style(ref_path)
        t0 = time.time()

        wavs: list[np.ndarray] = []
        s_prev = None
        for sentence in sentences:
            wav, s_prev = LFinference(
                text=sentence,
                s_prev=s_prev,
                ref_s=ref_s,
                alpha=alpha,
                beta=beta,
                t=t,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )
            wavs.append(wav)

        silence = np.zeros(int(0.2 * SAMPLE_RATE), dtype=np.float32)
        audio = np.concatenate(
            [chunk for wav in wavs for chunk in (wav, silence)], axis=0
        )
        metrics = evaluate_audio(audio, text)
        inference_time = time.time() - t0
        rtf = (time.time() - t0) / (len(audio) / SAMPLE_RATE)
        logger.info("[%s] done  %d sentences  RTF=%.4f", request_id, len(sentences), rtf)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] Inference error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup(ref_path, is_tmp)

    return _wav_response(
    audio,
    request_id,
    used_default,
    inference_time=inference_time,
    rtf=rtf,
    metrics=metrics
    )


# ── /synthesize/style ─────────────────────────────────────────────────────────

@app.post(
    "/synthesize/style",
    response_class=Response,
    responses={200: {"content": {"audio/wav": {}}}},
    tags=["Synthesis"],
    summary="Style-transfer TTS (reference text + audio)",
)
async def synthesize_style(
    text: Annotated[str, Form(description="Target text to synthesise.")],
    ref_text: Annotated[
        str,
        Form(description="Transcript of the reference audio (used as an additional style cue)."),
    ],
    reference_audio: UploadFile | None = File(
        default=None,
        description="Reference WAV providing the voice style. Optional."
    ),
    alpha: Annotated[float, Form()] = 0.3,
    beta: Annotated[float, Form()] = 0.7,
    diffusion_steps: Annotated[int, Form()] = 5,
    embedding_scale: Annotated[float, Form()] = 1.0,
):
    """
    Style-transfer synthesis.  Providing the reference audio *and* its
    transcript gives the model a richer style signal, often producing output
    that more closely matches the target speaker.

    **reference_audio is optional.** Omit to use the built-in default voice.

    Returns a 24 kHz mono PCM-16 WAV file.
    """
    request_id = str(uuid.uuid4())
    logger.info(
        "[%s] /synthesize/style  text=%r  ref_text=%r",
        request_id, text[:60], ref_text[:60],
    )

    if not text.strip():
        raise HTTPException(status_code=422, detail="'text' must not be empty.")
    if not ref_text.strip():
        raise HTTPException(status_code=422, detail="'ref_text' must not be empty.")

    ref_path, is_tmp, used_default = await _resolve_ref_audio(reference_audio, request_id)
    try:
        ref_s = compute_style(ref_path)
        t0 = time.time()
        audio = STinference(
            text=text,
            ref_s=ref_s,
            ref_text=ref_text,
            alpha=alpha,
            beta=beta,
            diffusion_steps=diffusion_steps,
            embedding_scale=embedding_scale,
        )
        metrics = evaluate_audio(audio, text)
        inference_time = time.time() - t0
        rtf = (time.time() - t0) / (len(audio) / SAMPLE_RATE)
        logger.info("[%s] done  RTF=%.4f", request_id, rtf)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[%s] Inference error: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        _cleanup(ref_path, is_tmp)

    return _wav_response(
    audio,
    request_id,
    used_default,
    inference_time=inference_time,
    rtf=rtf,
    metrics=metrics
    )


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False,
    )