"""
engine.py — StyleTTS2 model loading and inference engine.

Sets up the environment, loads all models, and exposes three inference
functions that main.py calls:
  - inference()        : basic TTS with a reference audio style
  - LFinference()      : long-form / sentence-chained TTS
  - STinference()      : style-transfer TTS (uses reference text too)
"""

import os
import sys

# ── Environment variables (must be set before heavy imports) ─────────────────
os.environ["HOME"] = "/workspace/rizoan"
os.environ["NLTK_DATA"] = "/workspace/rizoan/nltk_data"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import random
import time
import yaml
import logging

import numpy as np
import torch
import torchaudio
import librosa
import phonemizer
import nltk

from munch import Munch
from torch import nn
import torch.nn.functional as F
from nltk.tokenize import word_tokenize

logger = logging.getLogger(__name__)

# ── Reproducibility seeds ────────────────────────────────────────────────────
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(0)
np.random.seed(0)

# Limit CPU threads to avoid the server getting stuck under load
torch.set_num_threads(4)

# ── Change working directory to repo root (models expect relative paths) ─────
REPO_ROOT = "/workspace/rizoan/StyleTTS2-Fork"
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── NLTK data ────────────────────────────────────────────────────────────────
os.makedirs("/workspace/rizoan/nltk_data", exist_ok=True)
nltk.download("punkt", download_dir="/workspace/rizoan/nltk_data", quiet=True)
nltk.download("punkt_tab", download_dir="/workspace/rizoan/nltk_data", quiet=True)

# ── Repo-local imports (available after chdir + sys.path) ────────────────────
from models import *          # noqa: F401, F403
from utils import *           # noqa: F401, F403
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from Utils.PLBERT.util import load_plbert

# ── Device ───────────────────────────────────────────────────────────────────
device: str = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("StyleTTS2 engine using device: %s", device)

# ── Audio preprocessing helpers ───────────────────────────────────────────────
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
_LOG_MEAN, _LOG_STD = -4.0, 4.0

textcleaner = TextCleaner()


def length_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    mask = (
        torch.arange(lengths.max())
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


def preprocess(wave: np.ndarray) -> torch.Tensor:
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - _LOG_MEAN) / _LOG_STD
    return mel_tensor


# ── Phonemisers ───────────────────────────────────────────────────────────────
global_phonemizer_en = phonemizer.backend.EspeakBackend(
    language="en-us", preserve_punctuation=True, with_stress=True
)
global_phonemizer_ben = phonemizer.backend.EspeakBackend(
    language="bn", preserve_punctuation=True, with_stress=True
)


def is_bengali_text(text: str) -> bool:
    """Return True if *text* contains any Bengali Unicode codepoint."""
    return any("\u0980" <= ch <= "\u09FF" for ch in text)


def _phonemize(text: str) -> str:
    """Convert *text* to a space-joined phoneme string."""
    backend = global_phonemizer_ben if is_bengali_text(text) else global_phonemizer_en
    ps = backend.phonemize([text])
    tokens = word_tokenize(ps[0])
    return " ".join(tokens)


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_models():
    config_path = "Models/LJSpeech/config_ft.yml"
    checkpoint_path = "Models/LJSpeech/best_model_presecond.pth"

    config = yaml.safe_load(open(config_path))

    # ASR alignment model
    asr_config = config.get("ASR_config", False)
    asr_path = config.get("ASR_path", False)
    text_aligner = load_ASR_models(asr_path, asr_config)

    # F0 / pitch extractor
    f0_path = config.get("F0_path", False)
    pitch_extractor = load_F0_models(f0_path)

    # PL-BERT
    bert_path = config.get("PLBERT_dir", False)
    plbert = load_plbert(bert_path)

    # Main model
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[k].eval() for k in model]
    _ = [model[k].to(device) for k in model]

    # Load checkpoint weights
    params_whole = torch.load(checkpoint_path, map_location="cpu")
    params = params_whole["net"]

    for key in model:
        if key in params:
            logger.info("Loading weights: %s", key)
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict(
                    {k[7:]: v for k, v in state_dict.items()}  # strip 'module.'
                )
                model[key].load_state_dict(new_state_dict, strict=False)

    _ = [model[k].eval() for k in model]

    # Diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    return model, model_params, sampler


logger.info("Loading StyleTTS2 models — this may take a moment …")
model, model_params, sampler = _load_models()
logger.info("Models loaded successfully.")


# ── Public helpers ────────────────────────────────────────────────────────────

def compute_style(path: str) -> torch.Tensor:
    """Compute a style embedding tensor from a reference WAV file."""
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
    mel_tensor = preprocess(audio).to(device)

    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))

    return torch.cat([ref_s, ref_p], dim=1)


# ── Inference functions ───────────────────────────────────────────────────────

def inference(
    text: str,
    ref_s: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> np.ndarray:
    """
    Basic TTS inference.

    Args:
        text: Input text (English or Bengali).
        ref_s: Style embedding from :func:`compute_style`.
        alpha: Controls how much of the reference *prosody* is preserved (0–1).
        beta: Controls how much of the reference *style* is preserved (0–1).
        diffusion_steps: Number of diffusion denoising steps.
        embedding_scale: Classifier-free guidance scale.

    Returns:
        1-D NumPy float32 waveform at 24 kHz.
    """
    ps = _phonemize(text.strip())
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    # Trim the trailing pulse artifact
    return out.squeeze().cpu().numpy()[..., :-50]


def LFinference(
    text: str,
    s_prev: torch.Tensor | None,
    ref_s: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    t: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> tuple[np.ndarray, torch.Tensor]:
    """
    Long-form TTS inference with style continuity across sentences.

    Args:
        text: Current sentence.
        s_prev: Style tensor from the *previous* sentence (``None`` for first).
        ref_s: Reference style embedding.
        t: Interpolation weight between previous and current style (0–1).
        (other args same as :func:`inference`)

    Returns:
        Tuple of (waveform_ndarray, current_style_tensor).
        Pass the style tensor as *s_prev* for the next sentence.
    """
    ps = _phonemize(text.strip())
    ps = ps.replace("``", '"').replace("''", '"')
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        if s_prev is not None:
            s_pred = t * s_prev + (1 - t) * s_pred

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]
        s_pred = torch.cat([ref, s], dim=-1)

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-100], s_pred


def STinference(
    text: str,
    ref_s: torch.Tensor,
    ref_text: str,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
) -> np.ndarray:
    """
    Style-transfer TTS inference.

    In addition to the reference audio, the reference *text* (the transcript
    of the reference audio) is used to guide style prediction.

    Args:
        text: Target text to synthesise.
        ref_s: Style embedding from :func:`compute_style`.
        ref_text: Transcript of the reference audio.
        (other args same as :func:`inference`)

    Returns:
        1-D NumPy float32 waveform at 24 kHz.
    """
    ps = _phonemize(text.strip())
    tokens = textcleaner(ps)
    tokens.insert(0, 0)
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)

    ref_ps = _phonemize(ref_text.strip())
    ref_tokens = textcleaner(ref_ps)
    ref_tokens.insert(0, 0)
    ref_tokens = torch.LongTensor(ref_tokens).to(device).unsqueeze(0)

    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        ref_input_lengths = torch.LongTensor([ref_tokens.shape[-1]]).to(device)
        ref_text_mask = length_to_mask(ref_input_lengths).to(device)
        ref_bert_dur = model.bert(ref_tokens, attention_mask=(~ref_text_mask).int())  # noqa: F841

        s_pred = sampler(
            noise=torch.randn((1, 256)).unsqueeze(1).to(device),
            embedding=bert_dur,
            embedding_scale=embedding_scale,
            features=ref_s,
            num_steps=diffusion_steps,
        ).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]
        ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
        s = beta * s + (1 - beta) * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame : c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = t_en @ pred_aln_trg.unsqueeze(0).to(device)
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))

    return out.squeeze().cpu().numpy()[..., :-50]