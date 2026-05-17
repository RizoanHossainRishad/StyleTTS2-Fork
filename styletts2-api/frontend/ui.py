"""
ui.py — Streamlit front-end for the StyleTTS2 API.

Run with:
    streamlit run ui.py

Set the API base URL via the sidebar or via the environment variable:
    STYLETTS2_API_URL=http://localhost:8000
"""

import io
import os


os.environ["HOME"] = "/workspace/rizoan"
os.environ["XDG_CONFIG_HOME"] = "/workspace/rizoan/.config"
os.environ["XDG_CACHE_HOME"] = "/workspace/rizoan/.cache"
import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_API_URL = os.getenv("STYLETTS2_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="StyleTTS2 — Speech Synthesis",
    page_icon="🗣️",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    api_url = st.text_input("API base URL", value=DEFAULT_API_URL)

    st.divider()
    st.subheader("Synthesis parameters")
    alpha = st.slider(
        "Alpha (reference prosody)",
        min_value=0.0, max_value=1.0, value=0.3, step=0.05,
        help="How much of the reference speaker's *prosody* to preserve.",
    )
    beta = st.slider(
        "Beta (reference style)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="How much of the reference speaker's *style* to preserve.",
    )
    diffusion_steps = st.slider(
        "Diffusion steps",
        min_value=1, max_value=20, value=5,
        help="More steps → higher quality, but slower.",
    )
    embedding_scale = st.slider(
        "Embedding scale (CFG)",
        min_value=0.5, max_value=3.0, value=1.0, step=0.1,
        help="Classifier-free guidance scale. Higher = stronger style signal.",
    )
    t_lf = st.slider(
        "Style continuity (long-form)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.05,
        help="How much style carries over between sentences in long-form mode.",
    )

    st.divider()
    if st.button("🔍 Check API health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            r.raise_for_status()
            data = r.json()
            st.success(f"API is **{data['status']}** (device: `{data['device']}`)")
            if data.get("default_reference_audio_present"):
                st.info("✅ Default reference audio is available on the server.")
            else:
                st.warning("⚠️ Default reference audio file not found on server.")
        except Exception as exc:
            st.error(f"Cannot reach API: {exc}")

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🗣️ StyleTTS2 — Neural Speech Synthesis")
st.caption(
    "You can upload a file OR record your voice. "
    "If neither is provided, the default voice is used."
)

mode = st.radio(
    "Synthesis mode",
    options=["Basic", "Long-form", "Style transfer"],
    horizontal=True,
    help=(
        "**Basic** — single utterance.  "
        "**Long-form** — chains style across sentences.  "
        "**Style transfer** — also uses the reference transcript for richer style."
    ),
)

st.divider()

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    text_input = st.text_area(
        "Input text",
        height=160,
        placeholder=(
            "Type English or Bengali text here …\n\n"
            "e.g.  ছোটবেলার দিনগুলোর কথা মনে পড়লে আমার খুব ভালো লাগে।"
        ),
    )

    if mode == "Style transfer":
        ref_text_input = st.text_area(
            "Reference text (transcript of the reference audio)",
            height=80,
            placeholder="Type the transcript of the reference audio here …",
        )
    else:
        ref_text_input = ""

with col_right:
    st.markdown("**Reference audio** *(optional)*")
    st.caption(
        "Upload your own WAV to use a custom voice style.  "
        "Leave empty to use the built-in default voice."
    )
    
    reference_audio = st.file_uploader(
        "Upload WAV (optional)",
        type=["wav"],
        label_visibility="collapsed",
    )

    mic_audio = st.audio_input("🎤 Or record voice (optional)")

    # ── Preview uploaded file ──
    if reference_audio is not None:
        st.audio(reference_audio, format="audio/wav")

    # ── Preview microphone recording ──
    elif mic_audio is not None:
        st.audio(mic_audio)

    # ── Fallback message ──
    else:
        st.info("🎙️ No audio provided — server will use default voice.")

st.divider()

synthesize_btn = st.button("🎙️ Synthesise", type="primary", use_container_width=True)

if synthesize_btn:
    # ── Validation ────────────────────────────────────────────────────────────
    errors = []
    if not text_input.strip():
        errors.append("Please enter some text to synthesise.")
    if mode == "Style transfer" and not ref_text_input.strip():
        errors.append("Style transfer mode requires a reference transcript.")

    for err in errors:
        st.error(err)

    if not errors:
        # ── Build request ─────────────────────────────────────────────────────
        common_params = {
            "alpha": alpha,
            "beta": beta,
            "diffusion_steps": diffusion_steps,
            "embedding_scale": embedding_scale,
        }

        if mode == "Basic":
            endpoint = f"{api_url}/synthesize"
            data = {"text": text_input, **common_params}
        elif mode == "Long-form":
            endpoint = f"{api_url}/synthesize/longform"
            data = {"text": text_input, "t": t_lf, **common_params}
        else:  # Style transfer
            endpoint = f"{api_url}/synthesize/style"
            data = {"text": text_input, "ref_text": ref_text_input, **common_params}

        # Only include the file field if the user actually uploaded something
        audio_source = None

        # priority: mic > file > default
        if mic_audio is not None:
            audio_source = mic_audio
        elif reference_audio is not None:
            audio_source = reference_audio

        if audio_source is not None:
            audio_source.seek(0)
            files = {
                "reference_audio": (
                    "reference.wav",
                    audio_source,
                    "audio/wav"
                )
            }
        else:
            files = {}

        # ── Call API ──────────────────────────────────────────────────────────
        with st.spinner("Synthesising … this may take a few seconds."):
            try:
                response = requests.post(
                    endpoint,
                    data=data,
                    files=files,
                    timeout=120,
                )
                response.raise_for_status()
            except requests.exceptions.HTTPError as exc:
                try:
                    detail = exc.response.json().get("detail", str(exc))
                except Exception:
                    detail = str(exc)
                st.error(f"API error {exc.response.status_code}: {detail}")
                st.stop()
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to the API at **{api_url}**.  "
                    "Check that the backend is running."
                )
                st.stop()
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")
                st.stop()

        # ── Display results ───────────────────────────────────────────────────
        audio_bytes = response.content
        headers = response.headers

        mos = headers.get("X-MOS", "?")
        silence_ratio = headers.get("X-Silence-Ratio", "?")
        noise_ratio = headers.get("X-Noise-Ratio", "?")
        #clipping_ratio = headers.get("X-Clipping-Ratio", "?")
        speaking_rate = headers.get("X-Speaking-Rate", "?")
        used_default = headers.get("X-Used-Default-Reference", "false") == "true"
        st.success(
            "✅ Synthesis complete!"
            + (" *(used built-in default voice)*" if used_default else "")
        )

        dur = headers.get("X-Duration-Seconds", "?")
        rid = headers.get("X-Request-Id", "?")
        infer_time = headers.get("X-Inference-Time", "?")
        rtf = headers.get("X-RTF", "?")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        col_m1.metric("Audio Duration", f"{dur} s")
        col_m2.metric("Inference Time", f"{infer_time} s")
        col_m3.metric("RTF", rtf)
        col_m4.metric(
            "Request ID",
            rid[:8] + "…" if len(rid) > 8 else rid
        )

        st.subheader("🔊 Synthesised audio")
        st.audio(audio_bytes, format="audio/wav")

        st.download_button(
            label="⬇️ Download WAV",
            data=audio_bytes,
            file_name=f"styletts2_{rid[:8]}.wav",
            mime="audio/wav",
            use_container_width=True,
        )

        if reference_audio is not None:
            st.subheader("📖 Reference audio (uploaded)")
            reference_audio.seek(0)
            st.audio(reference_audio.read(), format="audio/wav")

        st.subheader("📊 Quality Analysis")

        q1, q2, q3 = st.columns(3)

        q1.metric("UTMOS", mos)
        q1.metric("Noise Ratio", noise_ratio)

        q2.metric("Silence Ratio", silence_ratio)
        #q2.metric("Clipping Ratio", clipping_ratio)

        q3.metric("Speaking Rate", f"{speaking_rate} w/s")