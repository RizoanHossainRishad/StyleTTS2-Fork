# Running the StyleTTS2 API + Streamlit UI

## 1. Activate Conda Environment

Open a terminal on the server and activate your TTS environment:

```bash
conda activate tts
```

---

# 2. Navigate to the Backend Directory

Go to the directory containing:

* `main.py`
* `engine.py`
* `sample_refs/`

Example:

```bash
cd /workspace/rizoan/StyleTTS2-Fork/styletts2-api/backend
```

---

# 3. Start the FastAPI Backend Server

Run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8001
```

You should see logs similar to:

```text
INFO:     Uvicorn running on http://0.0.0.0:8001
INFO:     Application startup complete.
```

The API will now be available at:

```text
http://localhost:8001
```

Swagger documentation:

```text
http://localhost:8001/docs
```

Health endpoint:

```text
http://localhost:8001/health
```

---

# 4. Open a New Terminal

Keep the FastAPI server running.

Open another terminal session (or tmux pane/window).

Activate the same environment again:

```bash
conda activate tts
```

---

# 5. Navigate to the Frontend Directory

Go to the directory containing `ui.py`.

Example:

```bash
cd /workspace/rizoan/StyleTTS2-Fork/styletts2-api/frontend
```

---

# 6. Launch the Streamlit UI

Run:

```bash
streamlit run ui.py
```

You should see output similar to:

```text
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

---

# 7. Access the UI in Browser

Open:

```text
http://localhost:8501
```

If using VSCode Remote SSH or SSH tunneling, forward port:

* `8501` → Streamlit UI
* `8001` → FastAPI backend (optional for docs/testing)

---

# 8. Using the Application

1. Enter text
2. Optionally upload a reference WAV
3. Select synthesis mode:

   * Basic
   * Long-form
   * Style transfer
4. Click:

   ```text
   Synthesise
   ```

The generated WAV can then:

* be played in-browser
* downloaded directly

---

# Optional: Run with tmux (Recommended for Servers)

## Start backend tmux session

```bash
tmux new -s tts-backend
```

Run:

```bash
conda activate tts
cd /workspace/rizoan/StyleTTS2-Fork/styletts2-api/backend
uvicorn main:app --host 0.0.0.0 --port 8001
```

Detach:

```text
CTRL+B then D
```

---

## Start frontend tmux session

```bash
tmux new -s tts-frontend
```

Run:

```bash
conda activate tts
cd /workspace/rizoan/StyleTTS2-Fork/styletts2-api/frontend
streamlit run ui.py
```

Detach again:

```text
CTRL+B then D
```

---

# Useful tmux Commands

List sessions:

```bash
tmux ls
```

Reconnect to backend:

```bash
tmux attach -t tts-backend
```

Reconnect to frontend:

```bash
tmux attach -t tts-frontend
```

Kill session:

```bash
tmux kill-session -t tts-backend
```

# Directory Issue Solving

Check $HOME directory:
Yes — this is NOT actually a StyleTTS2 inference failure.

Your backend architecture is mostly fine right now.

The real issue is:

* Streamlit internally tries to create/write its installation metadata
* it resolves your HOME/config path to:

  ```text
  /mnt/newworkspace/rizoan
  ```
* that directory either:

  * no longer exists
  * or your current user cannot write there

So Streamlit crashes BEFORE or DURING session handling.

The important clue is this:

```text
streamlit/runtime/metrics_util.py
```

and:

```text
PermissionError: [Errno 13] Permission denied: 'random directory that is unaccissible'
```

This is NOT coming from:

* FastAPI
* StyleTTS2
* inference()
* CUDA
* audio generation

It is specifically a filesystem/config issue.

---


# The Most Likely Cause

Your Conda environment or shell still has:

```bash
HOME=some_random_directory_unacessible
```

even though you override it INSIDE Python later.

Streamlit initializes some paths before your app fully controls them.

---

# Verify Immediately

Run these commands INSIDE THE SAME TERMINAL where you launch Streamlit:

```bash
echo $HOME
```

and:

```bash
env | grep -i streamlit
```

and:

```bash
env | grep -i xdg
```

and:

```bash
python -c "import os; print(os.environ.get('HOME'))"
```



---

# Fix Discussion

Before launching Streamlit:

```bash
export HOME=/workspace/rizoan
```

Then run:

```bash
streamlit run ui.py
```

---

#  Permanent Fix

Add this BEFORE importing streamlit in `ui.py`:

```python
import os

os.environ["HOME"] = "/workspace/rizoan"
os.environ["XDG_CONFIG_HOME"] = "/workspace/rizoan/.config"
os.environ["XDG_CACHE_HOME"] = "/workspace/rizoan/.cache"
```

BEFORE:

```python
import streamlit as st
```

Very important.

Because Streamlit internally uses:

* HOME
* XDG_CONFIG_HOME
* XDG_CACHE_HOME

for installation metadata.

---



# Recommended Architecture

Create:

```text
styletts2-api/
    .env
```

or:

```text
config/env.sh
```

Example:

```bash
export HOME=/workspace/rizoan
export XDG_CONFIG_HOME=/workspace/rizoan/.config
export XDG_CACHE_HOME=/workspace/rizoan/.cache
export NLTK_DATA=/workspace/rizoan/nltk_data
```

Then source it before launching anything.

---



#  Observation



| File      | Real Role         |
| --------- | ----------------- |
| ui.py     | frontend          |
| main.py   | API backend       |
| engine.py | inference backend |



---

#  System Flow Currently

Actual runtime architecture is:

```text
Streamlit UI
    ↓ HTTP
FastAPI Server
    ↓
StyleTTS2 Engine
    ↓
CUDA
```

The crash occurs BEFORE reaching:

* FastAPI
* engine.py
* inference

So inference is not yet failing.

---

# Immediate Working Fix

At shell level:

```bash
export HOME=/workspace/rizoan
export XDG_CONFIG_HOME=/workspace/rizoan/.config
export XDG_CACHE_HOME=/workspace/rizoan/.cache

mkdir -p /workspace/rizoan/.config
mkdir -p /workspace/rizoan/.cache
```

Then:

```bash
streamlit run ui.py
```

---


