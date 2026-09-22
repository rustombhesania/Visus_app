# ViSuS backend — Docker image for Render.
#
# Why Docker and not Render's native Python runtime: several accepted
# upload formats (m4a, mp4, aac, wma, amr, 3gp, webm, mka) aren't decoded
# by soundfile directly -- librosa falls back to audioread, which shells
# out to ffmpeg. Render's native runtime has no apt-get / buildpack
# mechanism to install system packages (confirmed via Render's own
# community forum, not an assumption), so ffmpeg has to come from the
# base image instead. wav/mp3/flac/ogg still work without this -- only
# the ffmpeg-dependent formats would silently fail on native.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py compute.py verdict.py ./

# Render sets $PORT at runtime; it is not fixed at build time.
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
