# 1) Base image: Python 3.10 (matches your pyenv 3.10.18)
FROM python:3.10-slim

# 2) System deps (Pillow/webp, git if you need it in container)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libjpeg62-turbo libpng16-16 libtiff6 \
    libwebp7 libxext6 libxrender1 libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 3) Workdir
WORKDIR /app

# 4) Pre-copy requirements for better build cache
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip wheel && \
    pip install -r /app/requirements.txt

# 5) Copy the rest of your application
# IMPORTANT: make sure you've run `git lfs pull` before building so binaries are real
COPY . /app

# 6) Streamlit settings for headless server
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 7) Expose Streamlit port
EXPOSE 8501

# 8) Healthcheck (optional but nice)
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD \
  wget -qO- http://localhost:8501/_stcore/health || exit 1

# 9) Default command
CMD ["streamlit", "run", "app.py"]
