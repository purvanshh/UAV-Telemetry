# UAV Sentinel API — Hugging Face Spaces (Docker SDK) image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA build.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0.0" \
 && pip install -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user src ./src
COPY --chown=user uav_model ./uav_model
COPY --chown=user config ./config

# Free Spaces have no persistent disk: uploads live here until the Space restarts.
ENV DATA_DIR=/home/user/data/flights \
    DIRECT_DATA_DIR=/home/user/data/direct
RUN mkdir -p /home/user/data && chown -R user /home/user/data

USER user
EXPOSE 7860
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
