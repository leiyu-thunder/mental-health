FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OpenCV/MediaPipe 常用依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# ---- Pre-download HF model at build time (so runtime works without Internet) ----
ENV MODEL_ID="mo-thecreator/vit-Facial-Expression-Recognition"
ENV MODEL_DIR="/models/vit-fer"
RUN python - << 'PY'
import os
from transformers import ViTImageProcessor, ViTForImageClassification
model_id = os.environ["MODEL_ID"]
model_dir = os.environ["MODEL_DIR"]
os.makedirs(model_dir, exist_ok=True)
processor = ViTImageProcessor.from_pretrained(model_id)
model = ViTForImageClassification.from_pretrained(model_id)
processor.save_pretrained(model_dir)
model.save_pretrained(model_dir)
print("Saved model to", model_dir)
PY
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV MODEL_DIR=/models/vit-fer

COPY app.py /app/app.py
COPY face_landmarker.task /app/face_landmarker.task
COPY hand_landmarker.task /app/hand_landmarker.task

ENV FACE_TASK_PATH=/app/face_landmarker.task
ENV HAND_TASK_PATH=/app/hand_landmarker.task

# 可选：让 HF 缓存走持久盘（配合 Space 开 Persistent Storage 才真正持久）
ENV HF_HOME=/data/.huggingface

EXPOSE 8080

CMD ["sh","-c","gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:${PORT:-8080} app:app"]
