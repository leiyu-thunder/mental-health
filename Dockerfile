FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
COPY face_landmarker.task /app/face_landmarker.task
COPY hand_landmarker.task /app/hand_landmarker.task

ENV FACE_TASK_PATH=/app/face_landmarker.task
ENV HAND_TASK_PATH=/app/hand_landmarker.task

# 关键：监听 CloudBase 提供的 PORT（兜底 8080）
CMD ["sh","-c","gunicorn -w 2 -k gthread --threads 4 -b 0.0.0.0:${PORT:-8080} app:app"]
