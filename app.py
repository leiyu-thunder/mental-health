import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify

import torch
from transformers import ViTImageProcessor, ViTForImageClassification

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import subprocess
import json

# -------------------------
# Video probe & normalization
# -------------------------
def _run_cmd(cmd: list) -> str:
    """Run a command and return stdout; raise RuntimeError with stderr on failure."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")
    return p.stdout


def ffprobe_video_info(path: str) -> dict:
    """Probe real container/codec info (works even if suffix is misleading)."""
    out = _run_cmd([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path
    ])
    info = json.loads(out)

    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
    if not vstreams:
        raise RuntimeError("No video stream found by ffprobe.")
    v0 = vstreams[0]
    fmt = info.get("format", {}) or {}

    return {
        "format_name": fmt.get("format_name"),
        "duration": fmt.get("duration"),
        "size": fmt.get("size"),
        "video_codec": v0.get("codec_name"),
        "width": v0.get("width"),
        "height": v0.get("height"),
        "avg_frame_rate": v0.get("avg_frame_rate"),
        "r_frame_rate": v0.get("r_frame_rate"),
        "raw": info,
    }


def normalize_to_15fps_mp4(input_path: str, output_path: str) -> None:
    """Standardize to CFR 15fps and re-encode to analysis-friendly H.264 MP4."""
    _run_cmd([
        "ffmpeg", "-y",
        "-i", input_path,
        # Force CFR=15 and make width/height even (required for yuv420p/h264)
        "-vf", "fps=15,scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-vsync", "cfr",
        "-an",
        "-c:v", "libx264",
        "-profile:v", "baseline",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        "-crf", "28",
        output_path
    ])

# =========================================================
# 0) 配置
# =========================================================
MODEL_ID = "mo-thecreator/vit-Facial-Expression-Recognition"
MODEL_DIR = os.environ.get("MODEL_DIR", "/models/vit-fer")  # local, pre-downloaded at build time
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_FPS = 5
MAX_SEC = 60.0

FACE_TASK_PATH = os.getenv("FACE_TASK_PATH", "./face_landmarker.task")
HAND_TASK_PATH = os.getenv("HAND_TASK_PATH", "./hand_landmarker.task")

# 6段刺激：每段10秒
TIMELINE_6SEG = {
    "seg1_greeting": (0.0, 10.0),
    "seg2_motion":   (10.0, 20.0),
    "seg3_more":     (20.0, 30.0),
    "seg4_surprise": (30.0, 40.0),
    "seg5_reward":   (40.0, 50.0),
    "seg6_calm":     (50.0, 60.0),
}

# 功能段（总览聚合用）
FUNCTION_SEG = {
    "baseline": (0.0, 20.0),
    "adapt":    (20.0, 30.0),
    "surprise": (30.0, 40.0),
    "reward":   (40.0, 50.0),
    "calm":     (50.0, 60.0),
}

AGE_GROUPS = {"9-12", "13-15", "16-18"}

# ViT FER 7类
EMO_CANON = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# FaceLandmarker landmark index（与常见 FaceMesh 近似一致）
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_LMS = [13, 14, 78, 308]


# =========================================================
# 1) ViT 表情模型加载
# =========================================================
if not os.path.isdir(MODEL_DIR):
    raise RuntimeError("MODEL_DIR not found: %s. This service is configured for offline model loading. Build the image with the model pre-downloaded (see Dockerfile) or mount the model directory." % MODEL_DIR)
processor = ViTImageProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
vit_model = ViTForImageClassification.from_pretrained(MODEL_DIR, local_files_only=True).to(DEVICE)
vit_model.eval()
ID2LABEL = vit_model.config.id2label


def predict_emotion_probs(face_bgr: np.ndarray) -> Dict[str, float]:
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = vit_model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    tmp = {}
    for i, p in enumerate(probs):
        label = str(ID2LABEL[i]).lower()
        tmp[label] = float(p)

    canon = {k: float(tmp.get(k, 0.0)) for k in EMO_CANON}
    s = sum(canon.values()) + 1e-8
    return {k: v / s for k, v in canon.items()}

# =========================================================
# 2) 工具函数
# =========================================================
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def seg_mask(ts: np.ndarray, seg: Tuple[float, float]) -> np.ndarray:
    a, b = seg
    return (ts >= a) & (ts < b)


def _pt_from_norm(lm, w, h) -> np.ndarray:
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def eye_aspect_ratio(face_lms, eye_idx, w, h) -> float:
    p1 = _pt_from_norm(face_lms[eye_idx[0]], w, h)
    p2 = _pt_from_norm(face_lms[eye_idx[1]], w, h)
    p3 = _pt_from_norm(face_lms[eye_idx[2]], w, h)
    p4 = _pt_from_norm(face_lms[eye_idx[3]], w, h)
    p5 = _pt_from_norm(face_lms[eye_idx[4]], w, h)
    p6 = _pt_from_norm(face_lms[eye_idx[5]], w, h)

    num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    den = 2.0 * (np.linalg.norm(p1 - p4) + 1e-6)
    return float(num / den)


def build_face_boxes_from_landmarks(face_lms, w, h):
    idxs = [10, 152, 234, 454]
    pts = np.array([_pt_from_norm(face_lms[i], w, h) for i in idxs], dtype=np.float32)
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))
    pad = 0.10 * max(w, h)
    face_box = (x1 - pad, y1 - pad, x2 + pad, y2 + pad)

    mpts = np.array([_pt_from_norm(face_lms[i], w, h) for i in MOUTH_LMS], dtype=np.float32)
    mx1, my1 = float(np.min(mpts[:, 0])), float(np.min(mpts[:, 1]))
    mx2, my2 = float(np.max(mpts[:, 0])), float(np.max(mpts[:, 1]))
    mpad = 0.06 * max(w, h)
    mouth_box = (mx1 - mpad, my1 - mpad, mx2 + mpad, my2 + mpad)

    return face_box, mouth_box


def point_in_box(x, y, box) -> bool:
    x1, y1, x2, y2 = box
    return (x1 <= x <= x2) and (y1 <= y <= y2)


def crop_face_from_landmarks(bgr: np.ndarray, face_lms, pad_ratio: float = 0.20) -> Optional[np.ndarray]:
    h, w = bgr.shape[:2]
    pts = np.array([_pt_from_norm(lm, w, h) for lm in face_lms], dtype=np.float32)
    x1, y1 = float(np.min(pts[:, 0])), float(np.min(pts[:, 1]))
    x2, y2 = float(np.max(pts[:, 0])), float(np.max(pts[:, 1]))

    pad = pad_ratio * max((x2 - x1), (y2 - y1))
    x1 = int(max(0, x1 - pad))
    y1 = int(max(0, y1 - pad))
    x2 = int(min(w - 1, x2 + pad))
    y2 = int(min(h - 1, y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2].copy()


def binary_events_stats(bin_seq: np.ndarray, fps: float) -> Dict[str, float]:
    """
    0/1序列 -> 事件次数/持续时间
      rate: 占比
      count: 事件次数（连续1算一次）
      total_sec: 总持续时间
      mean_sec: 平均每次持续
      max_sec: 最长一次持续
    """
    bin_seq = (bin_seq.astype(np.int32) > 0).astype(np.int32)
    n = int(bin_seq.size)
    if n == 0:
        return {"rate": 0.0, "count": 0.0, "total_sec": 0.0, "mean_sec": 0.0, "max_sec": 0.0}

    rate = float(np.mean(bin_seq))
    if n == 1:
        if bin_seq[0] == 1:
            d = 1.0 / float(fps)
            return {"rate": 1.0, "count": 1.0, "total_sec": d, "mean_sec": d, "max_sec": d}
        return {"rate": 0.0, "count": 0.0, "total_sec": 0.0, "mean_sec": 0.0, "max_sec": 0.0}

    starts = np.where((bin_seq[1:] == 1) & (bin_seq[:-1] == 0))[0] + 1
    ends = np.where((bin_seq[1:] == 0) & (bin_seq[:-1] == 1))[0] + 1
    if bin_seq[0] == 1:
        starts = np.insert(starts, 0, 0)
    if bin_seq[-1] == 1:
        ends = np.append(ends, n)

    durs_frames = (ends - starts).astype(np.int32)
    if durs_frames.size == 0:
        return {"rate": rate, "count": 0.0, "total_sec": 0.0, "mean_sec": 0.0, "max_sec": 0.0}

    durs_sec = durs_frames.astype(np.float32) / float(fps)
    return {
        "rate": rate,
        "count": float(durs_sec.size),
        "total_sec": float(np.sum(durs_sec)),
        "mean_sec": float(np.mean(durs_sec)),
        "max_sec": float(np.max(durs_sec)),
    }

# =========================================================
# 3) 抽帧（按 5fps）
# =========================================================
def sample_video(video_path: str, sample_fps: int = 5, max_sec: float = 60.0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if frame_count > 0 else max_sec
    use_sec = min(max_sec, duration)

    ts = np.arange(0.0, use_sec, 1.0 / sample_fps, dtype=np.float32)
    frames = []
    for t in ts:
        frame_idx = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        frames.append(frame if ok else None)

    cap.release()
    return frames, ts


# =========================================================
# 4) MediaPipe Tasks 创建（VIDEO 模式）
# =========================================================
def create_face_landmarker() -> mp_vision.FaceLandmarker:
    if not os.path.exists(FACE_TASK_PATH):
        raise FileNotFoundError(f"face_landmarker.task not found at: {FACE_TASK_PATH}")

    base = mp_python.BaseOptions(model_asset_path=FACE_TASK_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def create_hand_landmarker() -> mp_vision.HandLandmarker:
    if not os.path.exists(HAND_TASK_PATH):
        raise FileNotFoundError(f"hand_landmarker.task not found at: {HAND_TASK_PATH}")

    base = mp_python.BaseOptions(model_asset_path=HAND_TASK_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


# =========================================================
# 5) 主分析：逐帧特征（底层不改）
# =========================================================
@dataclass
class FrameFeat:
    t: float
    face_ok: bool
    emo: Dict[str, float]
    blink: int
    head_motion: float
    gaze_dev: float
    hand_touch_face: int
    hand_near_mouth: int


def analyze(video_path: str) -> Dict[str, Any]:
    frames, ts = sample_video(video_path, SAMPLE_FPS, MAX_SEC)

    face_task = create_face_landmarker()
    hand_task = create_hand_landmarker()

    prev_nose = None
    blink_state = 0
    EAR_CLOSE = 0.18
    EAR_OPEN = 0.21

    feats: List[FrameFeat] = []

    for i, bgr in enumerate(frames):
        t = float(ts[i])
        if bgr is None:
            feats.append(FrameFeat(
                t=t, face_ok=False, emo={k: 0.0 for k in EMO_CANON},
                blink=0, head_motion=0.0, gaze_dev=1.0,
                hand_touch_face=0, hand_near_mouth=0
            ))
            continue

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(round(t * 1000.0))

        face_res = face_task.detect_for_video(mp_image, timestamp_ms)
        has_face = bool(face_res.face_landmarks) and len(face_res.face_landmarks) > 0

        face_ok = False
        blink = 0
        head_motion = 0.0
        gaze_dev = 1.0
        emo = {k: 0.0 for k in EMO_CANON}

        face_box = None
        mouth_box = None

        if has_face:
            face_ok = True
            face_lms = face_res.face_landmarks[0]

            roi = crop_face_from_landmarks(bgr, face_lms, pad_ratio=0.20)
            if roi is not None:
                emo = predict_emotion_probs(roi)

            l_ear = eye_aspect_ratio(face_lms, LEFT_EYE, w, h)
            r_ear = eye_aspect_ratio(face_lms, RIGHT_EYE, w, h)
            ear = 0.5 * (l_ear + r_ear)

            if blink_state == 0 and ear < EAR_CLOSE:
                blink_state = 1
            elif blink_state == 1 and ear > EAR_OPEN:
                blink_state = 0
                blink = 1

            nose = _pt_from_norm(face_lms[1], w, h)
            if prev_nose is not None:
                head_motion = float(np.linalg.norm(nose - prev_nose))
            prev_nose = nose

            center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            gaze_dev = float(np.linalg.norm(nose - center) / (max(w, h) + 1e-6))

            face_box, mouth_box = build_face_boxes_from_landmarks(face_lms, w, h)

        hand_touch_face = 0
        hand_near_mouth = 0
        if face_ok and face_box and mouth_box:
            hand_res = hand_task.detect_for_video(mp_image, timestamp_ms)
            if hand_res.hand_landmarks:
                for hand_lms in hand_res.hand_landmarks:
                    for tip in [4, 8, 12, 16, 20]:
                        hx = hand_lms[tip].x * w
                        hy = hand_lms[tip].y * h
                        if point_in_box(hx, hy, face_box):
                            hand_touch_face = 1
                        if point_in_box(hx, hy, mouth_box):
                            hand_near_mouth = 1

        feats.append(FrameFeat(
            t=t, face_ok=face_ok, emo=emo,
            blink=blink, head_motion=head_motion, gaze_dev=gaze_dev,
            hand_touch_face=hand_touch_face, hand_near_mouth=hand_near_mouth
        ))

    face_task.close()
    hand_task.close()

    return compute_metrics(feats, ts)


# =========================================================
# 6) 底层指标聚合（不改）
# =========================================================
def compute_metrics(feats: List[FrameFeat], ts: np.ndarray) -> Dict[str, Any]:
    N = len(feats)
    if N == 0:
        raise RuntimeError("No frames sampled")

    face_ok = np.array([1.0 if f.face_ok else 0.0 for f in feats], dtype=np.float32)
    face_ok_ratio = float(np.mean(face_ok))

    blink_cnt = float(sum(f.blink for f in feats))
    duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 60.0
    blink_rate_per_min = float(blink_cnt / (duration + 1e-6) * 60.0)

    head_motion = np.array([f.head_motion for f in feats], dtype=np.float32)
    gaze_dev = np.array([f.gaze_dev for f in feats], dtype=np.float32)
    head_motion_mean = float(np.mean(head_motion))
    gaze_off_ratio = float(np.mean(gaze_dev > 0.22))

    touch_seq = np.array([f.hand_touch_face for f in feats], dtype=np.int32)
    near_seq  = np.array([f.hand_near_mouth for f in feats], dtype=np.int32)
    touch_stats = binary_events_stats(touch_seq, SAMPLE_FPS)
    near_stats  = binary_events_stats(near_seq, SAMPLE_FPS)

    emo_mat = np.zeros((N, 7), dtype=np.float32)
    for i, f in enumerate(feats):
        for j, k in enumerate(EMO_CANON):
            emo_mat[i, j] = float(f.emo.get(k, 0.0))

    idx = {k: i for i, k in enumerate(EMO_CANON)}
    happy = emo_mat[:, idx["happy"]]
    sad = emo_mat[:, idx["sad"]]
    angry = emo_mat[:, idx["angry"]]
    fear = emo_mat[:, idx["fear"]]
    surprise = emo_mat[:, idx["surprise"]]

    def seg_mean(arr: np.ndarray, seg: Tuple[float, float]) -> float:
        m = seg_mask(ts, seg)
        return float(np.mean(arr[m])) if np.any(m) else float(np.mean(arr))

    def seg_max(arr: np.ndarray, seg: Tuple[float, float]) -> float:
        m = seg_mask(ts, seg)
        return float(np.max(arr[m])) if np.any(m) else float(np.max(arr))

    baseline_happy = seg_mean(happy, FUNCTION_SEG["baseline"])
    baseline_sad = seg_mean(sad, FUNCTION_SEG["baseline"])
    baseline_tension = seg_mean((fear + angry) / 2.0, FUNCTION_SEG["baseline"])

    reward_happy_mean = seg_mean(happy, FUNCTION_SEG["reward"])
    happy_gain = float(reward_happy_mean - baseline_happy)

    happy_latency = None
    th = baseline_happy + 0.08
    start_t = FUNCTION_SEG["reward"][0]
    m_reward = seg_mask(ts, FUNCTION_SEG["reward"])
    for i in np.where(m_reward)[0]:
        if feats[i].face_ok and happy[i] >= th:
            happy_latency = float(ts[i] - start_t)
            break

    surprise_peak = seg_max(surprise, FUNCTION_SEG["surprise"])

    calm_recover_speed = 0.0
    m_calm = seg_mask(ts, FUNCTION_SEG["calm"])
    if np.sum(m_calm) >= 5:
        x = ts[m_calm] - ts[m_calm][0]
        y = ((fear + angry) / 2.0)[m_calm]
        calm_recover_speed = float(np.polyfit(x, y, 1)[0])  # 越负越好

    def nrm(x, lo, hi):
        return clamp01((x - lo) / (hi - lo + 1e-6))

    risk = 0.0
    risk += 0.24 * nrm(baseline_sad, 0.08, 0.35)
    risk += 0.20 * nrm(baseline_tension, 0.08, 0.35)
    risk += 0.18 * nrm(gaze_off_ratio, 0.05, 0.35)
    risk += 0.12 * nrm(head_motion_mean, 0.5, 6.0)
    risk += 0.12 * nrm(float(touch_stats["rate"]), 0.05, 0.30)
    risk += 0.10 * nrm(float(near_stats["rate"]), 0.02, 0.20)
    risk += 0.10 * nrm(0.06 - happy_gain, -0.05, 0.08)

    mental_health_index = float(np.clip(85.0 - risk * 60.0, 0.0, 100.0))
    confidence = float(np.clip(face_ok_ratio, 0.0, 1.0))
    if confidence < 0.7:
        mental_health_index *= confidence / 0.7

    segment_cards_raw: Dict[str, Any] = {}
    for seg_key, (a, b) in TIMELINE_6SEG.items():
        m = seg_mask(ts, (a, b))
        if not np.any(m):
            segment_cards_raw[seg_key] = {
                "happy_mean": 0.0, "sad_mean": 0.0, "tension_mean": 0.0, "surprise_mean": 0.0,
                "gaze_off_ratio": 1.0, "head_motion_mean": 0.0, "blink_rate_per_min": 0.0,
                "face_ok_ratio": 0.0,
                "touch_face_rate": 0.0, "touch_face_count": 0.0, "touch_face_mean_sec": 0.0,
                "hand_near_mouth_rate": 0.0, "hand_near_mouth_count": 0.0, "hand_near_mouth_mean_sec": 0.0,
            }
            continue

        seg_idx = np.where(m)[0]
        seg_face_ok_ratio = float(np.mean(face_ok[m]))
        seg_sec = float(b - a)

        seg_blink = np.array([feats[i].blink for i in seg_idx], dtype=np.float32)
        seg_blink_rate = float(np.sum(seg_blink) / (seg_sec + 1e-6) * 60.0)

        seg_head_motion = float(np.mean(head_motion[m]))
        seg_gaze_off = float(np.mean(gaze_dev[m] > 0.22))

        seg_touch_seq = np.array([feats[i].hand_touch_face for i in seg_idx], dtype=np.int32)
        seg_near_seq  = np.array([feats[i].hand_near_mouth for i in seg_idx], dtype=np.int32)
        seg_touch_stats = binary_events_stats(seg_touch_seq, SAMPLE_FPS)
        seg_near_stats  = binary_events_stats(seg_near_seq, SAMPLE_FPS)

        segment_cards_raw[seg_key] = {
            "happy_mean": float(np.mean(happy[m])),
            "sad_mean": float(np.mean(sad[m])),
            "tension_mean": float(np.mean(((fear + angry) / 2.0)[m])),
            "surprise_mean": float(np.mean(surprise[m])),

            "gaze_off_ratio": seg_gaze_off,
            "head_motion_mean": seg_head_motion,
            "blink_rate_per_min": seg_blink_rate,
            "face_ok_ratio": seg_face_ok_ratio,

            "touch_face_rate": float(seg_touch_stats["rate"]),
            "touch_face_count": float(seg_touch_stats["count"]),
            "touch_face_mean_sec": float(seg_touch_stats["mean_sec"]),

            "hand_near_mouth_rate": float(seg_near_stats["rate"]),
            "hand_near_mouth_count": float(seg_near_stats["count"]),
            "hand_near_mouth_mean_sec": float(seg_near_stats["mean_sec"]),
        }

    raw: Dict[str, Any] = {
        "mental_health_index": round(mental_health_index, 1),
        "confidence": round(confidence, 3),

        "happy_gain": round(float(happy_gain), 3),
        "happy_latency": None if happy_latency is None else round(float(happy_latency), 2),
        "sad_baseline": round(float(baseline_sad), 3),
        "tension_baseline": round(float(baseline_tension), 3),

        "surprise_peak": round(float(surprise_peak), 3),
        "calm_recover_speed": round(float(calm_recover_speed), 6),

        "gaze_off_ratio": round(float(gaze_off_ratio), 3),
        "head_motion_mean": round(float(head_motion_mean), 3),
        "blink_rate_per_min": round(float(blink_rate_per_min), 2),

        "touch_face_rate": round(float(touch_stats["rate"]), 3),
        "touch_face_count": int(round(float(touch_stats["count"]))),
        "touch_face_mean_sec": round(float(touch_stats["mean_sec"]), 2),

        "hand_near_mouth_rate": round(float(near_stats["rate"]), 3),
        "hand_near_mouth_count": int(round(float(near_stats["count"]))),
        "hand_near_mouth_mean_sec": round(float(near_stats["mean_sec"]), 2),

        "touch_face": {
            "rate": round(float(touch_stats["rate"]), 3),
            "count": int(round(float(touch_stats["count"]))),
            "total_sec": round(float(touch_stats["total_sec"]), 2),
            "mean_sec": round(float(touch_stats["mean_sec"]), 2),
            "max_sec": round(float(touch_stats["max_sec"]), 2),
        },
        "hand_near_mouth": {
            "rate": round(float(near_stats["rate"]), 3),
            "count": int(round(float(near_stats["count"]))),
            "total_sec": round(float(near_stats["total_sec"]), 2),
            "mean_sec": round(float(near_stats["mean_sec"]), 2),
            "max_sec": round(float(near_stats["max_sec"]), 2),
        },

        "qc_face_ok_ratio": round(float(face_ok_ratio), 3),
        "qc_n_samples": int(N),
        "qc_sample_fps": int(SAMPLE_FPS),

        "segment_cards_raw": segment_cards_raw,
    }
    return raw


# =========================================================
# 7) 输出层：10个大众指标（新增）
# =========================================================
def _score_good(x: float, lo: float, hi: float) -> float:
    """x 映射到 0~100（越大越好）"""
    if hi <= lo:
        return 50.0
    v = (x - lo) / (hi - lo)
    v = max(0.0, min(1.0, v))
    return float(round(v * 100.0, 0))


def _score_bad(x: float, lo: float, hi: float) -> float:
    """x 映射到 0~100（越小越好）"""
    if hi <= lo:
        return 50.0
    v = (x - lo) / (hi - lo)
    v = max(0.0, min(1.0, v))
    return float(round((1.0 - v) * 100.0, 0))

def _score_risk(x: float, lo: float, hi: float) -> float:
    """
    0~100 风险分（越大越“多/高/严重”）
    lo->0, hi->100（超出截断）
    """
    return _score_good(x, lo, hi)


def compute_simple_indicators(raw: Dict[str, Any], age_group: str) -> List[Dict[str, Any]]:
    """
    10 个大众指标（0-100）
    - 能力类：越高越好（稳定性/活力/积极反应/适应力/恢复力/专注/自信从容）
    - 风险类：越高越不好（压力水平/焦虑水平/低落程度）
    仅改输出层，不改底层检测。
    """
    seg = raw.get("segment_cards_raw", {}) or {}

    def seg_val(seg_key: str, k: str, default: float = 0.0) -> float:
        return float((seg.get(seg_key, {}) or {}).get(k, default))

    conf = float(raw.get("confidence", 0.0))
    sad_base = float(raw.get("sad_baseline", 0.0))
    tense_base = float(raw.get("tension_baseline", 0.0))
    gaze_off = float(raw.get("gaze_off_ratio", 1.0))
    head_motion = float(raw.get("head_motion_mean", 0.0))
    blink_rate = float(raw.get("blink_rate_per_min", 0.0))
    touch_rate = float(raw.get("touch_face_rate", 0.0))
    near_rate = float(raw.get("hand_near_mouth_rate", 0.0))
    happy_gain = float(raw.get("happy_gain", 0.0))
    happy_latency = raw.get("happy_latency", None)
    happy_latency = float(happy_latency) if happy_latency is not None else 10.0
    calm_slope = float(raw.get("calm_recover_speed", 0.0))  # 越负越好（更放松）

    seg3_tension = seg_val("seg3_more", "tension_mean", default=tense_base)
    seg3_gaze = seg_val("seg3_more", "gaze_off_ratio", default=gaze_off)
    seg6_tension = seg_val("seg6_calm", "tension_mean", default=tense_base)
    seg5_happy = seg_val("seg5_reward", "happy_mean", default=0.0)

    # 年龄阈值（可后续根据数据再标定）
    TH = {
        "9-12": {
            "tense_hi": 0.32, "sad_hi": 0.32,
            "gaze_hi": 0.45, "head_hi": 7.5, "blink_hi": 45.0,
            "touch_hi": 0.35, "near_hi": 0.22,
            "happy_gain_hi": 0.18, "latency_hi": 10.0,
            "seg3_delta_tense_hi": 0.12, "seg3_delta_gaze_hi": 0.18,
        },
        "13-15": {
            "tense_hi": 0.28, "sad_hi": 0.28,
            "gaze_hi": 0.38, "head_hi": 6.5, "blink_hi": 42.0,
            "touch_hi": 0.30, "near_hi": 0.20,
            "happy_gain_hi": 0.16, "latency_hi": 9.0,
            "seg3_delta_tense_hi": 0.10, "seg3_delta_gaze_hi": 0.16,
        },
        "16-18": {
            "tense_hi": 0.25, "sad_hi": 0.25,
            "gaze_hi": 0.33, "head_hi": 6.0, "blink_hi": 40.0,
            "touch_hi": 0.28, "near_hi": 0.18,
            "happy_gain_hi": 0.15, "latency_hi": 8.0,
            "seg3_delta_tense_hi": 0.09, "seg3_delta_gaze_hi": 0.14,
        }
    }[age_group]

    # ========= 能力类（越高越好） =========

    # 1) 情绪稳定性（越高越稳）
    emotion_stability = (
        0.50 * _score_bad(tense_base, 0.08, TH["tense_hi"]) +
        0.25 * _score_bad(head_motion, 0.5, TH["head_hi"]) +
        0.25 * _score_bad(gaze_off, 0.05, TH["gaze_hi"])
    )

    # 2) 活力（越高越有精神）
    vitality = (
        0.65 * _score_good(seg5_happy, 0.05, 0.35) +
        0.20 * _score_bad(gaze_off, 0.05, TH["gaze_hi"]) +
        0.15 * _score_bad(blink_rate, 8.0, TH["blink_hi"])
    )

    # 3) 积极情绪反应（越高越积极）
    positive_response = (
        0.70 * _score_good(happy_gain, -0.02, TH["happy_gain_hi"]) +
        0.30 * _score_bad(happy_latency, 0.0, TH["latency_hi"])
    )

    # 4) 适应力（越高越能适应复杂度增加）
    # seg3 与 baseline 的增幅越小越好，所以用 _score_bad
    delta_tense = seg3_tension - tense_base
    delta_gaze = seg3_gaze - gaze_off
    adaptability = (
        0.55 * _score_bad(delta_tense, -0.03, TH["seg3_delta_tense_hi"]) +
        0.45 * _score_bad(delta_gaze, -0.05, TH["seg3_delta_gaze_hi"])
    )

    # 5) 恢复力（越高越能放松恢复）
    # seg6紧张更低越好；calm_slope 越负越好（更放松趋势）
    recovery = (
        0.55 * _score_bad(seg6_tension, 0.06, TH["tense_hi"]) +
        0.45 * _score_bad(calm_slope, -0.004, 0.001)
    )

    # 6) 专注投入度（越高越专注）
    focus = (
        0.55 * _score_bad(gaze_off, 0.05, TH["gaze_hi"]) +
        0.30 * _score_good(conf, 0.70, 1.00) +
        0.15 * _score_bad(head_motion, 0.5, TH["head_hi"])
    )

    # 7) 自信/从容感（越高越从容）
    composure = (
        0.45 * _score_bad(tense_base, 0.08, TH["tense_hi"]) +
        0.25 * _score_bad(touch_rate, 0.03, TH["touch_hi"]) +
        0.15 * _score_bad(near_rate, 0.02, TH["near_hi"]) +
        0.15 * _score_good(seg5_happy, 0.05, 0.35)
    )

    # ========= 风险类（越高越不好） =========

    # 8) 压力水平（越高越“压力大”）
    # 用风险分：紧张表情、眨眼偏高、手部紧张动作偏多 -> 风险上升
    stress_level = (
        0.55 * _score_risk(tense_base, 0.08, TH["tense_hi"]) +
        0.20 * _score_risk(blink_rate, 8.0, TH["blink_hi"]) +
        0.15 * _score_risk(touch_rate, 0.03, TH["touch_hi"]) +
        0.10 * _score_risk(near_rate, 0.02, TH["near_hi"])
    )

    # 9) 焦虑水平（越高越“焦虑多”）
    anxiety_level = (
        0.45 * _score_risk(tense_base, 0.08, TH["tense_hi"]) +
        0.30 * _score_risk(touch_rate, 0.03, TH["touch_hi"]) +
        0.25 * _score_risk(near_rate, 0.02, TH["near_hi"])
    )

    # 10) 低落程度（越高越“低落明显”）
    # baseline 的 sad 越高越“低落”，同时奖励段开心越低也更像低落
    # 第二项用(1 - seg5_happy)的风险映射（seg5_happy 低 -> 风险高）
    low_mood = (
        0.70 * _score_risk(sad_base, 0.08, TH["sad_hi"]) +
        0.30 * _score_risk((0.35 - min(seg5_happy, 0.35)), 0.0, 0.30)
    )

    # ========= 置信度收敛：confidence低时，向50收敛，避免极端值 =========
    def shrink_by_conf(s: float) -> float:
        k = max(0.6, min(1.0, conf))
        v = 50.0 + (s - 50.0) * k
        # 保险 clamp（即使将来改权重也不会越界）
        return float(max(0.0, min(100.0, round(v, 0))))

    scores = {
        # 能力类（高好）
        "emotion_stability": shrink_by_conf(emotion_stability),
        "vitality": shrink_by_conf(vitality),
        "positive_response": shrink_by_conf(positive_response),
        "adaptability": shrink_by_conf(adaptability),
        "recovery": shrink_by_conf(recovery),
        "focus": shrink_by_conf(focus),
        "composure": shrink_by_conf(composure),

        # 风险类（高坏）
        "stress_level": shrink_by_conf(stress_level),
        "anxiety_level": shrink_by_conf(anxiety_level),
        "low_mood": shrink_by_conf(low_mood),
    }

    items = [
        ("emotion_stability", "情绪稳定性（越高越稳）"),
        ("stress_level", "压力水平（越高压力越大）"),
        ("anxiety_level", "焦虑水平（越高越焦虑）"),
        ("low_mood", "低落程度（越高越低落）"),
        ("positive_response", "积极情绪反应（越高越积极）"),
        ("vitality", "活力（越高越有精神）"),
        ("adaptability", "适应力（越高越能适应）"),
        ("recovery", "恢复力（越高越能放松恢复）"),
        ("focus", "专注投入度（越高越专注）"),
        ("composure", "自信/从容感（越高越从容）"),
    ]

    return [{"key": k, "name": n, "score": int(scores[k])} for k, n in items]



# =========================================================
# 8) format_for_age：只输出 10 个指标（替换原输出）
# =========================================================
def format_for_age(raw: Dict[str, Any], age_group: str) -> Dict[str, Any]:
    simple_indicators = compute_simple_indicators(raw, age_group)

    summary = {
        "overall_name": "整体状态参考",
        "overall_value": raw.get("mental_health_index", None),  # 可保留或前端隐藏
        "reliability_name": "本次结果可靠度",
        "reliability_value": raw.get("confidence", None),
        "note": "本结果为基于60秒视频的状态参考，不构成医学诊断。建议结合多次测量趋势与日常表现综合判断。"
    }

    return {
        "age_group": age_group,
        "summary": summary,
        "simple_indicators": simple_indicators,
    }

    # return {
    #     "age_group": age_group,
    #     "summary": summary,
    #     "simple_indicators": simple_indicators,

    #     # 仍保留时间线与段落raw（便于你后台调试/后续生成自然语言段落解释）
    #     "timeline": TIMELINE_6SEG,
    #     "segment_cards_raw": raw.get("segment_cards_raw", {}),

    #     # 原始底层指标保留（后端/管理员可见）
    #     "raw_metrics": raw,
    # }



# =========================================================
# 9) Flask API
# =========================================================
app = Flask(__name__)
app.json.ensure_ascii = False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "device": DEVICE,
        "vit_model_id": MODEL_ID,
        "face_task_path": FACE_TASK_PATH,
        "hand_task_path": HAND_TASK_PATH
    })


@app.route("/analyze_video", methods=["POST"])
def analyze_video_api():
    """
    Accept uploaded video (often named .mp4), probe real codec/container,
    normalize to CFR=15fps via ffmpeg, then run the analysis pipeline.
    """
    if "video" not in request.files:
        return jsonify({"ok": False, "error": "missing video file"}), 400

    age_group = (request.form.get("age_group", "9-12") or "9-12").strip()
    if age_group not in AGE_GROUPS:
        return jsonify({"ok": False, "error": f"invalid age_group, use one of {sorted(AGE_GROUPS)}"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"ok": False, "error": "empty filename"}), 400

    # Save as .mp4 suffix (even if content is actually webm/vp9 etc.)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(f.read())
        tmp_path = tmp.name

    normalized_path = None
    probe = None
    try:
        # 1) Probe the REAL container/codec (do not trust suffix)
        probe = ffprobe_video_info(tmp_path)

        # 2) Normalize to 15fps CFR MP4 (H.264 baseline) before analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out:
            normalized_path = out.name
        normalize_to_15fps_mp4(tmp_path, normalized_path)

        # 3) Run your existing analysis pipeline on the normalized file
        raw = analyze(normalized_path)
        report = format_for_age(raw, age_group)

        return jsonify({
            "ok": True,
            "report": report
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    finally:
        for p in (tmp_path, normalized_path):
            if p:
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
