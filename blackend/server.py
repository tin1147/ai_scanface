from pathlib import Path
import atexit
import json
import math
import os
import shutil
import tempfile
import threading
import time

# DeepFace CNN (optional — ถ้าไม่ติดตั้งจะ fallback เป็น blendshape อย่างเดียว)
try:
    from deepface import DeepFace as _DeepFace
    _DEEPFACE_AVAILABLE = True
except Exception:
    _DEEPFACE_AVAILABLE = False

_DEEPFACE_EN_TO_TH = {
    "angry":    "โกรธ",
    "disgust":  "รังเกียจ",
    "fear":     "กลัว",
    "happy":    "มีความสุข",
    "sad":      "เศร้า",
    "surprise": "ประหลาดใจ",
    "neutral":  "เฉยๆ",
}

try:
    from flask import Flask, jsonify, render_template, request
except ImportError as exc:
    raise RuntimeError(
        "Missing Flask. Install it with: python -m pip install flask"
    ) from exc


APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "model" / "face_landmarker.task"

server = Flask(
    __name__,
    template_folder=str(APP_ROOT / "frontend" / "templates"),
    static_folder=str(APP_ROOT / "frontend" / "static"),
)

_MEDIAPIPE_MODULES = None
_FACE_LANDMARKER = None
_FACE_LANDMARKER_INIT_LOCK = threading.Lock()
_FACE_LANDMARKER_DETECT_LOCK = threading.Lock()
# หมายเหตุ: lock นี้ทำให้ detect ได้ทีละ 1 ภาพ/thread-safe
# หากมีผู้ใช้พร้อมกันจำนวนมาก ควรพิจารณา queue worker (เช่น Celery) ในอนาคต
_RUNTIME_MODEL_DIR = None
_RUNTIME_MODEL_PATH = None
_RUNTIME_MODEL_LOCK = threading.Lock()

MAX_IMAGE_SIZE   = 10 * 1024 * 1024  # 10 MB
MAX_SHOTS        = 3                  # multi-shot: รับสูงสุด 3 รูป
CONFIDENCE_HIGH  = 0.08               # margin ≥ 8% → มั่นใจสูง
CONFIDENCE_MED   = 0.04               # margin ≥ 4% → มั่นใจปานกลาง
FEEDBACK_FILE    = Path(__file__).parent / "feedback.jsonl"
CLASSIFIER_PATH  = Path(__file__).parent / "classifier.pkl"

# ─── น้ำหนัก blend clf/rules (ปรับแบบ dynamic ตาม clf confidence) ───
CLF_WEIGHT_HIGH = 0.92   # clf มั่นใจ (prob > 0.65) → ใช้ clf เกือบทั้งหมด (เพิ่มจาก 0.85)
CLF_WEIGHT_LOW  = 0.60   # clf ไม่มั่นใจ → blend 60/40
CLF_WEIGHT_DISAGREE = 0.55  # clf กับ rules ขัดกัน แต่ clf มั่นใจ → ใช้ clf เป็นหลัก (55%)


def compute_interaction_features(bs):
    """
    คำนวณ interaction features จาก raw blendshape scores
    แต่ละ feature = ผลคูณของ key signals → discriminative กว่า single feature มาก
    ใช้ทั้งใน inference (get_classifier_probs) และ training (train_classifier.py)
    """
    def avg(l, r): return (bs.get(l, 0.0) + bs.get(r, 0.0)) / 2.0

    smile    = avg("mouthSmileLeft",   "mouthSmileRight")
    cheek    = avg("cheekSquintLeft",  "cheekSquintRight")
    brow_dn  = avg("browDownLeft",     "browDownRight")
    brow_out = avg("browOuterUpLeft",  "browOuterUpRight")
    eye_wide = avg("eyeWideLeft",      "eyeWideRight")
    eye_sq   = avg("eyeSquintLeft",    "eyeSquintRight")
    frown    = avg("mouthFrownLeft",   "mouthFrownRight")
    sneer    = avg("noseSneerLeft",    "noseSneerRight")
    upper_up = avg("mouthUpperUpLeft", "mouthUpperUpRight")
    press    = avg("mouthPressLeft",   "mouthPressRight")
    stretch  = avg("mouthStretchLeft", "mouthStretchRight")
    lower_dn = avg("mouthLowerDownLeft", "mouthLowerDownRight")

    brow_in  = bs.get("browInnerUp",    0.0)
    jaw_open = bs.get("jawOpen",        0.0)
    pucker   = bs.get("mouthPucker",    0.0)
    shrug    = bs.get("mouthShrugLower",0.0)

    return {
        # มีความสุข — Duchenne smile (ยิ้มจริง: ปากยิ้ม + แก้มยก)
        "IX_duchenne":        smile * cheek,
        # มีความสุข — eyeSquint + smile (ตาหยี + ยิ้ม = ความสุขชัดเจน)
        "IX_squint_smile":    eye_sq * smile,
        # โกรธ — browDown × mouthPucker (คิ้วกด + ปากห่อ = โกรธแท้)
        "IX_anger_core":      brow_dn * pucker,
        # โกรธ — browDown × press (กัดฟัน)
        "IX_anger_press":     brow_dn * press,
        # เศร้า — browInnerUp × frown (หัวคิ้วยก + ปากตก = เศร้าแท้)
        "IX_sad_core":        brow_in * frown,
        # เศร้า — frown × shrug (ปากตก + คาง = เศร้าลึก)
        "IX_sad_chin":        frown * shrug,
        # กลัว — eyeWide × browOuter ไม่อ้าปาก (กลัวแบบเกร็ง)
        "IX_fear_core":       eye_wide * brow_out * max(0.0, 1.0 - jaw_open * 1.5),
        # กลัว — stretch × eyeWide (ปากยืด + ตาเบิก)
        "IX_fear_stretch":    stretch * eye_wide,
        # ประหลาดใจ — eyeWide × jawOpen (ตาเบิก + อ้าปาก)
        "IX_surprise_core":   eye_wide * jaw_open,
        # รังเกียจ — noseSneer × upperLip (จมูกย่น + ริมฝีปากบนยก)
        "IX_disgust_core":    sneer * upper_up,
        # รังเกียจ — sneer × lowerLip
        "IX_disgust_lower":   sneer * lower_dn,
        # เฉยๆ / หน้าตึง — browDown แต่ไม่มี pucker/frown
        "IX_tense_neutral":   brow_dn * max(0.0, 1.0 - pucker) * max(0.0, 1.0 - frown),
        # กลัวแบบ nervous smile — ยิ้มแต่ไม่ Duchenne
        "IX_nervous_smile":   smile * max(0.0, 1.0 - cheek * 1.5),
        # ตาเบิก - ตาหรี่ (fear vs happy)
        "IX_wide_vs_squint":  max(0.0, eye_wide - eye_sq * 0.5),
        # browInner contrast (เศร้า vs โกรธ)
        "IX_inner_vs_down":   max(0.0, brow_in - brow_dn * 0.8),
    }

# ─── Trained sklearn classifier (โหลดตอน startup ถ้า classifier.pkl มีอยู่) ───
_CLF_MODEL    = None   # sklearn Pipeline
_CLF_FEATURES = None   # list of blendshape feature names (ลำดับที่ train)
_CLF_LOCK     = threading.Lock()


def _load_trained_classifier():
    global _CLF_MODEL, _CLF_FEATURES
    if not CLASSIFIER_PATH.exists():
        return
    try:
        import pickle, shutil, tempfile as _tmp
        # copy ออกมาเพราะ path มี Thai char
        tmp = _tmp.mktemp(suffix=".pkl")
        shutil.copy2(CLASSIFIER_PATH, tmp)
        with open(tmp, "rb") as f:
            data = pickle.load(f)
        Path(tmp).unlink(missing_ok=True)
        with _CLF_LOCK:
            _CLF_MODEL    = data["model"]
            _CLF_FEATURES = data["features"]
    except Exception as exc:
        pass  # classifier ใช้งานไม่ได้ → fallback rules


def get_classifier_probs(blendshape_scores):
    """คืน {thai_emotion: prob} จาก trained classifier หรือ None
    ใช้ raw blendshape + interaction features (ต้องตรงกับที่ train)
    """
    with _CLF_LOCK:
        model    = _CLF_MODEL
        features = _CLF_FEATURES
    if model is None or features is None:
        return None
    try:
        import numpy as np
        # รวม raw + interaction features
        expanded = {**blendshape_scores, **compute_interaction_features(blendshape_scores)}
        X = np.array([[expanded.get(f, 0.0) for f in features]])
        probs  = model.predict_proba(X)[0]
        labels = model.classes_
        return dict(zip(labels, probs.tolist()))
    except Exception:
        return None


_load_trained_classifier()  # โหลดตอน module import

EMOTION_EMOJIS = {
    "มีความสุข": "😊", "เศร้า": "😢", "โกรธ": "😠",
    "กลัว": "😨", "รังเกียจ": "🤢", "ประหลาดใจ": "😲", "เฉยๆ": "😐",
}
EMOTION_EN = {
    "มีความสุข": "happy", "เศร้า": "sad", "โกรธ": "angry",
    "กลัว": "fear", "รังเกียจ": "disgust", "ประหลาดใจ": "surprise", "เฉยๆ": "neutral",
}
FACE_CROP_PADDING = 0.35  # เผื่อขอบรอบใบหน้า (คิ้ว/คาง) ให้ landmark ครบขึ้น
MIN_FACE_CROP_PX = 320  # ครอปเล็กกว่านี้จะ upscale ก่อน detect รอบ 2
MIN_INPUT_DIMENSION = 640  # ภาพเล็กเกินไปจะ upscale ก่อน detect รอบ 1
MIN_DETECTION_CONFIDENCE = 0.55  # กรอง detection ที่ไม่มั่นใจ
SOFTMAX_TEMPERATURE = 5.5  # เพิ่มจาก 4.5 → peak โดดชัดขึ้น ให้ dominant emotion % สูงขึ้น
PENALTY_STRENGTH = 0.45  # ลดผล penalty ให้นุ่มนวล ไม่บิดอารมณ์เกินจริง
SIGNATURE_ADJUSTMENT_SCALE = 0.38  # เพิ่มจาก 0.28 → interaction features มีน้ำหนักมากขึ้น
NERVOUS_FEAR_SCALE = 0.65         # เพิ่มจาก 0.55 → nervous smile → กลัว แม่นขึ้น

# คู่ blendshape ซ้าย/ขวา — ใช้ค่าเฉลี่ยลด noise จากมุมกล้อง
BLENDSHAPE_SYMMETRY_PAIRS = (
    ("browDownLeft", "browDownRight"),
    ("browOuterUpLeft", "browOuterUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("eyeWideLeft", "eyeWideRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("noseSneerLeft", "noseSneerRight"),
)

BLENDSHAPE_LABELS = {
    "browDownLeft": "คิ้วซ้ายกดลง",
    "browDownRight": "คิ้วขวากดลง",
    "browInnerUp": "หัวคิ้วยกขึ้น",
    "browOuterUpLeft": "ปลายคิ้วซ้ายยกขึ้น",
    "browOuterUpRight": "ปลายคิ้วขวายกขึ้น",
    "cheekSquintLeft": "แก้มซ้ายยก/ตาหยี",
    "cheekSquintRight": "แก้มขวายก/ตาหยี",
    "eyeSquintLeft": "ตาซ้ายหรี่",
    "eyeSquintRight": "ตาขวาหรี่",
    "eyeWideLeft": "ตาซ้ายเบิกกว้าง",
    "eyeWideRight": "ตาขวาเบิกกว้าง",
    "jawOpen": "อ้าปาก",
    "mouthFrownLeft": "มุมปากซ้ายตก",
    "mouthFrownRight": "มุมปากขวาตก",
    "mouthLowerDownLeft": "ริมฝีปากล่างซ้ายตก",
    "mouthLowerDownRight": "ริมฝีปากล่างขวาตก",
    "mouthPressLeft": "กดริมฝีปากซ้าย",
    "mouthPressRight": "กดริมฝีปากขวา",
    "mouthPucker": "ห่อปาก",
    "mouthSmileLeft": "มุมปากซ้ายยิ้ม",
    "mouthSmileRight": "มุมปากขวายิ้ม",
    "mouthStretchLeft": "มุมปากซ้ายยืด",
    "mouthStretchRight": "มุมปากขวายืด",
    "mouthUpperUpLeft": "ริมฝีปากบนซ้ายยก",
    "mouthUpperUpRight": "ริมฝีปากบนขวายก",
    "mouthShrugLower": "คาง/ริมฝีปากล่างยก",
    "noseSneerLeft": "จมูกซ้ายย่น",
    "noseSneerRight": "จมูกขวาย่น",
}

# น้ำหนักอารมณ์: คุมช่วง 0.0–1.0 เพื่อความนุ่มนวล (ลดหน้าเบี้ยว)
# เน้นดวงตา/คิ้วมากกว่าปาก — เหมาะกับการแสดงออกแบบคนไทย
# Left/Right ตั้งเท่ากันก่อน symmetrize_blendshape_scores จัดการ noise เล็กน้อย
EMOTION_RULES = {
    "มีความสุข": {
        "mouthSmileLeft":   0.9,
        "mouthSmileRight":  0.9,
        "cheekSquintLeft":  0.6,
        "cheekSquintRight": 0.6,
    },
    "เศร้า": {
        "browInnerUp":     1.0,
        "mouthFrownLeft":  0.9,
        "mouthFrownRight": 0.9,
        "eyeSquintLeft":   0.3,
        "eyeSquintRight":  0.3,
        "mouthShrugLower": 0.45,
    },
    "โกรธ": {
        "browDownLeft":    0.85,
        "browDownRight":   0.85,
        "eyeSquintLeft":   0.25,
        "eyeSquintRight":  0.25,
        "mouthPressLeft":  0.55,
        "mouthPressRight": 0.55,
        "mouthPucker":     0.80,  # เพิ่มจาก 0.65 → key discriminator จาก หน้าตึง/เฉยๆ (ซึ่งไม่ห่อปาก)
        "noseSneerLeft":   0.35,
        "noseSneerRight":  0.35,
    },
    "ประหลาดใจ": {
        "eyeWideLeft":      1.0,
        "eyeWideRight":     1.0,
        "jawOpen":          0.9,
        "browOuterUpLeft":  0.7,
        "browOuterUpRight": 0.7,
        "browInnerUp":      0.5,
    },
    "กลัว": {
        "eyeWideLeft":       1.0,
        "eyeWideRight":      1.0,
        "browOuterUpLeft":   1.0,
        "browOuterUpRight":  1.0,
        "browInnerUp":       0.80,  # เพิ่มจาก 0.65 → key AU1 differentiator จาก ประหลาดใจ
        "mouthStretchLeft":  0.55,
        "mouthStretchRight": 0.55,
        "browDownLeft":      0.40,  # กลัวแบบเกร็ง — คิ้วกดลงจากความเครียด
        "browDownRight":     0.40,
        "eyeSquintLeft":     0.35,  # กลัวแบบหรี่ตา — ตึงจากความกลัว
        "eyeSquintRight":    0.35,
    },
    "รังเกียจ": {
        "noseSneerLeft":        1.0,
        "noseSneerRight":       1.0,
        "mouthUpperUpLeft":     0.7,
        "mouthUpperUpRight":    0.7,
        "mouthLowerDownLeft":   0.5,
        "mouthLowerDownRight":  0.5,
        "mouthFrownLeft":       0.35,
        "mouthFrownRight":      0.35,
        "eyeSquintLeft":        0.4,
        "eyeSquintRight":       0.4,
    },
}

# สัญญาณที่ขัดกับอารมณ์ — คุมช่วง 0.0–1.0 เช่นกัน
EMOTION_PENALTIES = {
    "มีความสุข": {
        "mouthFrownLeft":  0.8,
        "mouthFrownRight": 0.8,
        "browDownLeft":    0.75,
        "browDownRight":   0.75,
        "eyeWideLeft":     0.60,  # เพิ่มจาก 0.4 → ตาเบิกกว้าง = กลัว/ตกใจ ไม่ใช่ความสุข
        "eyeWideRight":    0.60,
        "noseSneerLeft":   0.55,
        "noseSneerRight":  0.55,
    },
    "เศร้า": {
        "mouthSmileLeft":    0.9,
        "mouthSmileRight":   0.9,
        "cheekSquintLeft":   0.5,
        "cheekSquintRight":  0.5,
        "eyeWideLeft":       0.95,
        "eyeWideRight":      0.95,
        "mouthStretchLeft":  0.7,
        "mouthStretchRight": 0.7,
        "browDownLeft":      0.45,
        "browDownRight":     0.45,
        "noseSneerLeft":     0.6,
        "noseSneerRight":    0.6,
        "jawOpen":           0.7,
        "browOuterUpLeft":   0.70,   # เพิ่ม: outer brow up = กลัว ไม่ใช่เศร้า
        "browOuterUpRight":  0.70,
    },
    "โกรธ": {
        "mouthSmileLeft":    0.8,
        "mouthSmileRight":   0.8,
        "browInnerUp":       0.6,    # หัวคิ้วยก (เศร้า/ตกใจ ไม่ใช่โกรธ)
        "cheekSquintLeft":   0.75,   # ยิ้มตาหยี (Duchenne) — กรองไม่ให้หลุดเป็นโกรธ
        "cheekSquintRight":  0.75,
        "jawOpen":           0.3,    # โกรธแบบกัดฟันมักไม่อ้าปากกว้าง
        "mouthUpperUpLeft":  0.6,    # ริมฝีปากบนยก = รังเกียจ ไม่ใช่โกรธ
        "mouthUpperUpRight": 0.6,
    },
    "ประหลาดใจ": {
        "mouthPressLeft":    0.5,
        "mouthPressRight":   0.5,
        "browDownLeft":      0.6,
        "browDownRight":     0.6,
        "mouthFrownLeft":    0.4,
        "mouthFrownRight":   0.4,
        "eyeSquintLeft":     0.5,
        "eyeSquintRight":    0.5,
        "mouthStretchLeft":  0.55,  # stretch แบบแนวนอน = กลัว ไม่ใช่ ประหลาดใจ
        "mouthStretchRight": 0.55,
        "noseSneerLeft":     0.45,  # จมูกย่น = รังเกียจ ไม่ใช่ ประหลาดใจ
        "noseSneerRight":    0.45,
    },
    "กลัว": {
        # ลบ browDown ออก — เพราะเพิ่ม browDown ใน rules แล้ว (เป็น signal ได้)
        "mouthSmileLeft":    0.45,  # ลดจาก 0.8 — nervous smile เป็น sign กลัวได้
        "mouthSmileRight":   0.45,
        "eyeSquintLeft":     0.25,
        "eyeSquintRight":    0.25,
        "mouthFrownLeft":    0.5,
        "mouthFrownRight":   0.5,
        "cheekSquintLeft":   0.4,
        "cheekSquintRight":  0.4,
        "mouthPucker":       0.6,
        "mouthPressLeft":    0.5,
        "mouthPressRight":   0.5,
        "jawOpen":           0.5,
        "noseSneerLeft":     0.5,
        "noseSneerRight":    0.5,
    },
    "รังเกียจ": {
        "mouthSmileLeft":    0.9,
        "mouthSmileRight":   0.9,
        "eyeWideLeft":       0.6,
        "eyeWideRight":      0.6,
        "jawOpen":           0.5,
        "browOuterUpLeft":   0.45,
        "browOuterUpRight":  0.45,
        "browInnerUp":       0.35,
    },
}


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup():
    global _FACE_LANDMARKER, _RUNTIME_MODEL_DIR

    with _FACE_LANDMARKER_INIT_LOCK:
        if _FACE_LANDMARKER is not None:
            try:
                _FACE_LANDMARKER.close()
            except Exception:
                pass
            _FACE_LANDMARKER = None

    with _RUNTIME_MODEL_LOCK:
        if _RUNTIME_MODEL_DIR is not None:
            try:
                shutil.rmtree(_RUNTIME_MODEL_DIR, ignore_errors=True)
            except Exception:
                pass
            _RUNTIME_MODEL_DIR = None


atexit.register(_cleanup)


# ---------------------------------------------------------------------------
# Module / model loading
# ---------------------------------------------------------------------------

def load_mediapipe_modules():
    global _MEDIAPIPE_MODULES
    if _MEDIAPIPE_MODULES is None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "Missing MediaPipe. Install it with: python -m pip install mediapipe"
            ) from exc
        _MEDIAPIPE_MODULES = (mp, python, vision)
    return _MEDIAPIPE_MODULES


def get_runtime_model_path():
    global _RUNTIME_MODEL_PATH, _RUNTIME_MODEL_DIR

    with _RUNTIME_MODEL_LOCK:
        if _RUNTIME_MODEL_PATH is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Face landmarker model not found: {MODEL_PATH}"
                )
            _RUNTIME_MODEL_DIR = tempfile.mkdtemp(prefix="face_landmarker_")
            runtime_model_path = Path(_RUNTIME_MODEL_DIR) / MODEL_PATH.name
            shutil.copy2(MODEL_PATH, runtime_model_path)
            _RUNTIME_MODEL_PATH = runtime_model_path

    return _RUNTIME_MODEL_PATH


def get_face_landmarker():
    global _FACE_LANDMARKER

    with _FACE_LANDMARKER_INIT_LOCK:
        if _FACE_LANDMARKER is None:
            _, mp_tasks, vision = load_mediapipe_modules()
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(
                    model_asset_path=str(get_runtime_model_path())
                ),
                running_mode=vision.RunningMode.IMAGE,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
            )

            original_cwd = os.getcwd()
            try:
                os.chdir(tempfile.gettempdir())
                _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
            finally:
                os.chdir(original_cwd)

    return _FACE_LANDMARKER


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _ensure_rgb(frame_np):
    """แปลง grayscale (2D หรือ HxWx1) → RGB เพื่อให้ MediaPipe และ PIL รับได้"""
    import numpy as np

    if frame_np.ndim == 2:
        return np.stack([frame_np] * 3, axis=-1)
    if frame_np.ndim == 3 and frame_np.shape[2] == 1:
        return np.concatenate([frame_np] * 3, axis=-1)
    return frame_np


def _numpy_to_mp_image(frame_np):
    import numpy as np
    mp, _, _ = load_mediapipe_modules()
    frame_np = _ensure_rgb(frame_np)
    if frame_np.shape[2] == 4:
        image_format = mp.ImageFormat.SRGBA
    else:
        image_format = mp.ImageFormat.SRGB
    return mp.Image(image_format=image_format, data=np.ascontiguousarray(frame_np))


def _mp_image_to_numpy(mp_image):
    return mp_image.numpy_view().copy()


def _upscale_numpy_frame(frame_np, min_dimension):
    import numpy as np

    frame_np = _ensure_rgb(frame_np)
    height, width = frame_np.shape[:2]
    longest = max(height, width)
    if longest >= min_dimension:
        return frame_np

    scale = min_dimension / longest
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))

    try:
        from PIL import Image

        pil_mode = "RGBA" if frame_np.shape[2] == 4 else "RGB"
        resized = (
            Image.fromarray(frame_np.astype(np.uint8), mode=pil_mode)
            .resize((new_w, new_h), Image.Resampling.LANCZOS)
        )
        return np.asarray(resized, dtype=np.uint8)
    except ImportError:
        y_idx = (np.linspace(0, height - 1, new_h)).astype(np.int32)
        x_idx = (np.linspace(0, width - 1, new_w)).astype(np.int32)
        return frame_np[y_idx][:, x_idx]


def _enhance_contrast(frame_np):
    """ยก contrast เล็กน้อยช่วยให้ landmark ชัดขึ้นในภาพมืด/จาง"""
    import numpy as np

    if frame_np.dtype != np.uint8:
        return frame_np

    rgb = frame_np[..., :3].astype(np.float32)
    mean = rgb.mean()
    if 70 <= mean <= 185:
        return frame_np

    alpha = 1.12
    beta = 8.0 if mean < 70 else -6.0
    enhanced = np.clip(alpha * rgb + beta, 0, 255).astype(np.uint8)
    if frame_np.shape[2] == 4:
        out = frame_np.copy()
        out[..., :3] = enhanced
        return out
    return enhanced


def preprocess_mp_image(mp_image, min_dimension=MIN_INPUT_DIMENSION):
    frame = _enhance_contrast(_mp_image_to_numpy(mp_image))
    upscaled = _upscale_numpy_frame(frame, min_dimension)
    if upscaled is frame:
        return mp_image
    return _numpy_to_mp_image(upscaled)


# ---------------------------------------------------------------------------
# Face crop
# ---------------------------------------------------------------------------

def crop_face_from_image(mp_image, padding=FACE_CROP_PADDING):
    """
    รอบที่ 1: detect หน้าคร่าว ๆ จากภาพเต็ม
    → คำนวณ bounding box จาก landmarks
    → ครอปเฉพาะส่วนหน้า + padding
    → คืน mp.Image ที่ครอปแล้ว พร้อม crop_info สำหรับ debug

    คืน (cropped_image, crop_info) หรือ (None, None) ถ้าไม่เจอหน้า
    """
    mp, _, _ = load_mediapipe_modules()

    # detect รอบแรก (ไม่เอา blendshapes ยังไม่จำเป็น)
    with _FACE_LANDMARKER_DETECT_LOCK:
        first_result = get_face_landmarker().detect(mp_image)

    if not first_result.face_landmarks:
        return None, None

    img_h, img_w = mp_image.height, mp_image.width
    landmarks = first_result.face_landmarks[0]

    # หา bounding box จาก normalized landmarks (0–1)
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # เพิ่ม padding รอบ bounding box (บนมากกว่าล่างเล็กน้อย — คิ้วสำคัญกับอารมณ์)
    face_w = x_max - x_min
    face_h = y_max - y_min
    pad_x = face_w * padding
    pad_top = face_h * (padding + 0.08)
    pad_bottom = face_h * (padding - 0.03)
    x_min = max(0.0, x_min - pad_x)
    x_max = min(1.0, x_max + pad_x)
    y_min = max(0.0, y_min - pad_top)
    y_max = min(1.0, y_max + pad_bottom)

    # ทำให้ crop เป็นสี่เหลี่ยมจัตุรัส — โมเดล detect ได้เสถียรกว่า
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min = max(0.0, cx - half)
    x_max = min(1.0, cx + half)
    y_min = max(0.0, cy - half)
    y_max = min(1.0, cy + half)

    # แปลงเป็น pixel
    px_x1 = int(x_min * img_w)
    px_y1 = int(y_min * img_h)
    px_x2 = int(x_max * img_w)
    px_y2 = int(y_max * img_h)

    crop_w = px_x2 - px_x1
    crop_h = px_y2 - px_y1
    if crop_w < 32 or crop_h < 32:
        return None, None

    import numpy as np
    frame = mp_image.numpy_view()
    cropped_np = frame[px_y1:px_y2, px_x1:px_x2].copy()
    cropped_np = _upscale_numpy_frame(cropped_np, MIN_FACE_CROP_PX)
    cropped_mp = _numpy_to_mp_image(cropped_np)

    crop_w, crop_h = cropped_np.shape[1], cropped_np.shape[0]

    crop_info = {
        "x1": px_x1, "y1": px_y1,
        "x2": px_x2, "y2": px_y2,
        "original_size": f"{img_w}x{img_h}",
        "crop_size": f"{crop_w}x{crop_h}",
    }
    return cropped_mp, crop_info


# ---------------------------------------------------------------------------
# Detection (2-pass)
# ---------------------------------------------------------------------------

def detect_image_suffix(image_bytes, filename=""):
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return ".jpg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return ".gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return ".webp"
    if image_bytes.startswith(b"BM"):
        return ".bmp"
    return ".img"


def run_face_landmarker(image_bytes, filename=""):
    """
    2-pass detection:
      Pass 1 — detect หน้าจากภาพเต็ม → ครอป bounding box + padding
      Pass 2 — detect อีกครั้งบนภาพครอป → blendshapes แม่นขึ้น
    ถ้าครอปไม่ได้ (ไม่เจอหน้าในรอบแรก) จะ fallback เป็น 1-pass
    """
    if not image_bytes:
        raise ValueError("ไฟล์ภาพว่างเปล่า")

    mp, _, _ = load_mediapipe_modules()
    temp_path = None
    suffix = detect_image_suffix(image_bytes, filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_bytes)
            temp_path = Path(temp_file.name)

        full_image = mp.Image.create_from_file(str(temp_path))
        full_image = preprocess_mp_image(full_image)

        # Pass 1: ครอปหน้า
        cropped_image, crop_info = crop_face_from_image(full_image)

        if cropped_image is not None:
            # Pass 2: detect บนภาพครอป
            with _FACE_LANDMARKER_DETECT_LOCK:
                result = get_face_landmarker().detect(cropped_image)
            return result, crop_info
        else:
            # Fallback: ใช้ภาพเต็ม
            with _FACE_LANDMARKER_DETECT_LOCK:
                result = get_face_landmarker().detect(full_image)
            return result, None

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def categories_to_scores(categories):
    return {
        category.category_name: float(category.score)
        for category in categories
        if category.category_name
    }


def symmetrize_blendshape_scores(scores):
    """เฉลี่ยคู่ซ้าย/ขวา ลด bias จากมุมกล้อง"""
    adjusted = dict(scores)
    for left, right in BLENDSHAPE_SYMMETRY_PAIRS:
        if left in adjusted and right in adjusted:
            avg = (adjusted[left] + adjusted[right]) / 2.0
            adjusted[left] = avg
            adjusted[right] = avg
    return adjusted


def relative_blendshape_scores(scores):
    """ลบ baseline ของใบหน้า — เน้นส่วนที่เคลื่อนไหวจริง ป้องกัน IndexError"""
    if not scores:
        return {}

    values = list(scores.values())
    if not values:
        return {}

    try:
        baseline = sorted(values)[len(values) // 2]
    except IndexError:
        baseline = 0.0

    return {name: max(0.0, value - baseline * 0.65) for name, value in scores.items()}


def weighted_score(scores, weight_map):
    total_weight = sum(weight_map.values())
    if total_weight == 0:
        return 0.0
    return sum(scores.get(name, 0.0) * w for name, w in weight_map.items()) / total_weight


def penalty_score(scores, penalty_map):
    total_weight = sum(penalty_map.values())
    if total_weight == 0:
        return 0.0
    return sum(scores.get(name, 0.0) * w for name, w in penalty_map.items()) / total_weight


def _blend_pair_mean(scores, left, right):
    return (scores.get(left, 0.0) + scores.get(right, 0.0)) / 2.0


def fear_signature_adjustment(scores):
    """
    กลัวคลาสสิก: ตาเบิก + คิ้วนอกยก + ปากยืด แยกจากประหลาดใจ (อ้าปากกว้าง)
    กลัวแบบเกร็ง: ยิ้มตึง (non-Duchenne) + คิ้วกด + ตาหรี่ — พบเมื่อกลัว/กังวลแบบ suppressed
    """
    outer_brow = _blend_pair_mean(scores, "browOuterUpLeft", "browOuterUpRight")
    eye_wide = _blend_pair_mean(scores, "eyeWideLeft", "eyeWideRight")
    mouth_stretch = _blend_pair_mean(scores, "mouthStretchLeft", "mouthStretchRight")
    eye_squint = _blend_pair_mean(scores, "eyeSquintLeft", "eyeSquintRight")
    jaw_open = scores.get("jawOpen", 0.0)
    brow_down = _blend_pair_mean(scores, "browDownLeft", "browDownRight")
    smile = _blend_pair_mean(scores, "mouthSmileLeft", "mouthSmileRight")
    cheek_squint = _blend_pair_mean(scores, "cheekSquintLeft", "cheekSquintRight")

    # กลัวคลาสสิก
    eye_signal = max(0.0, eye_wide - eye_squint * 0.5)
    surprise_like_jaw = max(0.0, jaw_open - eye_wide * 0.85)
    classic_raw = max(0.0, outer_brow * 0.35 + eye_signal * 0.4 + mouth_stretch * 0.25 - surprise_like_jaw * 0.3)

    # กลัวแบบ "worried brow" (AU1+4): browInnerUp + browDown ร่วมกัน
    # พบใน AffectNet fear ที่ไม่มี eyeWide ชัด — บ่งชี้ความกังวล/หวาดกลัวแบบ suppressed
    inner_brow = scores.get("browInnerUp", 0.0)
    worried_brow_raw = min(inner_brow, brow_down) * 0.8  # ต้องมีทั้งสองอย่าง
    # discount ถ้ามี mouthFrown (นั่นคือเศร้า ไม่ใช่กลัว)
    mouth_frown_local = _blend_pair_mean(scores, "mouthFrownLeft", "mouthFrownRight")
    worried_brow_raw = worried_brow_raw * max(0.0, 1.0 - mouth_frown_local * 3.0)

    # กลัวแบบเกร็ง: ยิ้มตึง (non-Duchenne) + browDown + eyeSquint
    # ใช้ mouthUpperUp เป็น discriminator — ยิ้มที่เห็นฟัน (genuine/big smile) มี upperUp สูง
    upper_lip = _blend_pair_mean(scores, "mouthUpperUpLeft", "mouthUpperUpRight")
    genuine_smile_discount = upper_lip * 1.2  # ยิ้มจริงที่เห็นฟัน → ลด nervous signal
    non_duchenne = max(0.0, smile - cheek_squint * 1.2 - genuine_smile_discount)
    moderate_factor = max(0.0, 1.0 - smile * 0.8)  # ยิ้มกว้างมาก → ไม่ใช่ nervous
    nervous_raw = max(0.0, non_duchenne * brow_down * 2.5 + brow_down * eye_squint * 0.8) * moderate_factor

    worried_scale = SIGNATURE_ADJUSTMENT_SCALE * 0.9
    return max(0.0, classic_raw * SIGNATURE_ADJUSTMENT_SCALE
               + nervous_raw * NERVOUS_FEAR_SCALE
               + worried_brow_raw * worried_scale)


def sadness_signature_adjustment(scores):
    """
    ปรับละเอียดเศร้า: คิ้วใน + มุมปากตก + ตาหม่น
    แยกจากกลัว (ตาเบิก/คิ้วนอกยก) และโกรธ (คิ้วกด/จมูกย่น)

    KEY FIX: discount sad_brow ด้วย fear signals (eyeWide + browOuterUp)
    เพื่อป้องกัน fear faces (browInnerUp สูง) ถูกเรียกว่าเศร้า
    """
    mouth_frown  = _blend_pair_mean(scores, "mouthFrownLeft",    "mouthFrownRight")
    inner_brow   = scores.get("browInnerUp", 0.0)
    brow_down    = _blend_pair_mean(scores, "browDownLeft",       "browDownRight")
    brow_out     = _blend_pair_mean(scores, "browOuterUpLeft",    "browOuterUpRight")
    eye_squint   = _blend_pair_mean(scores, "eyeSquintLeft",      "eyeSquintRight")
    eye_wide     = _blend_pair_mean(scores, "eyeWideLeft",        "eyeWideRight")
    mouth_smile  = _blend_pair_mean(scores, "mouthSmileLeft",     "mouthSmileRight")
    mouth_stretch= _blend_pair_mean(scores, "mouthStretchLeft",   "mouthStretchRight")
    nose_sneer   = _blend_pair_mean(scores, "noseSneerLeft",      "noseSneerRight")
    chin_raise   = scores.get("mouthShrugLower", 0.0)

    # ลด sad_brow เมื่อ fear signals ดัง แต่ปากไม่แสดงความเศร้า
    # Logic: กลัว → browInnerUp สูง แต่ปากไม่ตก (mouth_evidence ต่ำ)
    #        เศร้า → browInnerUp สูง AND ปากตก/คางยก (mouth_evidence สูง)
    fear_signal   = eye_wide * 2.5 + brow_out * 2.0 + mouth_stretch * 1.0
    mouth_evidence = mouth_frown * 2.5 + chin_raise * 1.5  # ปากยืนยันว่าเศร้า
    # discount ก็ต่อเมื่อ fear_signal > mouth_evidence (ปากไม่ยืนยันเศร้า)
    effective_discount = max(0.0, fear_signal - mouth_evidence)
    sad_brow_raw  = max(0.0, inner_brow - brow_down * 0.6)
    sad_brow      = max(0.0, sad_brow_raw - effective_discount)

    droopy_eyes   = max(0.0, eye_squint * 0.5 - eye_wide * 0.7)
    not_happy     = max(0.0, mouth_frown - mouth_smile)
    not_fear      = max(0.0, sad_brow + mouth_frown * 0.4 - eye_wide * 2.0 - brow_out * 1.5 - mouth_stretch * 0.8)
    not_anger     = max(0.0, sad_brow - nose_sneer * 0.5 - brow_down * 0.4)

    raw = (
        mouth_frown * 0.30
        + sad_brow  * 0.35
        + droopy_eyes * 0.20
        + chin_raise  * 0.20
        + not_happy   * 0.15
        + not_fear    * 0.15
        + not_anger   * 0.15
    )
    return max(0.0, raw) * SIGNATURE_ADJUSTMENT_SCALE


def disgust_signature_adjustment(scores):
    """
    ปรับละเอียดรังเกียจ: จมูกย่น + ริมฝีปากบนยก/ล่างตก
    แยกจากโกรธ (คิ้วกดแรงกว่า) และเศร้า (มุมปากตกแต่ไม่ย่นจมูก)
    """
    nose_sneer = _blend_pair_mean(scores, "noseSneerLeft", "noseSneerRight")
    upper_lip = _blend_pair_mean(scores, "mouthUpperUpLeft", "mouthUpperUpRight")
    lower_lip = _blend_pair_mean(scores, "mouthLowerDownLeft", "mouthLowerDownRight")
    eye_squint = _blend_pair_mean(scores, "eyeSquintLeft", "eyeSquintRight")
    brow_down = _blend_pair_mean(scores, "browDownLeft", "browDownRight")
    smile = _blend_pair_mean(scores, "mouthSmileLeft", "mouthSmileRight")

    disgust_signal = nose_sneer * 0.5 + upper_lip * 0.3 + lower_lip * 0.2 + eye_squint * 0.15
    anger_overlap = max(0.0, brow_down - nose_sneer * 0.5)

    raw = max(0.0, disgust_signal - anger_overlap * 0.35 - smile * 0.6)
    return raw * SIGNATURE_ADJUSTMENT_SCALE


def happy_signature_adjustment(scores):
    """
    Duchenne smile (ยิ้มจริง): mouthSmile × cheekSquint × eyeSquint
    เพิ่มคะแนนมีความสุขแบบ non-linear เมื่อเป็นการยิ้มแบบเต็ม
    แยกจาก nervous smile (ยิ้มแต่ไม่มีตาหยี/แก้มยก)
    กรอง fear signals (eyeWide + browOuterUp) ออก
    """
    smile    = _blend_pair_mean(scores, "mouthSmileLeft",    "mouthSmileRight")
    cheek    = _blend_pair_mean(scores, "cheekSquintLeft",   "cheekSquintRight")
    eye_sq   = _blend_pair_mean(scores, "eyeSquintLeft",     "eyeSquintRight")
    eye_wide = _blend_pair_mean(scores, "eyeWideLeft",       "eyeWideRight")
    brow_out = _blend_pair_mean(scores, "browOuterUpLeft",   "browOuterUpRight")
    frown    = _blend_pair_mean(scores, "mouthFrownLeft",    "mouthFrownRight")
    sneer    = _blend_pair_mean(scores, "noseSneerLeft",     "noseSneerRight")

    # Duchenne marker: ทั้งยิ้ม + แก้มยก (non-linear product = แม่นกว่า linear)
    duchenne = smile * cheek
    # ต้องเป็นยิ้มจริงๆ: Duchenne product ต้องมีนัยสำคัญ
    # (กรอง nervous smile ใน fear face ที่ smile ต่ำ × cheek เกือบ 0)
    if duchenne < 0.06:
        return 0.0
    # ยิ้มตาหยี: อีก signal ชัดๆ
    squint_smile = eye_sq * smile
    # กรองความสุขปลอม (frown/sneer ร่วม)
    not_negative = max(0.0, 1.0 - frown * 2.0 - sneer * 2.0)
    # กรอง fear signals: ตาเบิก + คิ้วนอกยก = กลัว/ตกใจ ไม่ใช่ความสุข
    not_fear = max(0.0, 1.0 - eye_wide * 2.0 - brow_out * 1.0)

    raw = (duchenne * 0.6 + squint_smile * 0.4) * not_negative * not_fear
    return max(0.0, raw) * SIGNATURE_ADJUSTMENT_SCALE * 1.2  # happy ชัดกว่าอารมณ์อื่น


def angry_signature_adjustment(scores):
    """
    โกรธแท้: browDown × mouthPucker (คิ้วกด + ห่อปาก)
    แยกจาก หน้าตึง (browDown แต่ไม่ pucker) และเศร้า (browDown แต่ frown ไม่ pucker)
    """
    brow_dn = _blend_pair_mean(scores, "browDownLeft", "browDownRight")
    pucker  = scores.get("mouthPucker",   0.0)
    press   = _blend_pair_mean(scores, "mouthPressLeft", "mouthPressRight")
    sneer   = _blend_pair_mean(scores, "noseSneerLeft",  "noseSneerRight")
    frown   = _blend_pair_mean(scores, "mouthFrownLeft", "mouthFrownRight")
    smile   = _blend_pair_mean(scores, "mouthSmileLeft", "mouthSmileRight")

    eye_sq = _blend_pair_mean(scores, "eyeSquintLeft", "eyeSquintRight")

    # โกรธแบบ core: คิ้วกด + ห่อปาก (ต้องมีทั้งคู่ — strong anger)
    anger_core = brow_dn * pucker
    # โกรธแบบ กัดฟัน
    anger_press = brow_dn * press
    # โกรธ subtle: คิ้วกด + ตาหรี่ (AffectNet subtle anger — ไม่ต้องมี pucker)
    # ต้องไม่มี frown (นั่นคือเศร้า) และไม่มี sneer (นั่นคือรังเกียจ)
    subtle_anger = brow_dn * eye_sq * max(0.0, 1.0 - frown * 3.0 - sneer * 2.0) * 0.6

    # กรองความสุขและเศร้า
    not_happy = max(0.0, 1.0 - smile * 2.0)
    not_sad   = max(0.0, 1.0 - frown * 1.5 + sneer * 0.5)

    raw = (anger_core * 0.5 + anger_press * 0.3 + subtle_anger * 0.2) * not_happy * not_sad
    return max(0.0, raw) * SIGNATURE_ADJUSTMENT_SCALE


def anger_happy_squint_penalty(scores):
    """
    ยิ้มตาหยี (cheekSquint + smile) เป็นลายเซ็นความสุข ไม่ใช่โกรธ
    ลดคะแนนโกรธเพิ่มเมื่อทั้งสองสัญญาณมาพร้อมกัน
    """
    cheek = _blend_pair_mean(scores, "cheekSquintLeft", "cheekSquintRight")
    smile = _blend_pair_mean(scores, "mouthSmileLeft", "mouthSmileRight")
    # ต้องมีทั้งแก้มยกและยิ้ม — กันลดโกรธเมื่อมีแค่ตาหยีอย่างเดียว
    happy_squint = min(cheek, smile) * 0.85 + cheek * 0.35
    return max(0.0, happy_squint) * SIGNATURE_ADJUSTMENT_SCALE


def softmax(values, temperature=3.0):
    scaled = [v * temperature for v in values]
    max_v = max(scaled)
    exps = [math.exp(v - max_v) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def dynamic_threshold(scores):
    values = list(scores.values()) if scores else []
    if not values:
        return 0.25

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(variance)

    # หน้านิ่งสนิท (std ต่ำ) → ดึง threshold สูงขึ้น กัน false positive
    if std < 0.05:
        return 0.30
    return max(0.18, min(0.40, mean + std))


def analyze_expression(scores):
    scores = symmetrize_blendshape_scores(scores)
    scores = relative_blendshape_scores(scores)

    raw_scores = {}
    for emotion, weight_map in EMOTION_RULES.items():
        positive = weighted_score(scores, weight_map)
        negative = penalty_score(scores, EMOTION_PENALTIES.get(emotion, {}))
        raw_scores[emotion] = max(0.0, positive - negative * PENALTY_STRENGTH)

    raw_scores["กลัว"] = max(
        0.0,
        raw_scores.get("กลัว", 0.0) + fear_signature_adjustment(scores),
    )
    raw_scores["เศร้า"] = max(
        0.0,
        raw_scores.get("เศร้า", 0.0) + sadness_signature_adjustment(scores),
    )
    raw_scores["มีความสุข"] = max(
        0.0,
        raw_scores.get("มีความสุข", 0.0) + happy_signature_adjustment(scores),
    )
    raw_scores["โกรธ"] = max(
        0.0,
        raw_scores.get("โกรธ", 0.0)
        + angry_signature_adjustment(scores)
        - anger_happy_squint_penalty(scores),
    )
    raw_scores["รังเกียจ"] = max(
        0.0,
        raw_scores.get("รังเกียจ", 0.0) + disgust_signature_adjustment(scores),
    )

    # หน้าเฉยๆ เป็น class จริง — คะแนนสูงเมื่อไม่มีสัญญาณอารมณ์ชัดเจน
    # baseline เพิ่มจาก 0.22 → 0.25 ให้ margin กับ neutral ชัดขึ้น
    max_emotion_signal = max(raw_scores.values()) if raw_scores else 0.0
    raw_scores["เฉยๆ"] = max(0.0, 0.23 - max_emotion_signal)

    emotions = list(raw_scores.keys())
    probs = softmax(list(raw_scores.values()), temperature=SOFTMAX_TEMPERATURE)
    emotion_probs = dict(zip(emotions, probs))

    best_emotion = max(emotion_probs, key=emotion_probs.get)
    best_raw = raw_scores[best_emotion]
    second_raw = sorted(raw_scores.values(), reverse=True)[1] if len(raw_scores) > 1 else 0.0
    margin = best_raw - second_raw

    threshold = dynamic_threshold(scores)
    if best_raw < threshold or margin < threshold * 0.35:
        summary = (
            f"สีหน้าค่อนข้างเป็นกลาง "
            f"(threshold: {threshold * 100:.1f}%, margin: {margin * 100:.1f}%)"
        )
    else:
        summary = (
            f"สีหน้าใกล้เคียง: {best_emotion} "
            f"(weighted {best_raw * 100:.1f}% / softmax {emotion_probs[best_emotion] * 100:.1f}%)"
        )

    return summary, emotion_probs, raw_scores


def check_quality(result, crop_info, raw_blendshapes):
    """ตรวจคุณภาพภาพก่อน classify — คืน list of issue dicts"""
    issues = []
    if result.face_landmarks:
        lm_count = len(result.face_landmarks[0])
        if lm_count < 400:
            issues.append({"code": "landmark_low",
                            "msg": "พบจุดอ้างอิงใบหน้าน้อย — ปรับมุมกล้องให้เห็นหน้าชัด"})
    if crop_info:
        try:
            w, h = (int(x) for x in crop_info["crop_size"].split("x"))
            if w < 150 or h < 150:
                issues.append({"code": "face_small",
                                "msg": "หน้าเล็กเกินไป — เข้าใกล้กล้องขึ้น"})
        except Exception:
            pass
    if raw_blendshapes:
        max_score = max(raw_blendshapes.values(), default=0)
        if max_score < 0.04:
            issues.append({"code": "low_signal",
                            "msg": "สัญญาณใบหน้าน้อย — เพิ่มแสงสว่าง หรือถ่ายใกล้ขึ้น"})
    return issues


def average_blendshape_scores(scores_list):
    """เฉลี่ยค่า blendshape จากหลายรูป ลด noise"""
    if not scores_list:
        return {}
    all_keys = set().union(*scores_list)
    n = len(scores_list)
    return {k: sum(s.get(k, 0.0) for s in scores_list) / n for k in all_keys}


def build_response(emotion_probs, raw_scores, blendshape_scores,
                   result, crop_info, quality_issues,
                   df_best=None, df_probs=None, shot_count=1):
    """สร้าง structured JSON response
    ถ้ามี trained classifier → ใช้แทน / blend กับ rules
    """
    # ─── Trained classifier (ถ้ามี) ───
    clf_probs = get_classifier_probs(blendshape_scores)
    clf_best  = max(clf_probs, key=clf_probs.get) if clf_probs else None
    rules_best = max(emotion_probs, key=emotion_probs.get)
    clf_max = max(clf_probs.values()) if clf_probs else 0.0

    if clf_probs is not None and clf_best == rules_best:
        # ─── ทั้งสองตกลงกัน → blend เพื่อ boost confidence ─────────────────
        # น้ำหนัก dynamic: clf มั่นใจมาก → ให้ clf มากขึ้น
        clf_w   = CLF_WEIGHT_HIGH if clf_max >= 0.65 else CLF_WEIGHT_LOW
        rules_w = 1.0 - clf_w
        all_ems = set(emotion_probs) | set(clf_probs)
        blended = {
            em: clf_probs.get(em, 0.0) * clf_w + emotion_probs.get(em, 0.0) * rules_w
            for em in all_ems
        }
        total = sum(blended.values()) or 1.0
        emotion_probs_used = {k: v / total for k, v in blended.items()}
        source = f"clf({clf_w:.0%})+rules (agree)"
    elif clf_probs is not None and clf_max >= 0.50:
        # ─── ขัดแย้ง แต่ clf มั่นใจ ≥50% → blend ชันลง clf ───────────────────
        # classifier CV 97.9% → ไว้ใจมากกว่า rules ในกรณีขัดกัน
        clf_w   = CLF_WEIGHT_DISAGREE if clf_max >= 0.35 else CLF_WEIGHT_LOW * 0.5
        rules_w = 1.0 - clf_w
        all_ems = set(emotion_probs) | set(clf_probs)
        blended = {
            em: clf_probs.get(em, 0.0) * clf_w + emotion_probs.get(em, 0.0) * rules_w
            for em in all_ems
        }
        total = sum(blended.values()) or 1.0
        emotion_probs_used = {k: v / total for k, v in blended.items()}
        source = f"clf({clf_w:.0%})+rules (disagree)"
    else:
        # ─── ขัดแย้ง หรือไม่มี clf → ใช้ rules เต็มๆ ────────────────────────
        emotion_probs_used = emotion_probs
        clf_w   = 0.0
        source  = "rules" if clf_probs is None else "rules (clf disagree)"

    bs_best = max(emotion_probs_used, key=emotion_probs_used.get)
    final_emotion, ensemble_note = ensemble_emotion(bs_best, emotion_probs_used, df_best, df_probs)

    ranked = sorted(emotion_probs_used.items(), key=lambda x: x[1], reverse=True)
    confidence = ranked[0][1]
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 0.0

    if margin >= CONFIDENCE_HIGH:
        certain_level = "high"
    elif margin >= CONFIDENCE_MED:
        certain_level = "medium"
    else:
        certain_level = "low"

    lm_count = len(result.face_landmarks[0]) if result.face_landmarks else 0

    top_bs = sorted(blendshape_scores.items(), key=lambda x: x[1], reverse=True)[:8]

    return {
        "status": "success",
        "emotion": final_emotion,
        "emotion_emoji": EMOTION_EMOJIS.get(final_emotion, ""),
        "emotion_en": EMOTION_EN.get(final_emotion, ""),
        "confidence": round(confidence, 4),
        "margin": round(margin, 4),
        "certain": margin >= CONFIDENCE_MED,
        "certain_level": certain_level,
        "emotions": {k: round(v, 4) for k, v in ranked},
        "quality": {
            "ok": len(quality_issues) == 0,
            "issues": quality_issues,
            "landmark_count": lm_count,
            "crop_size": crop_info["crop_size"] if crop_info else None,
            "shot_count": shot_count,
        },
        "cnn": {
            "available": _DEEPFACE_AVAILABLE,
            "emotion": df_best,
            "emotion_emoji": EMOTION_EMOJIS.get(df_best, "") if df_best else "",
            "agree": (df_best == bs_best) if df_best else None,
            "note": ensemble_note,
        },
        "classifier": {
            "available": clf_probs is not None,
            "emotion": clf_best,
            "emotion_emoji": EMOTION_EMOJIS.get(clf_best, "") if clf_best else "",
            "confidence": round(clf_max, 3),
            "weight": round(clf_w, 2),
            "source": source,
        },
        "blendshapes_top": [
            {"name": BLENDSHAPE_LABELS.get(k, k), "score": round(v, 3)}
            for k, v in top_bs
        ],
        # blendshape ครบ 52 ตัว (English keys) — ส่งกลับเพื่อใช้ใน feedback/training
        "blendshapes_raw": {k: round(v, 4) for k, v in blendshape_scores.items()},
    }


def save_feedback(predicted, actual, blendshapes_raw, margin):
    """บันทึก user feedback → JSONL สำหรับ train classifier ในอนาคต
    blendshapes_raw: dict {en_name: score} ครบ 52 ตัวจาก MediaPipe
    """
    entry = {
        "ts":         time.time(),
        "predicted":  predicted,
        "actual":     actual,
        "margin":     round(margin, 4),
        "blendshapes": {k: round(float(v), 5) for k, v in blendshapes_raw.items()},
    }
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def format_detection_result(result, crop_info=None, df_best=None, df_probs=None):
    """วิเคราะห์ result จาก MediaPipe → คืน structured dict"""
    if not result.face_landmarks:
        return {"status": "error", "code": "no_face",
                "message": "ไม่พบใบหน้าในภาพ กรุณาใช้ภาพที่เห็นใบหน้าชัดเจน"}

    if not result.face_blendshapes:
        return {"status": "error", "code": "no_blendshapes",
                "message": "พบใบหน้า แต่โมเดลไม่ได้ส่งค่า face blendshapes กลับมา"}

    scores = categories_to_scores(result.face_blendshapes[0])
    _, emotion_probs, raw_scores = analyze_expression(scores)
    quality_issues = check_quality(result, crop_info, scores)

    return build_response(emotion_probs, raw_scores, scores,
                          result, crop_info, quality_issues,
                          df_best=df_best, df_probs=df_probs)


def predict_with_deepface(image_bytes, filename=""):
    """
    รัน DeepFace CNN บนภาพ → คืน (best_thai, probs_thai) หรือ (None, None)
    probs_thai: dict {thai_emotion: probability 0-1}
    """
    if not _DEEPFACE_AVAILABLE:
        return None, None
    suffix = detect_image_suffix(image_bytes, filename)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(image_bytes)
            tmp = f.name
        result = _DeepFace.analyze(
            img_path=tmp,
            actions=["emotion"],
            enforce_detection=False,
            silent=True,
        )
        en_em = result[0]["emotion"]           # {str: float %} สรุปเป็น 100
        total = sum(en_em.values()) or 1.0
        th_probs = {
            _DEEPFACE_EN_TO_TH[k]: v / total
            for k, v in en_em.items()
            if k in _DEEPFACE_EN_TO_TH
        }
        best = max(th_probs, key=th_probs.get)
        return best, th_probs
    except Exception:
        return None, None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def ensemble_emotion(bs_best, bs_probs, df_best, df_probs):
    """
    รวม blendshape rules + DeepFace CNN

    Strategy:
      1. ถ้าทั้งสองระบบเห็นตรงกัน → ใช้ผลนั้น (confidence สูงขึ้น)
      2. ถ้าขัดแย้ง → ใช้ blendshape เป็นหลัก
         ยกเว้น: DeepFace มั่นใจ "มีความสุข" >90% หรือ "ประหลาดใจ" >85%
                 (2 อารมณ์นี้ชัดเจนข้ามวัฒนธรรม ไม่มี false-positive สูง)
    คืน (final_best, note)
    """
    if df_best is None:
        return bs_best, "blendshape_only"

    if bs_best == df_best:
        return bs_best, "agree"

    # ตรวจ deep-confidence overrides
    if df_best == "มีความสุข" and df_probs.get("มีความสุข", 0) > 0.90:
        return "มีความสุข", "deepface_happy_override"
    if df_best == "ประหลาดใจ" and df_probs.get("ประหลาดใจ", 0) > 0.85:
        return "ประหลาดใจ", "deepface_surprise_override"

    return bs_best, "blendshape_wins"


def _read_image_bytes(file_stream):
    data = file_stream.read(MAX_IMAGE_SIZE + 1)
    if len(data) > MAX_IMAGE_SIZE:
        raise ValueError(f"ไฟล์ภาพใหญ่เกิน {MAX_IMAGE_SIZE // (1024 * 1024)} MB")
    return data


def predict_single(image_bytes, filename=""):
    """วิเคราะห์รูปเดียว → structured dict"""
    result, crop_info = run_face_landmarker(image_bytes, filename)
    df_best, df_probs = predict_with_deepface(image_bytes, filename)
    return format_detection_result(result, crop_info,
                                   df_best=df_best, df_probs=df_probs)


def predict_multishot(shots):
    """
    วิเคราะห์หลายรูป (shots: list of (image_bytes, filename))
    → เฉลี่ย blendshapes ก่อน classify ลด noise
    """
    valid_results = []
    valid_scores  = []
    best_result   = None
    best_crop     = None

    for image_bytes, filename in shots:
        result, crop_info = run_face_landmarker(image_bytes, filename)
        if not result.face_landmarks or not result.face_blendshapes:
            continue
        scores = categories_to_scores(result.face_blendshapes[0])
        valid_results.append(result)
        valid_scores.append(scores)
        if best_result is None:
            best_result = result
            best_crop   = crop_info

    if not valid_scores:
        return {"status": "error", "code": "no_face",
                "message": "ไม่พบใบหน้าในภาพใดๆ กรุณาถ่ายใหม่"}

    avg_scores = average_blendshape_scores(valid_scores)
    _, emotion_probs, raw_scores = analyze_expression(avg_scores)
    quality_issues = check_quality(best_result, best_crop, avg_scores)

    # DeepFace บนรูปแรกที่ใช้งานได้
    df_best, df_probs = predict_with_deepface(shots[0][0], shots[0][1])

    resp = build_response(emotion_probs, raw_scores, avg_scores,
                          best_result, best_crop, quality_issues,
                          df_best=df_best, df_probs=df_probs,
                          shot_count=len(valid_scores))
    return resp


def predict_with_face_landmarker(file_stream, filename=""):
    """Backward-compat wrapper"""
    image_bytes = _read_image_bytes(file_stream)
    return predict_single(image_bytes, filename)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@server.route("/")
def main():
    return render_template("main.html")


@server.route("/api/data", methods=["POST"])
def upload_image():
    """Single-shot endpoint (backward compat + เพิ่ม structured response)"""
    if "photo" not in request.files:
        return jsonify({"status": "error", "message": "ไม่พบไฟล์ภาพ"}), 400
    file = request.files["photo"]
    if not file.filename:
        return jsonify({"status": "error", "message": "ชื่อไฟล์ว่างเปล่า"}), 400
    try:
        image_bytes = _read_image_bytes(file.stream)
        result = predict_single(image_bytes, file.filename)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาด: {exc}"}), 500


@server.route("/api/multishot", methods=["POST"])
def upload_multishot():
    """Multi-shot endpoint: photo1, photo2, photo3 → เฉลี่ย blendshapes"""
    shots = []
    for i in range(1, MAX_SHOTS + 1):
        key = f"photo{i}"
        f = request.files.get(key)
        if f and f.filename:
            try:
                shots.append((_read_image_bytes(f.stream), f.filename))
            except ValueError as exc:
                return jsonify({"status": "error", "message": str(exc)}), 400

    if not shots:
        return jsonify({"status": "error", "message": "ไม่พบไฟล์ภาพ"}), 400

    try:
        result = predict_multishot(shots)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"status": "error", "message": f"เกิดข้อผิดพลาด: {exc}"}), 500


@server.route("/api/feedback", methods=["POST"])
def receive_feedback():
    """รับ user feedback → บันทึกสำหรับ training
    ต้องส่ง: predicted, actual, blendshapes_raw (dict ครบ 52 ตัว), margin
    """
    data = request.get_json(silent=True) or {}
    predicted      = data.get("predicted", "")
    actual         = data.get("actual", "")
    blendshapes_raw = data.get("blendshapes_raw", {})
    margin         = float(data.get("margin", 0))

    if not predicted or not actual:
        return jsonify({"status": "error", "message": "ข้อมูลไม่ครบ"}), 400
    if not blendshapes_raw:
        return jsonify({"status": "error", "message": "ไม่มีข้อมูล blendshapes"}), 400

    save_feedback(predicted, actual, blendshapes_raw, margin)

    # นับ records ทั้งหมด เพื่อแจ้ง user ว่าพร้อม train หรือยัง
    try:
        total = sum(1 for _ in open(FEEDBACK_FILE, encoding="utf-8"))
    except Exception:
        total = 1

    MIN_TRAIN = 50
    msg = f"บันทึก feedback แล้ว ({total} records)"
    if total >= MIN_TRAIN:
        msg += f" — มีข้อมูลเพียงพอสำหรับ train classifier แล้ว! รัน python train_classifier.py"
    else:
        msg += f" — ต้องการอีก {MIN_TRAIN - total} records จึงจะ train ได้"

    return jsonify({"status": "success", "message": msg, "total_feedback": total})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"
    server.run(port=5555, debug=debug_mode)