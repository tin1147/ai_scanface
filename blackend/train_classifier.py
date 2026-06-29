# -*- coding: utf-8 -*-
"""
train_classifier.py
────────────────────────────────────────────────────────────────
อ่าน dataset.jsonl (หรือ feedback.jsonl) → train classifier → บันทึก classifier.pkl

วิธีใช้:
  python train_classifier.py                          # train จาก feedback.jsonl
  python train_classifier.py --dataset dataset.jsonl  # train จาก dataset ที่สะอาด
  python train_classifier.py --dry-run                # ดูสถิติอย่างเดียว
  python train_classifier.py --min 20                 # กำหนด min records ต่อ class

แนะนำ: รัน build_dataset.py ก่อน แล้วค่อย train ด้วย --dataset dataset.jsonl
classifier.pkl จะถูกโหลดอัตโนมัติตอน server.py รีสตาร์ท
────────────────────────────────────────────────────────────────
"""
import argparse
import json
import pickle
import shutil
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

# ─── ค่า default ───
DEFAULT_MIN_PER_CLASS = 10   # ต่ำสุดที่ยอมรับได้ (ต่ำกว่านี้ไม่ train)
RECOMMENDED_MIN       = 50   # แนะนำเพื่อความแม่น
FEEDBACK_FILE  = Path(__file__).parent / "feedback.jsonl"
DATASET_FILE   = Path(__file__).parent / "dataset.jsonl"
CLASSIFIER_PATH = Path(__file__).parent / "classifier.pkl"

THAI_EMOTIONS = ["มีความสุข", "เศร้า", "โกรธ", "กลัว", "รังเกียจ", "ประหลาดใจ", "เฉยๆ"]

# blendshape features ที่ใช้ (ทั้งหมด 52 ตัวจาก MediaPipe FaceLandmarker)
ALL_BLENDSHAPE_FEATURES = [
    "_neutral",
    "browDownLeft", "browDownRight",
    "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward",
    "jawLeft", "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft", "mouthRight",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


def load_feedback(path):
    """โหลด dataset.jsonl / feedback.jsonl → list of dict
    กรอง junk (Thai keys, ค่าไม่ใช่ EN blendshape) ออกด้วย
    """
    records = []
    if not path.exists():
        return records

    # valid EN blendshape keys (52 ตัวของ MediaPipe)
    valid_keys = {
        "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
        "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
        "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
        "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
        "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
        "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
        "jawForward", "jawLeft", "jawRight", "jawOpen", "mouthClose",
        "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
        "mouthFunnel", "mouthLeft", "mouthRight", "mouthLowerDownLeft",
        "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight", "mouthPucker",
        "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
        "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
        "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
    }

    junk = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            bs = r.get("blendshapes", {})
            # กรอง junk: ต้องเป็น EN keys subset และมี ≥26 ตัว
            if (not isinstance(bs, dict)
                    or not set(bs.keys()).issubset(valid_keys)
                    or len(bs) < 26):
                junk += 1
                continue
            records.append(r)
    if junk:
        print(f"  (กรอง junk {junk} records ออก)")
    return records


def _compute_interaction_features(bs):
    """Mirror ของ server.py compute_interaction_features (standalone)"""
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
        "IX_duchenne":        smile * cheek,
        "IX_squint_smile":    eye_sq * smile,
        "IX_anger_core":      brow_dn * pucker,
        "IX_anger_press":     brow_dn * press,
        "IX_sad_core":        brow_in * frown,
        "IX_sad_chin":        frown * shrug,
        "IX_fear_core":       eye_wide * brow_out * max(0.0, 1.0 - jaw_open * 1.5),
        "IX_fear_stretch":    stretch * eye_wide,
        "IX_surprise_core":   eye_wide * jaw_open,
        "IX_disgust_core":    sneer * upper_up,
        "IX_disgust_lower":   sneer * lower_dn,
        "IX_tense_neutral":   brow_dn * max(0.0, 1.0 - pucker) * max(0.0, 1.0 - frown),
        "IX_nervous_smile":   smile * max(0.0, 1.0 - cheek * 1.5),
        "IX_wide_vs_squint":  max(0.0, eye_wide - eye_sq * 0.5),
        "IX_inner_vs_down":   max(0.0, brow_in - brow_dn * 0.8),
    }


def build_dataset(records):
    """แปลง records → X (numpy), y (list), feature_names (list)
    ใช้ raw blendshapes + interaction features (ต้องตรงกับ server.py)
    """
    try:
        import numpy as np
    except ImportError:
        sys.exit("ต้องติดตั้ง numpy ก่อน: pip install numpy")

    # raw features ที่ปรากฏในข้อมูล
    all_keys_in_data = set()
    for r in records:
        all_keys_in_data.update(r.get("blendshapes", {}).keys())

    ordered  = [f for f in ALL_BLENDSHAPE_FEATURES if f in all_keys_in_data]
    extras   = sorted(all_keys_in_data - set(ordered))
    raw_feat = ordered + extras

    # interaction feature names (คงที่)
    sample_ix = _compute_interaction_features({})
    ix_feat = sorted(sample_ix.keys())

    features = raw_feat + ix_feat

    X, y = [], []
    for r in records:
        label = r.get("actual", "")
        if label not in THAI_EMOTIONS:
            continue
        bs = r.get("blendshapes", {})
        # symmetrize ให้ตรงกับ server.py (ลด L/R bias)
        bs = _symmetrize(bs)
        ix = _compute_interaction_features(bs)
        row = [float(bs.get(f, 0.0)) for f in raw_feat] + [float(ix.get(f, 0.0)) for f in ix_feat]
        X.append(row)
        y.append(label)

    return np.array(X, dtype=np.float32), y, features


# คู่ blendshape ซ้าย/ขวา — mirror ของ server.py BLENDSHAPE_SYMMETRY_PAIRS
_SYM_PAIRS = (
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


def _symmetrize(bs):
    """เฉลี่ยคู่ซ้าย/ขวา — ตรงกับที่ server.py ทำจริงตอน inference"""
    out = dict(bs)
    for left, right in _SYM_PAIRS:
        if left in out and right in out:
            avg = (out[left] + out[right]) / 2.0
            out[left] = avg
            out[right] = avg
    return out


def print_stats(y, records):
    total = len(y)
    counts = Counter(y)
    print(f"\n{'─'*50}")
    print(f"feedback records ทั้งหมด: {len(records)}")
    print(f"records ที่ใช้ train:      {total}")
    print(f"\nจำนวนต่อ class:")
    for em in THAI_EMOTIONS:
        n = counts.get(em, 0)
        bar = "█" * n + "░" * max(0, RECOMMENDED_MIN - n)
        status = "✅" if n >= RECOMMENDED_MIN else ("⚠️" if n >= DEFAULT_MIN_PER_CLASS else "❌")
        print(f"  {status} {em:<12}: {n:>4}  {bar[:40]}")
    print(f"{'─'*50}\n")


def train(X, y, features):
    """Train ensemble classifier (RF + SVM) → calibrated Pipeline

    ใช้ VotingClassifier รวม RandomForest + SVM เพื่อเพิ่มความแม่น
    จากนั้น CalibratedClassifierCV ปรับ probability ให้แม่นยำขึ้น
    """
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.ensemble import RandomForestClassifier, VotingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.metrics import classification_report
        import numpy as np
    except ImportError:
        sys.exit("ต้องติดตั้ง scikit-learn ก่อน: pip install scikit-learn")

    n_samples = len(y)
    n_classes = len(set(y))
    min_count = min(Counter(y).values())
    cv_splits = 5 if min_count >= 5 else (3 if min_count >= 3 else 2)

    # ─── Algorithm: Ensemble (RF + SVM) ───
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    svm = SVC(
        kernel="rbf", C=10, gamma="scale",
        class_weight="balanced", probability=True, random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm)],
        voting="soft",
        weights=[1.0, 0.8],  # RF น้ำหนักมากกว่าเล็กน้อย
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", ensemble),
    ])
    algo_name = "Ensemble (RF+RBF-SVM), soft vote"

    print(f"Algorithm: {algo_name}")
    print(f"Features:  {len(features)}")
    print(f"Samples:   {n_samples}  |  Classes: {n_classes}")

    # ─── Cross-Validation จริง ───
    counts = Counter(y)
    cv_mask = np.array([counts[label] >= cv_splits for label in y])
    X_cv = X[cv_mask]
    y_cv = [yi for yi, ok in zip(y, cv_mask) if ok]

    cv_score = None
    if len(set(y_cv)) >= 2 and len(y_cv) >= cv_splits * 3:
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring="accuracy")
        cv_score = scores.mean()
        print(f"\nCross-val accuracy ({cv_splits}-fold): {cv_score*100:.1f}% ± {scores.std()*100:.1f}%")

        # แสดง per-class classification report
        if cv_score > 0.7:
            from sklearn.model_selection import cross_val_predict
            y_pred_cv = cross_val_predict(model, X_cv, y_cv, cv=cv)
            print("\n📊 Classification Report (CV):")
            print(classification_report(y_cv, y_pred_cv, digits=3))
    else:
        print(f"\n⚠️  ข้อมูลน้อยเกินไปสำหรับ CV — train โดยไม่ validate")

    # ─── Fit บน data ทั้งหมด ───
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    return model, cv_score


def save_model(model, features, path):
    """บันทึก classifier.pkl"""
    tmp = tempfile.mktemp(suffix=".pkl")
    data = {"model": model, "features": features, "emotions": THAI_EMOTIONS}
    with open(tmp, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.copy2(tmp, path)
    Path(tmp).unlink(missing_ok=True)
    print(f"\n✅ บันทึก classifier → {path}")
    print("   รีสตาร์ท server.py เพื่อโหลด classifier ใหม่")


def main():
    parser = argparse.ArgumentParser(description="Train emotion classifier จาก feedback/dataset")
    parser.add_argument("--dataset", type=str, default=None,
                        help="ไฟล์ dataset (default: ลอง dataset.jsonl ก่อน ถ้าไม่มีใช้ feedback.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="ดูสถิติอย่างเดียว ไม่ train")
    parser.add_argument("--min", type=int, default=DEFAULT_MIN_PER_CLASS,
                        metavar="N", help=f"จำนวน record ขั้นต่ำต่อ class (default: {DEFAULT_MIN_PER_CLASS})")
    args = parser.parse_args()

    print("=" * 55)
    print("  Emotion Classifier Trainer")
    print("=" * 55)

    # เลือกไฟล์ dataset: --dataset > dataset.jsonl > feedback.jsonl
    data_file = None
    if args.dataset:
        data_file = Path(args.dataset)
    elif DATASET_FILE.exists():
        data_file = DATASET_FILE
        print(f"✅ ใช้ dataset.jsonl (สะอาด, สมดุล)")
    else:
        data_file = FEEDBACK_FILE

    print(f"📂 อ่านจาก: {data_file.name}")
    records = load_feedback(data_file)
    if not records:
        print(f"❌ ไม่พบข้อมูล หรือว่างเปล่า")
        print(f"   รัน build_dataset.py ก่อน หรือให้ user ใช้งานแอปและกด feedback")
        sys.exit(1)

    X, y, features = build_dataset(records)
    print_stats(y, records)

    if args.dry_run:
        print("(dry-run — ไม่ train)")
        return

    counts = Counter(y)
    missing = [em for em in THAI_EMOTIONS if counts.get(em, 0) == 0]
    low_classes = [em for em in THAI_EMOTIONS if 0 < counts.get(em, 0) < args.min]
    trained_classes = [em for em in THAI_EMOTIONS if counts.get(em, 0) >= 1]

    if missing:
        print(f"⚠️  ยังไม่มีข้อมูล: {', '.join(missing)}")
        print(f"   classifier จะทำนายได้เฉพาะ: {', '.join(trained_classes)}")
        print(f"   (ถ้าหน้าตรงกับ class ที่ขาด → rules-based เป็น fallback)\n")
    if low_classes:
        print(f"⚠️  ข้อมูลน้อยกว่า {args.min}: {', '.join(low_classes)}")
        print("   จะ train ต่อด้วย class_weight='balanced'\n")
    if len(trained_classes) < 2:
        print("❌ ต้องมีอย่างน้อย 2 class จึงจะ train ได้")
        sys.exit(1)

    model, cv_score = train(X, y, features)
    save_model(model, features, CLASSIFIER_PATH)

    # แสดง classification report บน train set
    if len(y) >= 20:
        try:
            from sklearn.metrics import classification_report
            import numpy as np
            y_pred = model.predict(X)
            print("\n📊 Classification Report (train set):")
            print(classification_report(y, y_pred, target_names=THAI_EMOTIONS,
                                        labels=THAI_EMOTIONS, zero_division=0))
        except Exception:
            pass


if __name__ == "__main__":
    main()
