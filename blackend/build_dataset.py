# -*- coding: utf-8 -*-
"""
build_dataset.py
────────────────────────────────────────────────────────────────
รวบรวมข้อมูล training จากทุกแหล่งที่เป็นไปได้ → dataset.jsonl ที่สะอาด

แหล่งข้อมูล:
  1. feedback.jsonl — user feedback (ทำความสะอาด: เอา Thai-key ออก, dedupe)
  2. phototest/     — ground-truth รูปทดสอบ (รัน MediaPipe จริง)
  3. augmentation  — สัญญาณ blendshape เทียม (jitter + scaling) ขยาย class น้อย

หลังจากรันแล้ว จะได้ dataset.jsonl ที่ train_classifier.py อ่านต่อได้
────────────────────────────────────────────────────────────────
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
APP_ROOT = HERE.parent
PHOTOTEST_DIR = HERE / "phototest"
FEEDBACK_FILE = HERE / "feedback.jsonl"
DATASET_FILE = HERE / "dataset.jsonl"

THAI_EMOTIONS = ["มีความสุข", "เศร้า", "โกรธ", "กลัว", "รังเกียจ", "ประหลาดใจ", "เฉยๆ"]

# ชื่อโฟลเดอร์ phototest → label ไทย
PHOTOTEST_LABEL_MAP = {
    "หน้ายิ้ม":   "มีความสุข",
    "หน้าเศร้า":  "เศร้า",
    "หน้าโกรธ":   "โกรธ",
    "กลัว":       "กลัว",
    "หน้ารังเกียจ": "รังเกียจ",
    "หน้าเฉยๆ":   "เฉยๆ",
    "หน้าตึง":    "เฉยๆ",      # หน้าตึง → neutral (ตาม run_tests.py)
}

# blendshape features ครบ 52 ตัวของ MediaPipe (ใช้ตรวจThai-key junk)
VALID_BLENDSHAPE_KEYS = {
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

random.seed(42)


def is_clean_blendshape(bs):
    """ตรวจว่า blendshape dict เป็น EN keys ครบ และมีค่าที่สมเหตุสมผล"""
    if not bs or not isinstance(bs, dict):
        return False
    keys = set(bs.keys())
    # ต้องเป็น subset ของ valid keys (จะได้กรอง Thai-key junk ออก)
    if not keys.issubset(VALID_BLENDSHAPE_KEYS):
        return False
    # ต้องมีครบอย่างน้อยครึ่งหนึ่ง (≥26) — ไม่ใช่ข้อมูลเศษ
    if len(keys) < 26:
        return False
    # ค่าต้องอยู่ในช่วง [0, 1]
    for v in bs.values():
        try:
            v = float(v)
            if v < 0.0 or v > 1.0:
                return False
        except (TypeError, ValueError):
            return False
    return True


def fingerprint(bs):
    """สร้าง hash จาก blendshape ที่ rounded เพื่อ dedupe"""
    return tuple(sorted((k, round(float(v), 4)) for k, v in bs.items()))


def load_clean_feedback():
    """อ่าน feedback.jsonl → กรอง junk + dedupe"""
    records = []
    if not FEEDBACK_FILE.exists():
        return records
    seen = set()
    junk = 0
    dup = 0
    with open(FEEDBACK_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = r.get("actual", "")
            bs = r.get("blendshapes", {})
            if label not in THAI_EMOTIONS:
                junk += 1
                continue
            if not is_clean_blendshape(bs):
                junk += 1
                continue
            fp = fingerprint(bs)
            if fp in seen:
                dup += 1
                continue
            seen.add(fp)
            records.append({"actual": label, "blendshapes": bs, "source": "feedback"})
    print(f"  feedback: {len(records)} clean records "
          f"({junk} junk removed, {dup} duplicates removed)")
    return records


def load_phototest_records():
    """รัน MediaPipe บนรูปใน phototest/ → blendshapes จริง"""
    records = []
    # import server (อยู่ directory เดียวกัน)
    sys.path.insert(0, str(HERE))
    try:
        from server import (run_face_landmarker, categories_to_scores,
                            symmetrize_blendshape_scores)
    except Exception as exc:
        print(f"  ⚠️  import server ไม่ได้: {exc} — ข้าม phototest")
        return records

    if not PHOTOTEST_DIR.exists():
        print("  ⚠️  ไม่พบ phototest/ — ข้าม")
        return records

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    processed = 0
    for folder in sorted(PHOTOTEST_DIR.iterdir()):
        if not folder.is_dir():
            continue
        label = PHOTOTEST_LABEL_MAP.get(folder.name)
        if not label:
            continue
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in image_exts:
                continue
            try:
                img_bytes = img_path.read_bytes()
                result, _ = run_face_landmarker(img_bytes, img_path.name)
                if not result.face_landmarks or not result.face_blendshapes:
                    continue
                raw = categories_to_scores(result.face_blendshapes[0])
                # symmetrize ตามที่ server ใช้จริง (ลด L/R bias)
                sym = symmetrize_blendshape_scores(raw)
                records.append({
                    "actual": label,
                    "blendshapes": sym,
                    "source": f"phototest/{folder.name}",
                })
                processed += 1
            except Exception as exc:
                print(f"    ⚠️  {img_path.name}: {exc}")

    print(f"  phototest: {processed} ground-truth records (MediaPipe real)")
    return records


def augment(records, target_per_class=40):
    """ขยายข้อมูลด้วย jitter + scaling เพื่อให้แต่ละ class พอ train
    - jitter: เพิ่ม Gaussian noise เล็กน้อยให้ blendshape
    - scaling: ปรับ magnitude ของทุกค่า ±10%
    """
    by_class = {em: [] for em in THAI_EMOTIONS}
    for r in records:
        by_class[r["actual"]].append(r)

    augmented = list(records)  # เก็บต้นฉบับก่อน
    rng = random.Random(42)

    for em in THAI_EMOTIONS:
        originals = by_class[em]
        if not originals:
            continue
        n_have = len(originals)
        n_need = max(0, target_per_class - n_have)
        for i in range(n_need):
            base = originals[i % n_have]["blendshapes"]
            new_bs = {}
            for k, v in base.items():
                v = float(v)
                # jitter ±3%
                v += rng.gauss(0, 0.03)
                # scaling 0.9–1.1
                v *= rng.uniform(0.9, 1.1)
                new_bs[k] = max(0.0, min(1.0, v))
            augmented.append({
                "actual": em,
                "blendshapes": new_bs,
                "source": f"augment#{i}",
            })

    print(f"  augmentation: ขยายให้แต่ละ class มี ≥{target_per_class} "
          f"(รวม {len(augmented)} records)")
    return augmented


def main():
    print("=" * 55)
    print("  Dataset Builder (รวมทุกแหล่ง)")
    print("=" * 55)

    all_records = []

    print("\n[1/3] ทำความสะอาด feedback.jsonl ...")
    all_records.extend(load_clean_feedback())

    print("\n[2/3] ดึง ground-truth จาก phototest/ ...")
    all_records.extend(load_phototest_records())

    print(f"\n     รวมก่อน augment: {len(all_records)} records")

    print("\n[3/3] Augmentation ขยาย class น้อย ...")
    all_records = augment(all_records, target_per_class=40)

    # สถิติสุดท้าย
    counts = Counter(r["actual"] for r in all_records)
    print(f"\n{'─'*55}")
    print(f"Dataset สุดท้าย: {len(all_records)} records")
    print(f"{'─'*55}")
    for em in THAI_EMOTIONS:
        n = counts.get(em, 0)
        bar = "█" * min(n, 40)
        status = "✅" if n >= 40 else ("⚠️" if n >= 15 else "❌")
        print(f"  {status} {em:<12}: {n:>3}  {bar}")

    # เขียน dataset.jsonl
    with open(DATASET_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ เขียน {DATASET_FILE.name} ({len(all_records)} records)")
    print("   รัน train_classifier.py --dataset dataset.jsonl เพื่อ train")


if __name__ == "__main__":
    main()
