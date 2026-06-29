"""
ทดสอบทุก test case ใน phototest/ และแสดงผลว่าตรงหรือไม่
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

EXPECTED = {
    "กลัว/กลัว1.png":           "กลัว",
    "หน้าตึง/ตึง1.png":          "เฉยๆ",   # หน้าตึง = neutral/เฉยๆ หรือ โกรธเล็กน้อย
    "หน้ายิ้ม/ยิ้ม1.png":        "มีความสุข",
    "หน้ารังเกียจ/รังเกียจ1.png": "รังเกียจ",
    "หน้าเฉยๆ/เฉยๆ1.png":       "เฉยๆ",
    "หน้าเศร้า/เศร้า1.png":      "เศร้า",
    "หน้าเศร้า/เศร้า2.png":      "เศร้า",
    "หน้าโกรธ/โกรธ1.png":       "โกรธ",
}

PHOTOTEST_DIR = Path(__file__).parent / "phototest"

from server import run_face_landmarker, format_detection_result, analyze_expression, categories_to_scores

def run_test(rel_path, expected):
    full_path = PHOTOTEST_DIR / rel_path
    if not full_path.exists():
        return None, f"ไม่พบไฟล์: {full_path}"

    image_bytes = full_path.read_bytes()
    result, crop_info = run_face_landmarker(image_bytes, full_path.name)

    if not result.face_landmarks:
        return None, "ไม่พบใบหน้า"
    if not result.face_blendshapes:
        return None, "ไม่มี blendshapes"

    scores = categories_to_scores(result.face_blendshapes[0])
    _, emotion_probs, raw_scores = analyze_expression(scores)

    best = max(emotion_probs, key=emotion_probs.get)
    ranked = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    return best, ranked


def main():
    print("=" * 65)
    print(f"{'ไฟล์':<30} {'คาด':<10} {'ได้':<10} {'ผล'}")
    print("=" * 65)

    passed = 0
    total = 0
    failures = []

    for rel_path, expected in EXPECTED.items():
        best, ranked = run_test(rel_path, expected)
        total += 1
        name = rel_path.split("/")[-1]

        if best is None:
            status = "ERROR"
            print(f"{name:<30} {expected:<10} {'—':<10} {ranked}")
            failures.append((name, expected, "error", ranked))
            continue

        ok = best == expected or (expected == "เฉยๆ" and best in ("เฉยๆ", "หน้าตึง"))
        if ok:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"
            failures.append((name, expected, best, ranked))

        top3 = "  |  ".join(f"{e}: {p*100:.1f}%" for e, p in ranked[:3])
        print(f"{name:<30} {expected:<10} {best:<10} [{status}]")
        print(f"  top3: {top3}")

    print("=" * 65)
    print(f"ผล: {passed}/{total} ผ่าน")

    if failures:
        print("\n--- รายละเอียดที่ FAIL ---")
        for name, expected, got, ranked in failures:
            print(f"\n{name}: คาด '{expected}' แต่ได้ '{got}'")
            for e, p in ranked:
                bar = "█" * int(p * 40)
                print(f"  {e:<10} {p*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
