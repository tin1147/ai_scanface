# AI ScanFace — ระบบวิเคราะห์อารมณ์จากใบหน้า

ระบบวิเคราะห์อารมณ์แบบ Real-time ผ่านกล้องหรือรูปภาพ โดยใช้ MediaPipe Face Landmarker + Blendshape Classifier พร้อม DeepFace CNN เป็น optional ensemble

## ฟีเจอร์หลัก

- วิเคราะห์อารมณ์ 7 ประเภท: **มีความสุข / เศร้า / โกรธ / กลัว / รังเกียจ / ประหลาดใจ / เฉยๆ**
- รองรับ **Multi-shot** (ส่งได้สูงสุด 3 รูปต่อครั้ง เพื่อเพิ่มความแม่นยำ)
- **Interaction features** — คำนวณ signal ผสมระหว่าง blendshapes เพื่อแยกแยะอารมณ์ได้ละเอียดขึ้น
- **User feedback loop** — ผู้ใช้แก้ผลที่ผิดได้ ระบบนำไปเทรน classifier ใหม่อัตโนมัติ
- **DeepFace ensemble** (optional) — ถ้าติดตั้ง DeepFace จะ blend ผลร่วมกับ blendshape

## โครงสร้างโปรเจกต์

```
aiซึมเศร้า/
├── blackend/
│   ├── server.py              # Flask API server หลัก
│   ├── train_classifier.py    # เทรน classifier จาก dataset.jsonl
│   ├── build_dataset.py       # รวม feedback + phototest → dataset.jsonl
│   ├── run_tests.py           # ทดสอบรูปใน phototest/
│   ├── dataset.jsonl          # Training data
│   ├── feedback.jsonl         # User feedback สะสม
│   ├── classifier.pkl         # Trained model (สร้างหลัง train)
│   └── phototest/             # รูปทดสอบแยกตามอารมณ์
│       ├── หน้าเศร้า/
│       ├── หน้ายิ้ม/
│       ├── หน้าโกรธ/
│       ├── หน้ารังเกียจ/
│       ├── หน้ากลัว/
│       └── หน้าเฉยๆ/
├── frontend/
│   ├── templates/
│   │   └── main.html          # หน้าหลัก (กล้อง + ผลวิเคราะห์)
│   ├── static/
│   │   └── stye.css           # Stylesheet
│   ├── button-page/
│   │   ├── donate.html        # หน้าบริจาค
│   │   └── history.html       # หน้าประวัติ
│   └── photo/logo/
│       └── logo.png
└── model/
    ├── face_landmarker.task   # MediaPipe Face Landmarker model
    └── AU_200.tflite          # Action Unit model
```

## การติดตั้ง

### 1. ติดตั้ง Dependencies

```bash
pip install flask mediapipe opencv-python scikit-learn numpy pillow
```

DeepFace (optional — ถ้าไม่ติดตั้ง ระบบยังทำงานได้ปกติ):
```bash
pip install deepface tf-keras
```

### 2. รัน Server

```bash
cd blackend
python server.py
```

เปิดเบราว์เซอร์ไปที่ `http://localhost:5000`

## API Endpoints

| Method | Path | คำอธิบาย |
|--------|------|-----------|
| `GET` | `/` | หน้าหลัก |
| `POST` | `/api/data` | วิเคราะห์รูปภาพเดี่ยว |
| `POST` | `/api/multishot` | วิเคราะห์หลายรูปพร้อมกัน (สูงสุด 3 รูป) |
| `POST` | `/api/feedback` | บันทึก feedback จากผู้ใช้ |

### ตัวอย่างเรียก API

```bash
# วิเคราะห์รูปเดี่ยว
curl -X POST http://localhost:5000/api/data \
  -F "image=@photo.jpg"

# Multi-shot
curl -X POST http://localhost:5000/api/multishot \
  -F "image[]=@photo1.jpg" \
  -F "image[]=@photo2.jpg"
```

## การเทรน Classifier

ถ้าสะสม feedback พอแล้ว หรือเพิ่มรูปใหม่ใน `phototest/`:

```bash
cd blackend

# 1. รวม dataset จากทุกแหล่ง
python build_dataset.py

# 2. เทรน classifier ใหม่
python train_classifier.py
```

## ทดสอบด้วยรูปในโฟลเดอร์

```bash
cd blackend
python run_tests.py
```

จะรันผ่านรูปทั้งหมดใน `phototest/` และแสดงผลการทำนายเทียบกับ label จริง

## Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| Backend | Python, Flask |
| Face Detection | MediaPipe Face Landmarker |
| Classifier | scikit-learn (SVM/RandomForest) |
| CNN Ensemble | DeepFace (optional) |
| Frontend | HTML, CSS, JavaScript |
| Model Format | TFLite, Pickle |
