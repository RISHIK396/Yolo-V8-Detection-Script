import cv2
import requests
import time
import threading
from ultralytics import YOLO
import torch
import numpy as np
import json

# ================================
# 🔧 CONFIG
# ================================
RTSP_URL = "rtsp://192.168.29.122:8554/live"
API_URL = "http://localhost:3000/detection"
DEVICE_ID = "019d25d7-ed51-7017-89f9-b7f2b619787b"

YOLO_MODEL_PATH = "best.pt"

SMOKE_THRESHOLD = 0.3
FIRE_THRESHOLD = 0.5
TEMPORAL_FRAMES = 3
COOLDOWN = 5  # seconds

# ================================
# 🚀 DEVICE SETUP
# ================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀 Using:", device)

model = YOLO(r"Rishik_2_micro_finetune\train\weights\best.pt")
model.to(device)
model.fuse()

class_names = model.names

# ================================
# 🎥 VIDEO
# ================================
cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ================================
# 🧠 STATE VARIABLES
# ================================
prev_frame = None
smoke_counter = 0
fire_counter = 0
last_sent = 0

# ================================
# 🔧 PREPROCESS (SMOKE FRIENDLY)
# ================================
def preprocess(frame):
    frame = cv2.resize(frame, (640, 640))

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    frame = cv2.merge((l, a, b))
    frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    return frame

# ================================
# 🌫️ MOTION DETECTION
# ================================
def detect_motion(prev, curr):
    if prev is None:
        return False

    diff = cv2.absdiff(prev, curr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    return np.mean(gray) > 8

# ================================
# 📡 API CALL (ASYNC)
# ================================
def send_to_backend(frame, confidence, label, crop):
    try:
        _, buffer = cv2.imencode(".jpg", frame)
        _, crop_buffer = cv2.imencode(".jpg", crop)

        files = {
            "file": ("full.jpg", buffer.tobytes(), "image/jpeg")
            # "file": ("crop.jpg", crop_buffer.tobytes(), "image/jpeg")
        }

        body = {
            "deviceId": DEVICE_ID,
            "temperature": 40,
            "smokeLevel": 0.6,
            "mlConfidence": confidence,
            "type": label,
            "sensorTriggered": False
        }

        data = {
            "body": json.dumps(body)
        }

        res = requests.post(API_URL, files=files, data=data, timeout=10)
        print("✅ API:", res.status_code)
        print("message : ",res.text)

    except Exception as e:
        print("❌ API Error:", e)

# ================================
# 🔥 MAIN LOOP
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Stream failed")
        break

    processed = preprocess(frame)

    # 🧠 Motion
    motion = detect_motion(prev_frame, processed)
    prev_frame = processed.copy()

    # 🤖 YOLO
    results = model(processed, imgsz=640, device=device, verbose=False)

    detected = False

    for r in results:
        if r.boxes is None:
            continue

        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            confidence = float(conf)
            label = class_names[int(cls)]

            x1, y1, x2, y2 = map(int, box)

            # 🎯 Threshold logic
            if label == "fire" and confidence > FIRE_THRESHOLD:
                fire_counter += 1
                color = (0, 0, 255)

            elif label == "smoke" and confidence > SMOKE_THRESHOLD:
                smoke_counter += 1
                color = (0, 255, 0)

            else:
                continue

            detected = True

            # 📦 Draw box
            cv2.rectangle(processed, (x1, y1), (x2, y2), color, 2)
            cv2.putText(processed, f"{label} {confidence:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # 📸 Crop region
            crop = processed[y1:y2, x1:x2]

    # ⏳ Temporal confirmation
    confirmed = (
        fire_counter >= TEMPORAL_FRAMES or
        (smoke_counter >= TEMPORAL_FRAMES and motion)
    )

    if not detected:
        smoke_counter = 0
        fire_counter = 0

    # 🚨 Send alert
    if confirmed and time.time() - last_sent > COOLDOWN:
        last_sent = time.time()

        print("🚨 CONFIRMED EVENT!")

        threading.Thread(
            target=send_to_backend,
            args=(processed.copy(), confidence, label, crop),
            daemon=True
        ).start()

        smoke_counter = 0
        fire_counter = 0

    # 👀 Display
    cv2.imshow("🔥 FINAL DETECTION", processed)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()