from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time
import numpy as np
import pandas as pd
import requests

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ============================================================
# CONFIG
# ============================================================
ESP32_STREAM = "http://192.168.29.188/stream"
ESP32_BUZZER = "http://192.168.29.188/buzzer?state="

FACE_DIR = "faces"
CSV_FILE = "people.csv"
CONF_LIMIT = 80
INTRUDER_IMG = "intruder.jpg"

EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "samueljefrin051@gmail.com")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "nfki omwc qfuu vpir")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "mariappanpvm2005@gmail.com")
EMAIL_COOLDOWN = 60

# ============================================================
# FACE DETECTION SETTINGS
# ============================================================
FACE_SCALE_FACTOR = 1.10
FACE_MIN_NEIGHBORS = 6
MIN_FACE_WIDTH = 45
MIN_FACE_HEIGHT = 45
FACE_PADDING = 0.18
BOX_SMOOTHING = 0.65
MAX_BOX_MOVEMENT = 80
FACE_SIZE = (200, 200)

# ============================================================
# APP STATE
# ============================================================
os.makedirs(FACE_DIR, exist_ok=True)
app = Flask(__name__)

latest_frame = None
last_buzzer_time = 0
last_email_time = 0
unknown_detected = False
last_email_status = "idle"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade face detector")

recognizer = cv2.face.LBPHFaceRecognizer_create()
model_trained = False
smooth_boxes = {}
next_tracking_id = 0

# ============================================================
# TRAIN MODEL ON STARTUP
# ============================================================
def train_model():
    global model_trained
    model_trained = False

    if not os.path.exists(CSV_FILE):
        print("[FACE] people.csv not found. Create people.csv and add photos to 'faces/' directory to train.")
        return

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print("[FACE] CSV read error:", e)
        return

    faces = []
    labels = []

    for _, row in df.iterrows():
        try:
            pid = int(row["id"])
        except:
            continue

        img_path = os.path.join(FACE_DIR, f"{pid}.jpg")
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, FACE_SIZE)
        img = cv2.equalizeHist(img)
        
        faces.append(img)
        labels.append(pid)

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        model_trained = True
        print(f"[FACE] Model trained with {len(faces)} face(s)")
    else:
        print("[FACE] No registered faces found")

train_model()

# ============================================================
# UTILITIES
# ============================================================
def get_name(pid):
    if not os.path.exists(CSV_FILE): return "Unknown"
    try:
        df = pd.read_csv(CSV_FILE)
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        result = df[df["id"] == int(pid)]
        if not result.empty: return str(result.iloc[0]["name"])
    except Exception as e:
        print("[NAME ERROR]", e)
    return "Unknown"

def set_buzzer(state):
    global last_buzzer_time
    now = time.time()
    if now - last_buzzer_time < 0.5: return
    last_buzzer_time = now
    try:
        requests.get(ESP32_BUZZER + state, timeout=0.2)
    except Exception:
        pass

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[CAMERA ERROR] Could not open laptop camera")
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try: cap.set(cv2.CAP_PROP_FPS, 30)
    except: pass
    print("[CAMERA] Camera connected")
    return cap

def send_intruder_email(frame):
    global last_email_time, last_email_status
    now = time.time()
    if now - last_email_time < EMAIL_COOLDOWN:
        return
    if not EMAIL_PASSWORD:
        last_email_status = "not_configured"
        return

    last_email_time = now
    last_email_status = "pending"
    try:
        cv2.imwrite(INTRUDER_IMG, frame)
        msg = MIMEMultipart("related")
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = "Anomaly Face Detected"

        detected_at = time.strftime("%d %b %Y, %I:%M:%S %p")
        body = f"""\
<html>
  <body style="margin:0;padding:24px;background:#f1f5f9;font-family:Arial,Helvetica,sans-serif;color:#172033;">
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:640px;margin:0 auto;background:#ffffff;border:1px solid #e2e8f0;border-radius:14px;overflow:hidden;">
      <tr><td style="padding:22px 28px;background:#111827;color:#ffffff;font-size:16px;font-weight:700;letter-spacing:.08em;">ANOMALY DETECTION USING MACHINE LEARNING</td></tr>
      <tr><td style="padding:28px;">
        <div style="margin-bottom:10px;color:#b45309;font-size:12px;font-weight:700;letter-spacing:.12em;">SURVEILLANCE ALERT</div>
        <h1 style="margin:0 0 12px;font-size:26px;line-height:1.25;color:#172033;">Anomaly Face Detected</h1>
        <p style="margin:0 0 22px;color:#475569;font-size:15px;line-height:1.6;">An unknown face has been detected by the machine-learning surveillance system and is currently under surveillance.</p>
        <table role="presentation" cellspacing="0" cellpadding="0" style="width:100%;margin-bottom:22px;border-collapse:collapse;">
          <tr><td style="padding:11px 0;border-bottom:1px solid #e2e8f0;color:#64748b;font-size:13px;">Detection time</td><td style="padding:11px 0;border-bottom:1px solid #e2e8f0;text-align:right;color:#172033;font-size:13px;font-weight:600;">{detected_at}</td></tr>
          <tr><td style="padding:11px 0;border-bottom:1px solid #e2e8f0;color:#64748b;font-size:13px;">Detection status</td><td style="padding:11px 0;border-bottom:1px solid #e2e8f0;text-align:right;color:#b45309;font-size:13px;font-weight:700;">Unknown face under surveillance</td></tr>
          <tr><td style="padding:11px 0;color:#64748b;font-size:13px;">Camera / source</td><td style="padding:11px 0;text-align:right;color:#172033;font-size:13px;font-weight:600;">Live surveillance camera</td></tr>
        </table>
        <div style="padding:6px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc;">
          <img src="cid:anomaly-snapshot" alt="Captured anomaly face snapshot" style="display:block;width:100%;max-width:560px;height:auto;margin:0 auto;border-radius:6px;">
        </div>
        <p style="margin:18px 0 0;color:#64748b;font-size:12px;line-height:1.5;">This is an automated notification from the Anomaly Detection using Machine Learning surveillance system.</p>
      </td></tr>
    </table>
  </body>
</html>"""
        msg.attach(MIMEText(body, "html"))

        with open(INTRUDER_IMG, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", "<anomaly-snapshot>")
            img.add_header("Content-Disposition", "inline", filename="anomaly-snapshot.jpg")
            msg.attach(img)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        last_email_status = "sent"
        print("[EMAIL] Anomaly notification sent")
    except Exception as e:
        last_email_status = "failed"
        print("[EMAIL ERROR]", e)

def expand_face_box(x, y, w, h, frame_width, frame_height):
    pad_x, pad_y = int(w * FACE_PADDING), int(h * FACE_PADDING)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2 = min(frame_width, x + w + pad_x)
    y2 = min(frame_height, y + h + pad_y)
    return (x1, y1, x2 - x1, y2 - y1)

def smooth_box(current_box, previous_box):
    if previous_box is None: return current_box
    x, y, w, h = current_box
    px, py, pw, ph = previous_box

    center_x, center_y = x + w / 2, y + h / 2
    previous_center_x, previous_center_y = px + pw / 2, py + ph / 2
    distance = np.sqrt((center_x - previous_center_x) ** 2 + (center_y - previous_center_y) ** 2)

    if distance > MAX_BOX_MOVEMENT: return previous_box

    alpha = BOX_SMOOTHING
    return (int(alpha * x + (1 - alpha) * px), int(alpha * y + (1 - alpha) * py), 
            int(alpha * w + (1 - alpha) * pw), int(alpha * h + (1 - alpha) * ph))

def detect_faces(gray):
    detected = face_detector.detectMultiScale(gray, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBORS, minSize=(MIN_FACE_WIDTH, MIN_FACE_HEIGHT))
    if detected is None: return []
    
    faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected if w >= MIN_FACE_WIDTH and h >= MIN_FACE_HEIGHT]
    faces.sort(key=lambda box: box[2] * box[3], reverse=True)
    return faces

def recognize_face(gray, box):
    x, y, w, h = box
    frame_h, frame_w = gray.shape
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame_w, x + w), min(frame_h, y + h)

    if x2 <= x1 or y2 <= y1: return (None, 999)
    face_img = gray[y1:y2, x1:x2]
    if face_img.size == 0: return (None, 999)

    face_img = cv2.resize(face_img, FACE_SIZE, interpolation=cv2.INTER_AREA)
    face_img = cv2.equalizeHist(face_img)

    if not model_trained: return (None, 999)

    try:
        pid, confidence = recognizer.predict(face_img)
        return (int(pid), float(confidence))
    except Exception as e:
        print("[RECOGNITION ERROR]", e)
        return (None, 999)

# ============================================================
# VIDEO STREAM LOGIC
# ============================================================
def gen_frames():
    global latest_frame, smooth_boxes, unknown_detected

    cap = open_camera()
    if cap is None: return

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[CAMERA] Frame read failed")
            cap.release()
            time.sleep(1)
            cap = open_camera()
            continue

        frame = cv2.flip(frame, 1)
        latest_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_detection = cv2.GaussianBlur(gray, (3, 3), 0)
        detected_faces = detect_faces(gray_detection)
        
        intruder = False
        current_boxes = {}

        for index, face in enumerate(detected_faces):
            x, y, w, h = face
            box = expand_face_box(x, y, w, h, frame_width, frame_height)
            previous_box = smooth_boxes.get(index, None)
            box = smooth_box(box, previous_box)
            current_boxes[index] = box
            x, y, w, h = box

            label = "UNKNOWN"
            color = (0, 0, 255)

            if model_trained:
                pid, confidence = recognize_face(gray, box)
                if pid is not None and confidence < CONF_LIMIT:
                    label = f"KNOWN: {get_name(pid)}"
                    color = (0, 255, 0)
                else:
                    intruder = True
            else:
                intruder = True

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            label_y = max(0, y - text_height - 12)

            cv2.rectangle(frame, (x, label_y), (x + text_width + 10, label_y + text_height + 10), color, -1)
            cv2.putText(frame, label, (x + 5, label_y + text_height + 4), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        smooth_boxes = current_boxes
        unknown_detected = intruder

        if intruder:
            set_buzzer("1")
            send_intruder_email(frame)
        else:
            set_buzzer("0")

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success: continue

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

# ============================================================
# ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return jsonify({
        "model_trained": model_trained,
        "camera_ready": latest_frame is not None,
        "unknown_detected": unknown_detected,
        "email_status": last_email_status
    })

if __name__ == "__main__":
    print("=" * 50)
    print("ESP32 AI INTRUDER SYSTEM - LIVE MONITORING MODE")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)