from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import time
import numpy as np
import pandas as pd
import requests

# ===== EMAIL IMPORTS =====
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

# Lower = stricter recognition
CONF_LIMIT = 80

INTRUDER_IMG = "intruder.jpg"

# ============================================================
# EMAIL CONFIG
# ============================================================
# IMPORTANT:
# Set these as environment variables instead of putting
# your Gmail password directly in this file.
#
# Windows CMD:
#   set EMAIL_SENDER=your_email@gmail.com
#   set EMAIL_PASSWORD=your_app_password
#   set EMAIL_RECEIVER=receiver@gmail.com
#
# PowerShell:
#   $env:EMAIL_SENDER="your_email@gmail.com"
#   $env:EMAIL_PASSWORD="your_app_password"
#   $env:EMAIL_RECEIVER="receiver@gmail.com"

EMAIL_SENDER = os.environ.get(
    "EMAIL_SENDER",
    "samueljefrin051@gmail.com"
)

EMAIL_PASSWORD = os.environ.get(
    "EMAIL_PASSWORD",
    "nfki omwc qfuu vpir"
)

EMAIL_RECEIVER = os.environ.get(
    "EMAIL_RECEIVER",
    "Jerinjefrrin2007@gmail.com"
)

EMAIL_COOLDOWN = 60

# ============================================================
# FACE DETECTION SETTINGS
# ============================================================

# Detection
FACE_SCALE_FACTOR = 1.10
FACE_MIN_NEIGHBORS = 6

# Ignore extremely small detections
MIN_FACE_WIDTH = 45
MIN_FACE_HEIGHT = 45

# Extra space around detected face
FACE_PADDING = 0.18

# Smoothing factor:
# Smaller = smoother
# Larger = faster response
BOX_SMOOTHING = 0.65

# Maximum movement allowed per frame.
# Helps prevent the box from jumping to random detections.
MAX_BOX_MOVEMENT = 80

# Recognition image size
FACE_SIZE = (200, 200)

# ============================================================
# APP
# ============================================================

os.makedirs(FACE_DIR, exist_ok=True)

app = Flask(__name__)

mode = "live"

latest_frame = None

last_buzzer_time = 0
last_email_time = 0

# ============================================================
# FACE DETECTOR
# ============================================================

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade face detector")

# ============================================================
# FACE RECOGNIZER
# ============================================================

recognizer = cv2.face.LBPHFaceRecognizer_create()

model_trained = False

# ============================================================
# TRACKING DATA
# ============================================================

# Stores smoothed boxes:
# {
#     tracking_id: [x, y, w, h]
# }
smooth_boxes = {}

next_tracking_id = 0


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    global model_trained

    model_trained = False

    if not os.path.exists(CSV_FILE):
        print("[FACE] people.csv not found")
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

        img_path = os.path.join(
            FACE_DIR,
            f"{pid}.jpg"
        )

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            continue

        img = cv2.resize(
            img,
            FACE_SIZE
        )

        # Histogram equalization helps recognition
        # when lighting changes.
        img = cv2.equalizeHist(img)

        faces.append(img)
        labels.append(pid)

    if len(faces) > 0:

        recognizer.train(
            faces,
            np.array(labels)
        )

        model_trained = True

        print(
            f"[FACE] Model trained with {len(faces)} face(s)"
        )

    else:

        print("[FACE] No registered faces found")


# Train when application starts
train_model()


# ============================================================
# GET PERSON NAME
# ============================================================

def get_name(pid):

    if not os.path.exists(CSV_FILE):
        return "Unknown"

    try:

        df = pd.read_csv(CSV_FILE)

        df["id"] = pd.to_numeric(
            df["id"],
            errors="coerce"
        )

        result = df[df["id"] == int(pid)]

        if not result.empty:

            return str(
                result.iloc[0]["name"]
            )

    except Exception as e:

        print("[NAME ERROR]", e)

    return "Unknown"


# ============================================================
# BUZZER
# ============================================================

def set_buzzer(state):

    global last_buzzer_time

    now = time.time()

    if now - last_buzzer_time < 0.5:
        return

    last_buzzer_time = now

    try:

        requests.get(
            ESP32_BUZZER + state,
            timeout=0.2
        )

    except Exception:
        pass


# ============================================================
# CAMERA
# ============================================================

def open_camera():

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_DSHOW
    )

    if not cap.isOpened():

        print(
            "[CAMERA ERROR] "
            "Could not open laptop camera"
        )

        return None

    # Camera resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    # Keep only the newest frame
    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Try to reduce autofocus exposure instability
    try:
        cap.set(
            cv2.CAP_PROP_FPS,
            30
        )
    except:
        pass

    print(
        "[CAMERA] Laptop camera connected"
    )

    return cap


# ============================================================
# EMAIL
# ============================================================

def send_intruder_email(frame):

    global last_email_time

    now = time.time()

    if now - last_email_time < EMAIL_COOLDOWN:
        return

    if not EMAIL_PASSWORD:

        print(
            "[EMAIL] EMAIL_PASSWORD not configured"
        )

        return

    last_email_time = now

    try:

        cv2.imwrite(
            INTRUDER_IMG,
            frame
        )

        msg = MIMEMultipart()

        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = (
            "ALERT: Unknown Person Detected"
        )

        body = f"""
ALERT!

Unknown person detected by
ESP32 AI Intruder System.

Time:
{time.strftime("%Y-%m-%d %H:%M:%S")}

Mode:
LIVE TRACK
"""

        msg.attach(
            MIMEText(
                body,
                "plain"
            )
        )

        with open(
            INTRUDER_IMG,
            "rb"
        ) as f:

            img = MIMEImage(
                f.read()
            )

            img.add_header(
                "Content-Disposition",
                "attachment",
                filename="intruder.jpg"
            )

            msg.attach(img)

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            EMAIL_SENDER,
            EMAIL_PASSWORD
        )

        server.send_message(msg)

        server.quit()

        print(
            "[EMAIL] Intruder alert sent"
        )

    except Exception as e:

        print(
            "[EMAIL ERROR]",
            e
        )


# ============================================================
# BOX UTILITIES
# ============================================================

def expand_face_box(
    x,
    y,
    w,
    h,
    frame_width,
    frame_height
):

    """
    Adds padding around the face.

    This makes the box more visually useful and gives
    the recognizer a little more stable area.
    """

    pad_x = int(
        w * FACE_PADDING
    )

    pad_y = int(
        h * FACE_PADDING
    )

    x1 = max(
        0,
        x - pad_x
    )

    y1 = max(
        0,
        y - pad_y
    )

    x2 = min(
        frame_width,
        x + w + pad_x
    )

    y2 = min(
        frame_height,
        y + h + pad_y
    )

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


# ============================================================
# BOX SMOOTHING
# ============================================================

def smooth_box(
    current_box,
    previous_box
):

    if previous_box is None:
        return current_box

    x, y, w, h = current_box

    px, py, pw, ph = previous_box

    # Check large sudden jump
    center_x = x + w / 2
    center_y = y + h / 2

    previous_center_x = (
        px + pw / 2
    )

    previous_center_y = (
        py + ph / 2
    )

    distance = np.sqrt(
        (center_x - previous_center_x) ** 2 +
        (center_y - previous_center_y) ** 2
    )

    if distance > MAX_BOX_MOVEMENT:

        # Ignore sudden jump
        return previous_box

    alpha = BOX_SMOOTHING

    new_x = int(
        alpha * x +
        (1 - alpha) * px
    )

    new_y = int(
        alpha * y +
        (1 - alpha) * py
    )

    new_w = int(
        alpha * w +
        (1 - alpha) * pw
    )

    new_h = int(
        alpha * h +
        (1 - alpha) * ph
    )

    return (
        new_x,
        new_y,
        new_w,
        new_h
    )


# ============================================================
# FIND BEST FACE
# ============================================================

def detect_faces(gray):

    """
    Detect faces and filter weak/small detections.
    """

    detected = face_detector.detectMultiScale(
        gray,
        scaleFactor=FACE_SCALE_FACTOR,
        minNeighbors=FACE_MIN_NEIGHBORS,
        minSize=(
            MIN_FACE_WIDTH,
            MIN_FACE_HEIGHT
        )
    )

    if detected is None:
        return []

    faces = []

    for (
        x,
        y,
        w,
        h
    ) in detected:

        if (
            w < MIN_FACE_WIDTH or
            h < MIN_FACE_HEIGHT
        ):
            continue

        faces.append(
            (
                int(x),
                int(y),
                int(w),
                int(h)
            )
        )

    # Largest faces first
    faces.sort(
        key=lambda box: box[2] * box[3],
        reverse=True
    )

    return faces


# ============================================================
# RECOGNITION
# ============================================================

def recognize_face(gray, box):

    x, y, w, h = box

    frame_h, frame_w = gray.shape

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        frame_w,
        x + w
    )

    y2 = min(
        frame_h,
        y + h
    )

    if x2 <= x1 or y2 <= y1:

        return (
            None,
            999
        )

    face_img = gray[
        y1:y2,
        x1:x2
    ]

    if face_img.size == 0:

        return (
            None,
            999
        )

    face_img = cv2.resize(
        face_img,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Improve lighting consistency
    face_img = cv2.equalizeHist(
        face_img
    )

    if not model_trained:

        return (
            None,
            999
        )

    try:

        pid, confidence = recognizer.predict(
            face_img
        )

        return (
            int(pid),
            float(confidence)
        )

    except Exception as e:

        print(
            "[RECOGNITION ERROR]",
            e
        )

        return (
            None,
            999
        )


# ============================================================
# VIDEO STREAM
# ============================================================

def gen_frames():

    global latest_frame
    global mode
    global smooth_boxes
    global next_tracking_id

    cap = open_camera()

    if cap is None:
        return

    while True:

        ret, frame = cap.read()

        if (
            not ret or
            frame is None
        ):

            print(
                "[CAMERA] Frame read failed"
            )

            cap.release()

            time.sleep(1)

            cap = open_camera()

            if cap is None:
                continue

            continue

        # ----------------------------------------------------
        # Mirror camera for more natural interaction
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        latest_frame = frame.copy()

        frame_height, frame_width = (
            frame.shape[:2]
        )

        # ----------------------------------------------------
        # Grayscale
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Small blur reduces camera noise
        gray_detection = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        detected_faces = detect_faces(
            gray_detection
        )

        intruder = False

        current_boxes = {}

        # ----------------------------------------------------
        # Process each face
        # ----------------------------------------------------

        for index, face in enumerate(
            detected_faces
        ):

            x, y, w, h = face

            # Expand box slightly
            box = expand_face_box(
                x,
                y,
                w,
                h,
                frame_width,
                frame_height
            )

            # ------------------------------------------------
            # Find closest previous tracking box
            # ------------------------------------------------

            previous_box = None

            if index in smooth_boxes:

                previous_box = smooth_boxes[
                    index
                ]

            # Smooth movement
            box = smooth_box(
                box,
                previous_box
            )

            current_boxes[index] = box

            x, y, w, h = box

            # ------------------------------------------------
            # Default values
            # ------------------------------------------------

            label = "FACE"

            color = (
                255,
                255,
                0
            )

            # ------------------------------------------------
            # ADD MODE
            # ------------------------------------------------

            if mode == "add":

                label = "ADD PERSON MODE"

            # ------------------------------------------------
            # VERIFY MODE
            # ------------------------------------------------

            elif mode == "verify":

                if model_trained:

                    pid, confidence = recognize_face(
                        gray,
                        box
                    )

                    if (
                        pid is not None and
                        confidence < CONF_LIMIT
                    ):

                        label = (
                            f"VERIFIED: "
                            f"{get_name(pid)}"
                        )

                        color = (
                            0,
                            255,
                            0
                        )

                    else:

                        label = (
                            "NOT REGISTERED"
                        )

                        color = (
                            0,
                            0,
                            255
                        )

                else:

                    label = (
                        "NO REGISTERED FACES"
                    )

                    color = (
                        0,
                        0,
                        255
                    )

            # ------------------------------------------------
            # LIVE MODE
            # ------------------------------------------------

            else:

                if model_trained:

                    pid, confidence = recognize_face(
                        gray,
                        box
                    )

                    if (
                        pid is not None and
                        confidence < CONF_LIMIT
                    ):

                        label = (
                            f"KNOWN: "
                            f"{get_name(pid)}"
                        )

                        color = (
                            0,
                            255,
                            0
                        )

                    else:

                        label = "UNKNOWN"

                        color = (
                            0,
                            0,
                            255
                        )

                        intruder = True

                else:

                    label = "UNKNOWN"

                    color = (
                        0,
                        0,
                        255
                    )

                    intruder = True

            # ------------------------------------------------
            # Draw face box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            # ------------------------------------------------
            # Label background
            # ------------------------------------------------

            font = cv2.FONT_HERSHEY_SIMPLEX

            font_scale = 0.65
            thickness = 2

            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                label,
                font,
                font_scale,
                thickness
            )

            label_y = max(
                0,
                y - text_height - 12
            )

            # Background rectangle
            cv2.rectangle(
                frame,
                (
                    x,
                    label_y
                ),
                (
                    x + text_width + 10,
                    label_y + text_height + 10
                ),
                color,
                -1
            )

            # Label text
            cv2.putText(
                frame,
                label,
                (
                    x + 5,
                    label_y + text_height + 4
                ),
                font,
                font_scale,
                (
                    0,
                    0,
                    0
                ),
                thickness,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Save smoothed boxes
        # ----------------------------------------------------

        smooth_boxes = current_boxes

        # ----------------------------------------------------
        # LIVE ACTIONS
        # ----------------------------------------------------

        if mode == "live":

            if intruder:

                set_buzzer("1")

                send_intruder_email(
                    frame
                )

            else:

                set_buzzer("0")

        else:

            set_buzzer("0")

        # ----------------------------------------------------
        # STREAM FRAME
        # ----------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# VIDEO
# ============================================================

@app.route("/video")
def video():

    return Response(
        gen_frames(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# CHANGE MODE
# ============================================================

@app.route(
    "/set_mode",
    methods=["POST"]
)
def set_mode():

    global mode
    global smooth_boxes

    data = request.get_json(
        silent=True
    ) or {}

    new_mode = data.get(
        "mode",
        "live"
    )

    if new_mode not in [
        "live",
        "add",
        "verify"
    ]:

        new_mode = "live"

    mode = new_mode

    # Reset tracking when changing mode
    smooth_boxes = {}

    print(
        f"[MODE] Changed to: {mode}"
    )

    return jsonify({
        "mode": mode
    })


# ============================================================
# ADD PERSON
# ============================================================

@app.route(
    "/add_person",
    methods=["POST"]
)
def add_person():

    global latest_frame

    if latest_frame is None:

        return (
            "Camera warming up, "
            "wait 2 seconds and try again"
        )

    name = request.form.get(
        "name",
        "Unknown"
    ).strip()

    if not name:

        return "Please enter a name"

    # --------------------------------------------------------
    # Generate unique ID
    # --------------------------------------------------------

    pid = int(
        time.time() * 1000
    )

    # --------------------------------------------------------
    # Convert current frame
    # --------------------------------------------------------

    frame = latest_frame.copy()

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    gray_detection = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    faces = detect_faces(
        gray_detection
    )

    if len(faces) == 0:

        return (
            "No face detected. "
            "Look directly at the camera."
        )

    # Use largest face
    x, y, w, h = faces[0]

    # Add padding
    x, y, w, h = expand_face_box(
        x,
        y,
        w,
        h,
        frame.shape[1],
        frame.shape[0]
    )

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    face_img = gray[
        y:y + h,
        x:x + w
    ]

    if face_img.size == 0:

        return (
            "Could not capture face. "
            "Try again."
        )

    # Resize
    face_img = cv2.resize(
        face_img,
        FACE_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # Normalize lighting
    face_img = cv2.equalizeHist(
        face_img
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    image_path = os.path.join(
        FACE_DIR,
        f"{pid}.jpg"
    )

    saved = cv2.imwrite(
        image_path,
        face_img
    )

    if not saved:

        return (
            "Failed to save face image"
        )

    # --------------------------------------------------------
    # Update CSV
    # --------------------------------------------------------

    try:

        if os.path.exists(CSV_FILE):

            df = pd.read_csv(
                CSV_FILE
            )

        else:

            df = pd.DataFrame(
                columns=[
                    "id",
                    "name"
                ]
            )

        new_row = pd.DataFrame(
            [{
                "id": pid,
                "name": name
            }]
        )

        df = pd.concat(
            [
                df,
                new_row
            ],
            ignore_index=True
        )

        df.to_csv(
            CSV_FILE,
            index=False
        )

    except Exception as e:

        print(
            "[CSV ERROR]",
            e
        )

        return (
            "Face saved but CSV update failed"
        )

    # --------------------------------------------------------
    # Retrain
    # --------------------------------------------------------

    train_model()

    return (
        "Person Added Successfully"
    )


# ============================================================
# CAMERA STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({
        "mode": mode,
        "model_trained": model_trained,
        "camera_ready": latest_frame is not None
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("ESP32 AI INTRUDER SYSTEM")
    print("=" * 50)

    print(
        f"Face confidence limit: {CONF_LIMIT}"
    )

    print(
        f"Face scale factor: {FACE_SCALE_FACTOR}"
    )

    print(
        f"Face padding: {FACE_PADDING * 100:.0f}%"
    )

    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True
    )