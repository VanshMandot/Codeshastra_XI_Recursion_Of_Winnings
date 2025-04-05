# app.py
import cv2
import numpy as np
import threading
import time
import os
import joblib
import uuid
from flask import Flask, render_template, Response, request, redirect, url_for, send_file
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global Variables
camera = None
baseline = None
changed = None
stage = "idle"

# Auto capture vars
baseline_images = []
capture_thread = None
stop_baseline_capture = False

# ML model path
MODEL_PATH = "static/knn_model.pkl"
YOLO_MODEL = YOLO("yolov8n.pt")

# Camera handling

def find_working_camera():
    for i in range(5):
        temp_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if temp_cam.isOpened():
            return temp_cam
    return None

def start_camera():
    global camera
    if camera is None:
        camera = find_working_camera()

def stop_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def get_frame():
    global camera
    if camera is None:
        return None
    ret, frame = camera.read()
    return frame if ret else None

def preprocess_image(image):
    resized = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray

@app.route('/')
def index():
    baseline_files = [f"static/{img}" for img in os.listdir('static') if img.startswith('baseline_')]
    changed_exists = os.path.exists("static/changed.jpg")
    return render_template('index.html', baseline_files=baseline_files, changed_exists=changed_exists)

@app.route('/start_camera', methods=['POST'])
def start_cam():
    global stage, capture_thread, stop_baseline_capture, baseline_images
    start_camera()
    baseline_images = []
    stop_baseline_capture = False
    stage = "baseline"
    capture_thread = threading.Thread(target=capture_baseline_loop)
    capture_thread.start()
    return redirect(url_for('index'))

@app.route('/stop_camera', methods=['POST'])
def stop_cam():
    global stop_baseline_capture, capture_thread
    stop_baseline_capture = True
    if capture_thread is not None:
        capture_thread.join()
    stop_camera()
    return redirect(url_for('index'))

def capture_baseline_loop():
    global baseline_images, stop_baseline_capture
    count = 0
    while not stop_baseline_capture:
        frame = get_frame()
        if frame is not None:
            filename = f"static/baseline_{count}.jpg"
            cv2.imwrite(filename, frame)
            baseline_images.append(filename)
            print(f"[INFO] Saved: {filename}")
            count += 1
        time.sleep(3)

@app.route('/capture_changed', methods=['POST'])
def capture_changed():
    global changed, stage
    start_camera()
    frame = get_frame()
    if frame is not None:
        resized, gray = preprocess_image(frame)
        cv2.imwrite("static/changed.jpg", resized)
        changed = gray
        stage = "free"
    stop_camera()
    return redirect(url_for('index'))

@app.route('/train', methods=['POST'])
def train_model():
    features = []
    labels = []
    orb = cv2.ORB_create()

    for path in baseline_images:
        img = cv2.imread(path)
        _, gray = preprocess_image(img)
        kp, des = orb.detectAndCompute(gray, None)
        if des is not None:
            features.append(des.flatten())
            labels.append("baseline")

    if features:
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(features, labels)
        joblib.dump(knn, MODEL_PATH)
        print("[INFO] Model trained and saved.")

    return redirect(url_for('index'))

@app.route('/predict', methods=['POST'])
def predict():
    global changed
    if changed is None or not os.path.exists(MODEL_PATH):
        print("[ERROR] Model or changed image missing.")
        return redirect(url_for('index'))

    knn = joblib.load(MODEL_PATH)
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(changed, None)

    if des is not None:
        flat = des.flatten().reshape(1, -1)
        prediction = knn.predict(flat)
        print(f"[RESULT] Prediction: {prediction[0]}")
    else:
        print("[WARN] Could not compute features for changed image.")

    return redirect(url_for('index'))

@app.route('/compare', methods=['POST'])
def compare():
    global stage, baseline_images, changed
    if changed is None or not baseline_images:
        print("[ERROR] No data to compare.")
        return redirect(url_for('index'))

    orb = cv2.ORB_create()
    kp2, des2 = orb.detectAndCompute(changed, None)
    best_score = 0
    best_match_image = None
    match_visual_path = "static/match_visual.jpg"
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for path in baseline_images:
        base = cv2.imread(path)
        resized, gray = preprocess_image(base)
        kp1, des1 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None:
            continue
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        score = len(matches)
        if score > best_score:
            best_score = score
            best_match_image = path
            img_matches = cv2.drawMatches(gray, kp1, changed, kp2, matches[:20], None, flags=2)
            cv2.imwrite(match_visual_path, img_matches)

    print(f"[INFO] Best match found: {best_match_image} with {best_score} matches")
    stage = "free"
    return redirect(url_for('index'))

@app.route('/match_visual')
def match_visual():
    return send_file("static/match_visual.jpg", mimetype='image/jpeg')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return "No video file uploaded", 400
    file = request.files['video']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return redirect(url_for('process_video', filename=filename))

@app.route('/process_video/<filename>')
def process_video(filename):
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    cap = cv2.VideoCapture(video_path)

    results = []
    total_objects = 0

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = YOLO_MODEL(frame)[0]
        object_count = len(detections.boxes)
        object_positions = detections.boxes.xywh.tolist()
        annotated_frame = frame.copy()

        for box in detections.boxes:
            x, y, w, h = box.xywh[0].tolist()
            cls_id = int(box.cls[0])
            label = YOLO_MODEL.model.names[cls_id]
            cv2.rectangle(annotated_frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        snap_path = f"static/frame_{frame_id}.jpg"
        cv2.imwrite(snap_path, annotated_frame)

        results.append({
            "frame": snap_path,
            "count": object_count,
            "positions": object_positions
        })

        total_objects += object_count
        frame_id += 1

    cap.release()
    print(f"[INFO] Total objects detected across video: {total_objects}")

    return render_template("video_results.html", results=results, total=total_objects)

if __name__ == '__main__':
    app.run(debug=True)
