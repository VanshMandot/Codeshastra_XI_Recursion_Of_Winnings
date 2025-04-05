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

    tracker = SimpleObjectTracker()
    unique_ids = set()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = YOLO_MODEL(frame)[0]
        annotated_frame = frame.copy()
        object_positions = []

        for box in detections.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = YOLO_MODEL.names[cls_id]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            object_positions.append([cx, cy, w, h])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        tracked_objects = tracker.update(object_positions)
        for obj_id, (x, y, w, h) in tracked_objects.items():
            unique_ids.add(obj_id)
            cv2.putText(annotated_frame, f"ID: {obj_id}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out_path = f"static/processed_frame_{frame_count}.jpg"
        cv2.imwrite(out_path, annotated_frame)
        frame_count += 1

    cap.release()
    print(f"[INFO] Total unique objects detected across video: {len(unique_ids)}")
    return render_template("video_results.html", unique_count=len(unique_ids))

class SimpleObjectTracker:
    def __init__(self, iou_threshold=0.3):
        self.next_id = 0
        self.tracks = {}
        self.iou_threshold = iou_threshold

    def update(self, detections):
        updated_tracks = {}
        used_ids = set()

        for det in detections:
            if not isinstance(det, (list, tuple)) or len(det) != 4:
                continue
            x, y, w, h = det
            best_iou = 0
            best_id = None

            for obj_id, (tx, ty, tw, th) in self.tracks.items():
                iou = self.compute_iou((x, y, w, h), (tx, ty, tw, th))
                if iou > best_iou and iou >= self.iou_threshold and obj_id not in used_ids:
                    best_iou = iou
                    best_id = obj_id

            if best_id is not None:
                updated_tracks[best_id] = (x, y, w, h)
                used_ids.add(best_id)
            else:
                updated_tracks[self.next_id] = (x, y, w, h)
                self.next_id += 1

        self.tracks = updated_tracks
        return self.tracks

    def compute_iou(self, boxA, boxB):
        ax, ay, aw, ah = boxA
        bx, by, bw, bh = boxB

        ax1, ay1 = ax - aw/2, ay - ah/2
        ax2, ay2 = ax + aw/2, ay + ah/2
        bx1, by1 = bx - bw/2, by - bh/2
        bx2, by2 = bx + bw/2, by + bh/2

        xA = max(ax1, bx1)
        yA = max(ay1, by1)
        xB = min(ax2, bx2)
        yB = min(ay2, by2)

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (ax2 - ax1) * (ay2 - ay1)
        boxBArea = (bx2 - bx1) * (by2 - by1)
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        return iou

if __name__ == '__main__':
    app.run(debug=True)