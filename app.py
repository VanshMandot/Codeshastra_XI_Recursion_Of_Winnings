# app.py
import cv2
import numpy as np
import threading
import time
import os
import joblib
from flask import Flask, render_template, Response, request, redirect, url_for, send_file
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO

app = Flask(__name__)

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


def detect_objects(frame):
    results = YOLO_MODEL(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = results.names[int(box.cls[0])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame


def generate_frames():
    global camera, baseline, changed, stage
    while True:
        if camera is None:
            break
        frame = get_frame()
        if frame is None:
            continue

        frame = detect_objects(frame)
        resized, gray = preprocess_image(frame)

        if stage == "free" and baseline is not None and changed is not None:
            diff = cv2.absdiff(baseline, changed)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(resized, "Change", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', resized)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


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


@app.route('/')
def index():
    files = []
    if os.path.exists("static"):
        files = sorted([f for f in os.listdir("static") if f.startswith("baseline_")],
                       key=lambda x: int(x.split('_')[1].split('.')[0]))
    return render_template("index.html", stage=stage, baseline_files=files,
                           changed_exists=os.path.exists("static/changed.jpg"))


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


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


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


if __name__ == '__main__':
    app.run(debug=True)
