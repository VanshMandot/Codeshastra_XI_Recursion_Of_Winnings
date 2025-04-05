import cv2
import numpy as np
from flask import Flask, render_template, Response, request

app = Flask(__name__)

# Global variables to store images
baseline = None
changed = None

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def preprocess_image(image):
    resized = cv2.resize(image, (640, 480))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized, gray

@app.route('/capture_baseline', methods=['POST'])
def capture_baseline():
    global baseline
    frame = capture_frame()
    if frame is not None:
        resized, gray = preprocess_image(frame)
        cv2.imwrite("static/baseline.jpg", resized)
        baseline = gray
        return "Baseline captured successfully!"
    return "Failed to capture baseline."

@app.route('/capture_changed', methods=['POST'])
def capture_changed():
    global changed
    frame = capture_frame()
    if frame is not None:
        resized, gray = preprocess_image(frame)
        cv2.imwrite("static/changed.jpg", resized)
        changed = gray
        return "Changed image captured successfully!"
    return "Failed to capture changed image."

def generate_frames():
    while True:
        frame = capture_frame()
        if frame is None:
            break
        resized, gray = preprocess_image(frame)

        # Show changes only if both baseline and changed are captured
        if baseline is not None and changed is not None:
            diff = cv2.absdiff(baseline, changed)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(resized, "Change", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', resized)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)