import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import zipfile
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
HOUSES_FOLDER = 'static/houses'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['HOUSES_FOLDER'] = HOUSES_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(HOUSES_FOLDER, exist_ok=True)

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        ext = os.path.splitext(filename)[-1].lower()
        if ext in ['.jpg', '.jpeg', '.png']:
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames

def resize_images(images, target_size=(500, 500)):
    resized_images = []
    for img in images:
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        resized_images.append(resized)
    return resized_images

def compute_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

def draw_separate_difference_boxes(img1, img2):
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 800:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            label = f"Object {idx+1}"
            cv2.circle(img1, (cx, cy), 3, (255, 255, 255), -1)
            cv2.circle(img2, (cx, cy), 3, (255, 255, 255), -1)
            cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.rectangle(img2, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img1, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(img2, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return img1, img2

def find_closest_match(reference_folder, test_image_path):
    ref_images, ref_filenames = load_images_from_folder(reference_folder)
    if not ref_images:
        return None, None

    test_img = cv2.imread(test_image_path)
    if test_img is None:
        return None, None

    target_size = (500, 500)
    ref_images = resize_images(ref_images, target_size)
    test_img = cv2.resize(test_img, target_size, interpolation=cv2.INTER_AREA)

    best_score = -1
    best_match_idx = -1
    for i, ref_img in enumerate(ref_images):
        score = compute_similarity(test_img, ref_img)
        if score > best_score:
            best_score = score
            best_match_idx = i

    if best_match_idx == -1:
        return None, None

    best_match = ref_images[best_match_idx]
    test_img_boxed, best_match_boxed = draw_separate_difference_boxes(test_img.copy(), best_match.copy())

    test_img_save_path = os.path.join(app.config['RESULT_FOLDER'], 'test_result.jpg')
    match_img_save_path = os.path.join(app.config['RESULT_FOLDER'], 'match_result.jpg')
    cv2.imwrite(test_img_save_path, test_img_boxed)
    cv2.imwrite(match_img_save_path, best_match_boxed)

    return test_img_save_path, match_img_save_path

@app.route('/', methods=['GET', 'POST'])
def index():
    houses = [d for d in os.listdir(app.config['HOUSES_FOLDER']) if os.path.isdir(os.path.join(app.config['HOUSES_FOLDER'], d))]

    if request.method == 'POST' and 'folder' in request.files:
        file = request.files['folder']
        if file.filename == '':
            flash('No selected file')
        elif file and file.filename.endswith('.zip'):
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                house_name = zip_ref.namelist()[0].split('/')[0]
                extract_path = os.path.join(app.config['HOUSES_FOLDER'], house_name)
                zip_ref.extractall(app.config['HOUSES_FOLDER'])
            os.remove(zip_path)
            flash(f'House "{house_name}" uploaded successfully!')
            return redirect(url_for('index'))
        else:
            flash('Please upload a ZIP file')

    return render_template('index.html', houses=houses)

@app.route('/house/<house_name>', methods=['GET', 'POST'])
def house(house_name):
    reference_folder = os.path.join(app.config['HOUSES_FOLDER'], house_name)
    images, filenames = load_images_from_folder(reference_folder)
    image_paths = [f"houses/{house_name}/{fname}" for fname in filenames]

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(test_image_path)
            test_result, match_result = find_closest_match(reference_folder, test_image_path)
            if test_result and match_result:
                return render_template('result.html', test_img=url_for('static', filename='results/test_result.jpg'),
                                       match_img=url_for('static', filename='results/match_result.jpg'))
            else:
                flash('Processing failed')
                return redirect(request.url)
    return render_template('house.html', house_name=house_name, image_paths=image_paths)

if __name__ == '__main__':
    app.run(debug=True)