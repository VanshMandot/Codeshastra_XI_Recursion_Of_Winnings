import cv2
import numpy as np
import os
import zipfile
import shutil
from flask import Flask, render_template, request, redirect, url_for, flash, session
from ultralytics import YOLO

model = YOLO('yolov8m.pt') 

app = Flask(__name__)
app.secret_key = "supersecretkey_for_ai_detection"

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ENVIRONMENTS_FOLDER = 'static/environments'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['HOUSES_FOLDER'] = ENVIRONMENTS_FOLDER 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(ENVIRONMENTS_FOLDER, exist_ok=True)


def recognize_objects_in_image(img):
    """
    Takes an image, runs YOLO object recognition, and returns a list of detected objects.
    """
    results = model(img)
    detected_objects = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            
            if confidence > 0.4: # Filter out low-confidence detections
                detected_objects.append({
                    'label': label,
                    'box': (x1, y1, x2 - x1, y2 - y1) # Format: x, y, width, height
                })
    return detected_objects

def draw_objectwise_changes(img1, img2):
    """
    Uses YOLO to recognize objects in the baseline image (img2) and checks if they are missing
    from the test image (img1). It only shows and names the missing items.
    """
    baseline_objects = recognize_objects_in_image(img2)
    test_objects = recognize_objects_in_image(img1)

    img1_result = img1.copy()
    img2_result = img2.copy()
    
    change_descriptions = []
    coordinate_descriptions = []

    for base_obj in baseline_objects:
        x_base, y_base, w_base, h_base = base_obj['box']
        label = base_obj['label'].title()
        
        is_present_in_test = False
        for test_obj in test_objects:
            if test_obj['label'] == base_obj['label']:
                x_test, y_test, w_test, h_test = test_obj['box']
                # Calculate Intersection over Union (IoU) to verify it's the same object
                overlap_x = max(0, min(x_base + w_base, x_test + w_test) - max(x_base, x_test))
                overlap_y = max(0, min(y_base + h_base, y_test + h_test) - max(y_base, y_test))
                intersection = overlap_x * overlap_y
                union = (w_base * h_base) + (w_test * h_test) - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > 0.5: # 50% overlap threshold
                    is_present_in_test = True
                    break
        
        if not is_present_in_test:
            color = (0, 0, 255) # Red for MISSING
            
            # Draw on test image
            cv2.rectangle(img1_result, (x_base, y_base), (x_base + w_base, y_base + h_base), color, 2)
            cv2.putText(img1_result, f"Missing: {label}", (x_base, y_base - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw on baseline image for context
            cv2.rectangle(img2_result, (x_base, y_base), (x_base + w_base, y_base + h_base), color, 2)
            cv2.putText(img2_result, label, (x_base, y_base - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            change_descriptions.append(f"'{label}' is MISSING")
            coordinate_descriptions.append(f"Missing {label} at (x={x_base}, y={y_base})")

    return img1_result, img2_result, coordinate_descriptions, change_descriptions

def analyze_environment_changes_feature_based(environment_name, test_image_path):
    """
    Analyzes changes using robust feature-based alignment.
    """
    reference_folder = os.path.join(app.config['HOUSES_FOLDER'], environment_name)
    baseline_images, _ = load_images_from_folder(reference_folder)
    test_img_original = cv2.imread(test_image_path)

    if not baseline_images or test_img_original is None:
        return None, None, None, None

    orb = cv2.ORB_create(nfeatures=2000)
    kp_test, des_test = orb.detectAndCompute(test_img_original, None)
    
    if des_test is None:
        return None, None, None, None

    best_match_info = {'image': None, 'good_matches': -1}
    for baseline_img in baseline_images:
        kp_base, des_base = orb.detectAndCompute(baseline_img, None)
        if des_base is None: continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_test, des_base)
        
        if len(matches) > best_match_info['good_matches']:
            best_match_info['good_matches'] = len(matches)
            best_match_info['image'] = baseline_img
            best_match_info['matches'] = matches
            best_match_info['kp_base'] = kp_base

    if best_match_info['good_matches'] < 10: # Require at least 10 matches
        return None, None, None, None

    best_matching_baseline = best_match_info['image']
    
    # Align Perspectives
    src_pts = np.float32([ kp_test[m.queryIdx].pt for m in best_match_info['matches'] ]).reshape(-1,1,2)
    dst_pts = np.float32([ best_match_info['kp_base'][m.trainIdx].pt for m in best_match_info['matches'] ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if M is not None:
        h, w, _ = best_matching_baseline.shape
        test_img_aligned = cv2.warpPerspective(test_img_original, M, (w, h))
    else: # Fallback if homography fails
        h, w, _ = best_matching_baseline.shape
        test_img_aligned = cv2.resize(test_img_original, (w, h))

    # Detect Changes
    test_img_boxed, baseline_boxed, coords, changes = draw_objectwise_changes(test_img_aligned, best_matching_baseline)

    # Save results
    test_img_save_path = os.path.join(app.config['RESULT_FOLDER'], 'test_result.jpg')
    match_img_save_path = os.path.join(app.config['RESULT_FOLDER'], 'baseline_result.jpg')
    cv2.imwrite(test_img_save_path, test_img_boxed)
    cv2.imwrite(match_img_save_path, baseline_boxed)
    
    return test_img_save_path, match_img_save_path, coords, changes

def load_images_from_folder(folder):
    images = []
    filenames = []  # Re-added this list
    for filename in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, filename)
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(file_path)
            if img is not None:
                images.append(img)
                filenames.append(filename) # Added this line back
    return images, filenames # Now returns two values as expected


# --- Flask Routes ---
@app.route('/')
def landing():
    """Serves the new landing page."""
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles the login logic."""
    error = None
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['logged_in'] = True
            flash('You were successfully logged in!')
            return redirect(url_for('index'))
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash('You were logged out.')
    return redirect(url_for('landing'))

@app.route('/dashboard', methods=['GET', 'POST'])
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    environments = [d for d in os.listdir(app.config['HOUSES_FOLDER']) if os.path.isdir(os.path.join(app.config['HOUSES_FOLDER'], d))]
    if request.method == 'POST' and 'folder' in request.files:
        file = request.files['folder']
        if file and file.filename.endswith('.zip'):
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(zip_path)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                env_name = os.path.splitext(file.filename)[0]
                extract_path = os.path.join(app.config['HOUSES_FOLDER'], env_name)
                os.makedirs(extract_path, exist_ok=True)
                zip_ref.extractall(extract_path)

                # Fix double-folder issue
                inner_path = os.path.join(extract_path, env_name)
                if os.path.exists(inner_path) and os.path.isdir(inner_path):
                    for item in os.listdir(inner_path):
                        shutil.move(os.path.join(inner_path, item), extract_path)
                    os.rmdir(inner_path)
            os.remove(zip_path)
            flash(f'Environment "{env_name}" uploaded successfully!')
            return redirect(url_for('index'))
        else:
            flash('Please upload a valid ZIP file.')
    return render_template('index.html', houses=environments)

@app.route('/house/<house_name>', methods=['GET', 'POST'])
def house(house_name):
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    reference_folder = os.path.join(app.config['HOUSES_FOLDER'], house_name)
    images = load_images_from_folder(reference_folder)
    # Generate relative paths for the template
    filenames = [f for f in sorted(os.listdir(reference_folder)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_paths = [f"environments/{house_name}/{fname}" for fname in filenames]

    if request.method == 'POST':
        if 'file' not in request.files or request.files['file'].filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        test_image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(test_image_path)
        
        test_result, match_result, coords, changes = analyze_environment_changes_feature_based(house_name, test_image_path)
        
        if test_result:
            return render_template('result.html', 
                                   test_img=url_for('static', filename='results/test_result.jpg'),
                                   match_img=url_for('static', filename='results/baseline_result.jpg'),
                                   coordinate_descriptions=coords,
                                   change_descriptions=changes)
        else:
            flash('Processing failed. Could not find a suitable match or detect features.')
            return redirect(request.url)
            
    return render_template('house.html', house_name=house_name, image_paths=image_paths)

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True)