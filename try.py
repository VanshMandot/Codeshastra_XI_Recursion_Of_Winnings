# Extra

# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
# import os
# from matplotlib import pyplot as plt

# def load_images_from_folder(folder):
#     images = []
#     filenames = []
#     for filename in os.listdir(folder):
#         file_path = os.path.join(folder, filename)
#         ext = os.path.splitext(filename)[-1].lower()
#         if ext in ['.jpg', '.jpeg', '.png']:  # HEIC excluded
#             img = cv2.imread(file_path)
#         else:
#             continue
#         if img is not None:
#             images.append(img)
#             filenames.append(filename)
#     return images, filenames

# def resize_images(images, target_size=(500, 500)):
#     return [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in images]

# def compute_similarity(img1, img2):
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     score, _ = ssim(gray1, gray2, full=True)
#     return score

# def detect_objects(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
#     # Find contours in both images
#     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Filter contours by area
#     filtered_contours = []
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area > 800:  # Minimum area threshold
#             filtered_contours.append(contour)
    
#     return filtered_contours

# def draw_objectwise_changes(img1, img2):
#     img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
#     # Detect objects in both images
#     contours1 = detect_objects(img1)
#     contours2 = detect_objects(img2)
    
#     # Create copies for drawing
#     img1_result = img1.copy()
#     img2_result = img2.copy()
    
#     # Convert to grayscale for comparison
#     gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
#     # Also detect changes between images
#     diff = cv2.absdiff(gray1, gray2)
#     diff = cv2.GaussianBlur(diff, (5, 5), 0)
#     _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     change_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     # Convert change_contours to list if it's a tuple
#     change_contours_list = list(change_contours)
    
#     # Combine all contours for processing
#     all_contours = contours1 + contours2 + change_contours_list
    
#     # Remove duplicates by checking overlap
#     unique_contours = []
#     for contour in all_contours:
#         area = cv2.contourArea(contour)
#         if area < 800:  # Skip small contours
#             continue
            
#         x, y, w, h = cv2.boundingRect(contour)
#         is_duplicate = False
        
#         for existing in unique_contours:
#             ex, ey, ew, eh = cv2.boundingRect(existing)
#             # Check for significant overlap
#             overlap_x = max(0, min(x + w, ex + ew) - max(x, ex))
#             overlap_y = max(0, min(y + h, ey + eh) - max(y, ey))
#             overlap_area = overlap_x * overlap_y
#             min_area = min(w * h, ew * eh)
            
#             if overlap_area > 0.5 * min_area:  # If more than 50% overlap
#                 is_duplicate = True
#                 break
                
#         if not is_duplicate:
#             unique_contours.append(contour)
    
#     change_descriptions = []
#     coordinate_descriptions = []
    
#     for idx, contour in enumerate(unique_contours):
#         area = cv2.contourArea(contour)
#         if area > 800:  # Minimum area threshold
#             x, y, w, h = cv2.boundingRect(contour)
#             cx, cy = x + w // 2, y + h // 2
#             label = f"Object {idx + 1}"
            
#             # Extract ROIs from both images
#             # Make sure we don't go out of bounds
#             y_start = max(0, y)
#             y_end = min(gray1.shape[0], y + h)
#             x_start = max(0, x)
#             x_end = min(gray1.shape[1], x + w)
            
#             if y_end <= y_start or x_end <= x_start:
#                 continue  # Skip invalid ROIs
                
#             roi1 = gray1[y_start:y_end, x_start:x_end]
#             roi2 = gray2[y_start:y_end, x_start:x_end]
            
#             # Check if object is present in both images
#             present_in_both = False
#             try:
#                 if roi1.size > 0 and roi2.size > 0 and roi1.shape == roi2.shape:
#                     local_score = ssim(roi1, roi2)
#                     present_in_both = local_score > 0.75
#             except Exception as e:
#                 print(f"Error comparing ROIs: {e}")
                
#             # Set color based on presence
#             color = (0, 255, 0) if present_in_both else (0, 0, 255)
#             status_text = "present in both" if present_in_both else "MISSING"
            
#             # Add to descriptions
#             coordinate_descriptions.append(f"{label}: ({cx}, {cy})")
#             change_descriptions.append(f"{label} is {status_text}")
            
#             # Draw on both images
#             cv2.rectangle(img1_result, (x, y), (x + w, y + h), color, 2)
#             cv2.rectangle(img2_result, (x, y), (x + w, y + h), color, 2)
#             cv2.circle(img1_result, (cx, cy), 3, (255, 255, 255), -1)
#             cv2.circle(img2_result, (cx, cy), 3, (255, 255, 255), -1)
#             cv2.putText(img1_result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
#             cv2.putText(img2_result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    
#     return img1_result, img2_result, coordinate_descriptions, change_descriptions

# def find_closest_match(reference_folder, test_image_path):
#     ref_images, ref_filenames = load_images_from_folder(reference_folder)
#     print(f"✅ Found {len(ref_images)} reference images.")
#     if not ref_images:
#         print("❌ No reference images found.")
#         return
    
#     test_img = cv2.imread(test_image_path)
#     if test_img is None:
#         print("❌ Test image not found.")
#         return
    
#     ref_images = resize_images(ref_images, (500, 500))
#     test_img = cv2.resize(test_img, (500, 500))
    
#     best_score = -1
#     best_match_idx = -1
#     for i, ref_img in enumerate(ref_images):
#         score = compute_similarity(test_img, ref_img)
#         if score > best_score:
#             best_score = score
#             best_match_idx = i
    
#     if best_match_idx == -1:
#         print("❌ No match found.")
#         return
    
#     best_match = ref_images[best_match_idx]
#     best_filename = ref_filenames[best_match_idx]
#     print(f"\n✅ Best match: {best_filename} with similarity score: {best_score:.4f}")
    
#     test_img_boxed, best_match_boxed, coords, changes = draw_objectwise_changes(test_img.copy(), best_match.copy())
    
#     summary_text = "🧭 Coordinates of Detected Objects:\n" + "\n".join(coords) + "\n\n📋 Status:\n" + "\n".join(changes)
    
#     fig = plt.figure(figsize=(14, 6))
#     ax1 = fig.add_subplot(1, 3, 1)
#     ax1.set_title("🧪 Test Image")
#     ax1.imshow(cv2.cvtColor(test_img_boxed, cv2.COLOR_BGR2RGB))
#     ax1.axis('off')
    
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax2.set_title(f"📂 Baseline Match: {best_filename}")
#     ax2.imshow(cv2.cvtColor(best_match_boxed, cv2.COLOR_BGR2RGB))
#     ax2.axis('off')
    
#     ax3 = fig.add_subplot(1, 3, 3)
#     ax3.set_title("🔍 Summary", fontsize=11)
#     ax3.axis('off')
#     ax3.text(0, 1, summary_text, verticalalignment='top', fontsize=10, family='monospace', color='black')
    
#     plt.tight_layout()
#     plt.show()

# # === RUN ===
# reference_folder = "static/houses/house1"
# test_image_path = "static/uploads/test1.jpg"
# find_closest_match(reference_folder, test_image_path)