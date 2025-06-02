import os
import cv2
import pickle
import pytesseract
from PIL import Image
import sys

RAW_IMAGES_DIR = "data/raw_demo_images"
RAW_TEXT_DIR = "data/raw_demo_text"
MODEL_PATH = "models/cleaner.pickle" 
TESSERACT_LANG = 'rus'
TESSERACT_CONFIG = '--oem 1 --psm 7' # psm 7 для строки


def load_cleaning_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def clean_image_rfr(image_cv, model, blur_and_threshold_func):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    orig_shape = gray.shape 
    
    gray_padded = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    gray_processed = blur_and_threshold_func(gray_padded) 

    roi_features = [
        gray_processed[y_p:y_p + 5, x_p:x_p + 5].flatten()
        for y_p in range(gray_processed.shape[0] - 4) 
        for x_p in range(gray_processed.shape[1] - 4)
    ]

    if not roi_features:
        print(f"[WARNING] Could not extract ROI features for image {image_cv.shape}. Returning original grayscale image")
        return gray

    pixels = model.predict(roi_features)
    
    cleaned_image_flat = (pixels * 255).astype("uint8")
    
    if cleaned_image_flat.size != (orig_shape[0] * orig_shape[1]):
        print(f"[WARNING] Prediction size {cleaned_image_flat.size} does not match original area {orig_shape[0] * orig_shape[1]}. Cannot reshape correctly")
        return gray 

    cleaned_image_cv = cleaned_image_flat.reshape(orig_shape)
    return cleaned_image_cv

def recognize_text_from_image_cv(image_cv_gray, lang, config):
    pil_image = Image.fromarray(image_cv_gray)
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
    return text.strip()

def main_process():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    try:
        from mapocr_toolkit.image_processing.blur_and_threshold import blur_and_threshold
    except ImportError:
        print("[ERROR] Failed to import 'blur_and_threshold' from 'mapocr_toolkit.image_processing'. Ensure project structure is correct and file exists")
        return

    print(f"[INFO] Starting batch processing")
    print(f"[INFO] Source images directory: {RAW_IMAGES_DIR}")
    print(f"[INFO] Output text directory: {RAW_TEXT_DIR}")
    print(f"[INFO] Cleaner model path: {MODEL_PATH}")

    if not os.path.exists(RAW_IMAGES_DIR):
        print(f"[ERROR] Source images directory not found: {RAW_IMAGES_DIR}")
        return

    if not os.path.exists(RAW_TEXT_DIR):
        print(f"[INFO] Creating output text directory: {RAW_TEXT_DIR}")
        os.makedirs(RAW_TEXT_DIR)

    try:
        print("[INFO] Loading image cleaning model")
        cleaner_model = load_cleaning_model(MODEL_PATH)
        print("[INFO] Cleaning model loaded successfully")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return
    except Exception as e:
        print(f"[ERROR] Failed to load cleaning model: {e}")
        return

    image_files = [f for f in os.listdir(RAW_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"[INFO] No images found in {RAW_IMAGES_DIR}")
        return

    print(f"[INFO] Found {len(image_files)} images to process")

    for image_filename in image_files:
        base_filename, _ = os.path.splitext(image_filename)
        image_path = os.path.join(RAW_IMAGES_DIR, image_filename)
        output_text_path = os.path.join(RAW_TEXT_DIR, f"{base_filename}.txt")

        print(f"[INFO] Processing {image_path}")

        try:
            img_cv_bgr = cv2.imread(image_path)
            if img_cv_bgr is None:
                print(f"[WARNING] Failed to read image: {image_path}. Skipping")
                continue

            cleaned_img_cv_gray = clean_image_rfr(img_cv_bgr, cleaner_model, blur_and_threshold)

            recognized_text = recognize_text_from_image_cv(cleaned_img_cv_gray, TESSERACT_LANG, TESSERACT_CONFIG)
            
            with open(output_text_path, "w", encoding="utf-8") as f:
                f.write(recognized_text)
            print(f"[INFO] Recognized text saved to {output_text_path}")
            if not recognized_text:
                print(f"[WARNING] No text recognized for {image_path}")

        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")

    print("[INFO] Batch processing complete")

if __name__ == "__main__":
    main_process() 