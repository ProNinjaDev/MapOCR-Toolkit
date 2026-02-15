import os
import pandas as pd


LABELS_FILE_PATH = os.path.join('data', 'dataset_CLEANED.csv')
RAW_IMAGES_DIR = os.path.join('data', 'dataset_crops') 

def load_raw_data_paths_and_labels():
    data_items = []
    class_labels_set = set()

    if not os.path.exists(LABELS_FILE_PATH):
        print(f"[ERROR] CSV файл не найден: {LABELS_FILE_PATH}")
        return [], class_labels_set

    try:
        df = pd.read_csv(LABELS_FILE_PATH, sep=';')
        
        if 'filename' not in df.columns or 'label' not in df.columns or 'ocr_text' not in df.columns:
            print("[ERROR] В CSV нет нужных колонок (filename, label, ocr_text)")
            return [], class_labels_set

        for index, row in df.iterrows():
            filename = row['filename']
            ocr_text = str(row['ocr_text'])
            label = str(row['label']).strip()

            if not label or label.lower() == 'nan':
                continue

            image_full_path = os.path.join(RAW_IMAGES_DIR, filename)

            if os.path.exists(image_full_path):
                data_items.append((image_full_path, ocr_text, label))
                class_labels_set.add(label)
            else:
                print(f"[WARNING] Картинка не найдена: {image_full_path}")

    except Exception as e:
        print(f"[ERROR] Ошибка чтения CSV: {e}")

    return data_items, class_labels_set

def create_class_maps(class_labels_set):
    sorted_labels = sorted(list(class_labels_set))
    
    class_to_int_map = {}
    int_to_class_map = {}
    
    for i, label in enumerate(sorted_labels):
        class_to_int_map[label] = i
        int_to_class_map[i] = label
    
    return class_to_int_map, int_to_class_map