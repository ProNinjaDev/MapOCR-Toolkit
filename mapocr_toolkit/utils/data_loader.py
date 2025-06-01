import os
import csv

LABELS_FILE_PATH = os.path.join('data', 'raw_demo_images', 'labels.csv')
RAW_TEXT_DIR = os.path.join('data', 'raw_demo_text')

def load_raw_data_paths_and_labels():
    data_items = []
    class_labels_set = set()

    if not os.path.exists(LABELS_FILE_PATH):
        print(f"[INFO] The tag file was not found: {LABELS_FILE_PATH}")
        return [], class_labels_set

    with open(LABELS_FILE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            if len(row) == 2:
                image_filename_with_ext, class_label = row
                base_filename, _ = os.path.splitext(image_filename_with_ext)
                text_filename = f"{base_filename}.txt"
                text_file_path = os.path.join(RAW_TEXT_DIR, text_filename)
                
                if os.path.exists(text_file_path):
                    data_items.append((text_file_path, class_label.strip()))
                    class_labels_set.add(class_label.strip())
                else:
                    print(f"[WARNING] The text file was not found for the label {image_filename_with_ext}: {text_file_path}")
            else:
                print(f"[WARNING] Invalid line in labels.csv: {row}")
                
    return data_items, class_labels_set

if __name__ == '__main__':
    items, labels = load_raw_data_paths_and_labels()
    if items:
        print(f"Loaded items: {len(items)}")
        print("The first 5 elements:")
        for item in items[:5]:
            print(item)
        print(f"\nUnique class labels: {labels}")
        print(f"Number of unique labels: {len(labels)}") 