import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
if project_root not in sys.path:
    sys.path.append(project_root)

from mapocr_toolkit.utils.data_loader import load_raw_data_paths_and_labels, create_class_maps
from mapocr_toolkit.utils.cnn_preprocessor import prepare_cnn_data
from mapocr_toolkit.cnn.cnn_model import create_cnn_model

EPOCHS = 10
BATCH_SIZE = 32 

def main():
    print("[INFO] Starting CNN training process")
    print("[INFO] Loading data paths and raw labels...")

    data_items, raw_unique_labels = load_raw_data_paths_and_labels()

    if not data_items:
        print("[ERROR] No data items loaded")
        return
    
    print(f"[INFO] Loaded {len(data_items)} data items")
    print(f"[INFO] Found {len(raw_unique_labels)} unique labels: {raw_unique_labels}")

    class_to_int, int_to_class = create_class_maps(raw_unique_labels)
    num_classes = len(class_to_int)
    print("[INFO] Created class mappings")
    print(f"[INFO] Number of classes: {num_classes}")

    print("[INFO] Preparing CNN data...")
    cnn_data_result = prepare_cnn_data(data_items, class_to_int) 
    
    if cnn_data_result is None or not cnn_data_result[0] or cnn_data_result[0][0].size == 0:
        print("[ERROR] Failed to prepare CNN data or no data available after preprocessing")
        return

    (x_train, y_train), (x_val, y_val), processing_info = cnn_data_result
    
    print(f"[INFO] Training data shape: X={x_train.shape}, y={y_train.shape}")
    print(f"[INFO] Validation data shape: X={x_val.shape}, y={y_val.shape}")

    input_shape = x_train.shape[1:]
    print(f"[INFO] Determined input shape for CNN model: {input_shape}")

    print("[INFO] Creating CNN model...")
    model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    print("[INFO] Training the model...")
    history = model.fit(x_train, y_train,
                        validation_data=(x_val, y_val),
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
    print("[INFO] Training finished")

    if history.history.get('val_accuracy'):
        final_val_accuracy = history.history['val_accuracy'][-1]
        print(f"[INFO] Final validation accuracy: {final_val_accuracy:.4f}")
    
    # TODO: Добавить сохранение модели и процессинг инфо


if __name__ == '__main__':
    main()