import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI, работает на CPU без дисплея
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mapocr_toolkit.utils.data_loader import load_raw_data_paths_and_labels, create_class_maps
from mapocr_toolkit.utils.cnn_preprocessor import prepare_cnn_data
from mapocr_toolkit.cnn.cnn_model import create_cnn_model

EPOCHS = 10
BATCH_SIZE = 32

# сколько примеров city_major хотим иметь после аугментации
CITY_MAJOR_TARGET = 70


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

    # датасет несбалансирован: settlement 82.6%, city_major 0.4%
    # без весов модель просто угадывает settlement всегда
    y_train_int = np.argmax(y_train, axis=1)  # из one-hot в целые числа для compute_class_weight

    weights = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    class_weight_dict = dict(enumerate(weights))
    print(f"[INFO] Class weights (before augmentation): {class_weight_dict}")

    # искусственно расширяем
    city_major_class_idx = class_to_int['city_major']

    city_major_mask = (y_train_int == city_major_class_idx)
    x_city_major = x_train[city_major_mask]  # (9, 60, 200, 3)
    y_city_major = y_train[city_major_mask]  # (9, num_classes)

    print(f"[INFO] city_major samples in train before aug: {len(x_city_major)}")
    print(f"[INFO] augmenting city_major to {CITY_MAJOR_TARGET} samples...")

    aug_gen = ImageDataGenerator(
        rotation_range=5,            # ±5° - текст на картах слегка наклонён
        width_shift_range=0.08,      # ±8% горизонтальный сдвиг рамки кропа
        height_shift_range=0.05,     # ±5% вертикальный сдвиг
        brightness_range=[0.8, 1.2], # ±20% яркость - разные условия сканирования
        fill_mode='reflect',         # отражение лучше для текста, чем 'constant'
        # horizontal_flip намеренно не использую - текст станет зеркальным
    )

    augmented_images = []
    augmented_labels = []
    needed = CITY_MAJOR_TARGET - len(x_city_major)

    # генерит бесконечный поток батчей, нужное количество берём
    aug_iterator = aug_gen.flow(x_city_major, y_city_major, batch_size=1, shuffle=True)
    for _ in range(needed):
        img_batch, lbl_batch = next(aug_iterator)
        augmented_images.append(img_batch[0])
        augmented_labels.append(lbl_batch[0])

    x_aug = np.array(augmented_images)
    y_aug = np.array(augmented_labels)

    # дописываем аугментированные примеры в тренировочный сет
    x_train = np.concatenate([x_train, x_aug], axis=0)
    y_train = np.concatenate([y_train, y_aug], axis=0)

    # перетасовываем, чтобы аугментированные не шли пачкой в конце батча
    shuffle_idx = np.random.permutation(len(x_train))
    x_train = x_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    # пересчитываем веса - теперь city_major стало ~70, его вес должен упасть
    y_train_int_aug = np.argmax(y_train, axis=1)
    weights_aug = compute_class_weight('balanced', classes=np.unique(y_train_int_aug), y=y_train_int_aug)
    class_weight_dict = dict(enumerate(weights_aug))
    print(f"[INFO] x_train after augmentation: {x_train.shape}")
    print(f"[INFO] Updated class weights: {class_weight_dict}")

    input_shape = x_train.shape[1:]
    print(f"[INFO] Determined input shape for CNN model: {input_shape}")

    print("[INFO] Creating CNN model...")
    model = create_cnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    print("[INFO] Training the model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,  
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stopping],
    )
    print("[INFO] Training finished")

    print("[INFO] Evaluating model on validation set...")

    y_pred_probs = model.predict(x_val)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    y_val_int = np.argmax(y_val, axis=1)

    class_names = [int_to_class[i] for i in range(num_classes)]

    # текстовый отчёт в консоль
    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_val_int, y_pred_int, target_names=class_names, zero_division=0))

    print("[INFO] Saving model and processing_info...")

    save_dir = os.path.join(project_root, "models", "demo", "cnn")
    os.makedirs(save_dir, exist_ok=True)

    # confusion matrix как PNG, кладём рядом с моделькой
    cm = confusion_matrix(y_val_int, y_pred_int)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(f'CNN Confusion Matrix\n(val size: {len(y_val_int)}, dataset: 1808 examples)')
    plt.tight_layout()

    cm_save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_save_path}")

    model_save_path = os.path.join(save_dir, "cnn_model.keras")
    model.save(model_save_path)
    print(f"[INFO] CNN model saved to {model_save_path}")

    processing_info['class_to_int'] = class_to_int
    processing_info['int_to_class'] = int_to_class

    info_save_path = os.path.join(save_dir, "cnn_processing_info.json")
    with open(info_save_path, 'w') as f:
        json.dump(processing_info, f, indent=4)
    print(f"[INFO] CNN processing_info saved to {info_save_path}")


if __name__ == '__main__':
    main()