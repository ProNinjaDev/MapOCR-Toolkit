import os
import sys
import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mapocr_toolkit.utils.data_loader import load_raw_data_paths_and_labels, create_class_maps
from mapocr_toolkit.utils.rnn_preprocessor import prepare_rnn_data
from mapocr_toolkit.rnn.rnn_model import create_char_level_lstm_model


EPOCHS = 30
BATCH_SIZE = 8
PATIENCE_EARLY_STOPPING = 10
VAL_SPLIT = 0.35  # было 0.2, выравниваем с CNN для сопоставимости разбивок


def main():
    print("[INFO] Starting RNN training process")
    print("[INFO] Loading data paths and raw labels...")

    data_items, raw_unique_labels = load_raw_data_paths_and_labels()

    if not data_items:
        print("[ERROR] No data items loaded. Exiting.")
        return

    print(f"[INFO] Loaded {len(data_items)} data items")
    print(f"[INFO] Found {len(raw_unique_labels)} unique labels: {raw_unique_labels}")

    class_to_int, int_to_class = create_class_maps(raw_unique_labels)
    num_classes = len(class_to_int)
    print("[INFO] Created class mappings")
    print(f"[INFO] Number of classes: {num_classes}")

    print("[INFO] Preparing RNN data...")
    rnn_data_result = prepare_rnn_data(data_items, class_to_int, VAL_SPLIT, 42)

    if rnn_data_result is None:
        print("[ERROR] Failed to prepare RNN data. Exiting.")
        return

    (x_train, y_train), (x_val, y_val), processing_info = rnn_data_result

    if x_train.size == 0 or x_val.size == 0:
        print("[ERROR] Training or validation data is empty after preprocessing")
        return

    print(f"[INFO] Training data shape: X={x_train.shape}, y={y_train.shape}")
    print(f"[INFO] Validation data shape: X={x_val.shape}, y={y_val.shape}")

    max_seq_len = processing_info['max_seq_len']
    num_chars_vocab = processing_info['num_chars_vocab']
    print(f"[INFO] Max sequence length: {max_seq_len}")
    print(f"[INFO] Character vocabulary size (incl. padding): {num_chars_vocab}")

    # без весов LSTM игнорирует редкие классы
    y_train_int = np.argmax(y_train, axis=1)  # из one-hot в целые числа

    weights = compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
    class_weight_dict = dict(enumerate(weights))
    print(f"[INFO] Class weights: {class_weight_dict}")

    print("[INFO] Creating RNN model...")
    model = create_char_level_lstm_model(max_seq_len,
                                         num_chars_vocab,
                                         num_classes)
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=PATIENCE_EARLY_STOPPING,
                                   restore_best_weights=True,
                                   verbose=1)

    print("[INFO] Training the RNN model...")
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weight_dict,
    )
    print("[INFO] RNN training finished")

    if early_stopping.stopped_epoch > 0:
        print(f"[INFO] Early stopping triggered at epoch {early_stopping.stopped_epoch + 1}")
        print(f"[INFO] Best validation loss: {early_stopping.best:.4f}")

    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    print(f"[INFO] Final validation accuracy (best model): {accuracy:.4f}")
    print(f"[INFO] Final validation loss (best model): {loss:.4f}")

    print("[INFO] Evaluating model on validation set...")

    y_pred_probs = model.predict(x_val)
    y_pred_int = np.argmax(y_pred_probs, axis=1)
    y_val_int = np.argmax(y_val, axis=1)

    class_names = [int_to_class[i] for i in range(num_classes)]

    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_val_int, y_pred_int, target_names=class_names, zero_division=0))

    print("[INFO] Saving RNN model and processing_info...")
    save_dir = os.path.join(project_root, "models", "demo", "rnn")
    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(y_val_int, y_pred_int)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(f'RNN Confusion Matrix\n(val size: {len(y_val_int)}, dataset: 1808 examples)')
    plt.tight_layout()

    cm_save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved to {cm_save_path}")

    model_save_path = os.path.join(save_dir, "rnn_model.keras")
    model.save(model_save_path)
    print(f"[INFO] RNN model saved to {model_save_path}")

    processing_info['class_to_int'] = class_to_int
    processing_info['int_to_class'] = int_to_class

    info_save_path = os.path.join(save_dir, "rnn_processing_info.json")
    with open(info_save_path, 'w') as f:
        serializable_processing_info = {}
        for key, value in processing_info.items():
            if isinstance(value, dict):
                serializable_processing_info[key] = {
                    k: (int(v) if isinstance(v, np.int64) else v) for k, v in value.items()
                }
            elif isinstance(value, np.int64):
                serializable_processing_info[key] = int(value)
            else:
                serializable_processing_info[key] = value

        json.dump(serializable_processing_info, f, indent=4)
    print(f"[INFO] RNN processing_info saved to {info_save_path}")


if __name__ == '__main__':
    main()