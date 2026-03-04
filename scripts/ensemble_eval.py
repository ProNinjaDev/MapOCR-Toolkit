"""
CNN + RNN

загружает обе обученные модели, прогоняет их на одном валидационном сете
(тот же random_state=42 и val_split=0.35, что и в train_cnn и в train_rnn)
и сравнивает три стратегии ансамблирования с отдельными моделями.

Запуск:
    python scripts/ensemble_eval.py --strategy soft
    python scripts/ensemble_eval.py --strategy weighted --cnn-weight 0.7
    python scripts/ensemble_eval.py --strategy max_confidence
    python scripts/ensemble_eval.py --strategy all
"""

import os
import sys
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mapocr_toolkit.utils.data_loader import load_raw_data_paths_and_labels, create_class_maps
from mapocr_toolkit.utils.cnn_preprocessor import prepare_cnn_data
from mapocr_toolkit.utils.rnn_preprocessor import prepare_rnn_data

# пути к артефактам обученных моделек
CNN_MODEL_PATH  = os.path.join(project_root, 'models', 'demo', 'cnn', 'cnn_model.keras')
RNN_MODEL_PATH  = os.path.join(project_root, 'models', 'demo', 'rnn', 'rnn_model.keras')
ENSEMBLE_DIR    = os.path.join(project_root, 'models', 'demo', 'ensemble')

# должны совпадать с train_cnn, train_rnn, иначе val сеты разъедутся
VAL_SPLIT    = 0.35
RANDOM_STATE = 42

def soft_voting(p_cnn: np.ndarray, p_rnn: np.ndarray) -> np.ndarray:
    """
    Обе модели выдают вектор вероятностей длиной N_классов.
    Складываем и делим на 2 — победитель тот класс, у которого
    суммарная вероятность выше.
    """
    return (p_cnn + p_rnn) / 2.0


def weighted_voting(p_cnn: np.ndarray, p_rnn: np.ndarray,
                    cnn_weight: float) -> np.ndarray:
    """
    Взвешенное среднее CNN получает больший вес, потому что
    у неё macro F1 = 0.58 против 0.35 у RNN
    """
    rnn_weight = 1.0 - cnn_weight
    return cnn_weight * p_cnn + rnn_weight * p_rnn


def max_confidence(p_cnn: np.ndarray, p_rnn: np.ndarray) -> np.ndarray:
    """
    Для каждого примера побеждает та модель, у которой
    максимальная вероятность выше
    """
    # np.max по axis=1 — максимум по классам для каждого примера
    cnn_conf = np.max(p_cnn, axis=1)  # shape: (N,)
    rnn_conf = np.max(p_rnn, axis=1)  # shape: (N,)

    # для каждого примера выбираем строку из той модели, где уверенность выше
    cnn_wins = (cnn_conf >= rnn_conf)

    result = np.where(
        cnn_wins[:, np.newaxis],  # нужна ось чтобы broadcasting сработал по классам
        p_cnn,
        p_rnn
    )
    return result

def print_report(name: str, y_true: np.ndarray, y_pred: np.ndarray,
                 class_names: list) -> dict:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    print(classification_report(
        y_true, y_pred,
        target_names=class_names,
        zero_division=0,
    ))
    return report


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           class_names: list, title: str,
                           save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion matrix saved → {save_path}")


def print_comparison_table(results: dict) -> None:
    print(f"\n{'='*60}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'='*60}")
    header = f"{'Стратегия':<22} {'Accuracy':>10} {'Macro F1':>10}"
    print(header)
    print('-' * len(header))
    for name, report in results.items():
        acc   = report['accuracy']
        macro = report['macro avg']['f1-score']
        print(f"{name:<22} {acc:>10.4f} {macro:>10.4f}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Ансамбль CNN+RNN')
    parser.add_argument(
        '--strategy',
        choices=['soft', 'weighted', 'max_confidence', 'all'],
        default='all',
        help='Стратегия ансамблирования (default: all — сравнить все)',
    )
    parser.add_argument(
        '--cnn-weight',
        type=float,
        default=0.65,
        help='Вес CNN при weighted voting (default: 0.65). RNN получит 1 - cnn_weight.',
    )
    args = parser.parse_args()

    os.makedirs(ENSEMBLE_DIR, exist_ok=True)

    print("[INFO] Loading dataset...")
    data_items, raw_unique_labels = load_raw_data_paths_and_labels()
    if not data_items:
        print("[ERROR] No data items loaded.")
        return

    class_to_int, int_to_class = create_class_maps(raw_unique_labels)
    num_classes = len(class_to_int)
    class_names = [int_to_class[i] for i in range(num_classes)]
    print(f"[INFO] {len(data_items)} items, {num_classes} classes: {class_names}")

    print("[INFO] Preparing CNN data (val_split=0.35, random_state=42)...")
    (_, _), (x_val_cnn, y_val_cnn), _ = prepare_cnn_data(
        data_items, class_to_int,
        val_split_size=VAL_SPLIT,
        random_state_value=RANDOM_STATE,
    )
    y_val_int = np.argmax(y_val_cnn, axis=1)  # истинные метки — одинаковы для обеих моделей
    print(f"[INFO] CNN val shape: {x_val_cnn.shape}")

    print("[INFO] Preparing RNN data (val_split=0.35, random_state=42)...")
    (_, _), (x_val_rnn, y_val_rnn), _ = prepare_rnn_data(
        data_items, class_to_int,
        val_split_size=VAL_SPLIT,
        random_state_value=RANDOM_STATE,
    )
    print(f"[INFO] RNN val shape: {x_val_rnn.shape}")

    # убеждаемся что val сеты одного размера
    assert len(y_val_int) == np.argmax(y_val_rnn, axis=1).shape[0], \
        "Val sets have different sizes! Check VAL_SPLIT and RANDOM_STATE."

    print(f"[INFO] Loading CNN model from {CNN_MODEL_PATH}...")
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

    print(f"[INFO] Loading RNN model from {RNN_MODEL_PATH}...")
    rnn_model = tf.keras.models.load_model(RNN_MODEL_PATH)

    print("[INFO] Running inference...")
    p_cnn = cnn_model.predict(x_val_cnn, verbose=0)
    p_rnn = rnn_model.predict(x_val_rnn, verbose=0)

    y_pred_cnn = np.argmax(p_cnn, axis=1)
    y_pred_rnn = np.argmax(p_rnn, axis=1)

    all_reports = {}

    # считаем CNN и RNN отдельно
    all_reports['CNN'] = print_report('CNN (одна модель)', y_val_int, y_pred_cnn, class_names)
    save_confusion_matrix(
        y_val_int, y_pred_cnn, class_names,
        'CNN Confusion Matrix',
        os.path.join(ENSEMBLE_DIR, 'cm_cnn.png'),
    )

    all_reports['RNN'] = print_report('RNN (одна модель)', y_val_int, y_pred_rnn, class_names)
    save_confusion_matrix(
        y_val_int, y_pred_rnn, class_names,
        'RNN Confusion Matrix',
        os.path.join(ENSEMBLE_DIR, 'cm_rnn.png'),
    )

    strategies_to_run = (
        ['soft', 'weighted', 'max_confidence']
        if args.strategy == 'all'
        else [args.strategy]
    )

    for strategy in strategies_to_run:
        if strategy == 'soft':
            p_ensemble = soft_voting(p_cnn, p_rnn)
            label = 'Soft Voting (avg)'

        elif strategy == 'weighted':
            p_ensemble = weighted_voting(p_cnn, p_rnn, args.cnn_weight)
            label = f'Weighted (CNN={args.cnn_weight:.2f})'

        elif strategy == 'max_confidence':
            p_ensemble = max_confidence(p_cnn, p_rnn)
            label = 'Max Confidence'

        y_pred_ensemble = np.argmax(p_ensemble, axis=1)
        all_reports[label] = print_report(label, y_val_int, y_pred_ensemble, class_names)
        save_confusion_matrix(
            y_val_int, y_pred_ensemble, class_names,
            f'Ensemble: {label}',
            os.path.join(ENSEMBLE_DIR, f'cm_{strategy}.png'),
        )

    print_comparison_table(all_reports)


if __name__ == '__main__':
    main()