"""
scripts/train_crnn.py  -  Issue #13

обучение CRNN-классификатора в два этапа:
  этап 1 - CNN заморожена, обучаются только BiLSTM + Dense (быстрый старт)
  этап 2 - верхние Conv-блоки размораживаются, дообучение с малым lr

сравнение строго на тех же 703 валидационных примерах,
что и у CNN (Issue #9), RNN (Issue #10) и ансамбля (Issue #11):
  val_split=0.35, random_state=42

запуск:
    python scripts/train_crnn.py
    python scripts/train_crnn.py --skip-transfer   # без переноса весов CNN
    python scripts/train_crnn.py --epochs1 5 --epochs2 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from mapocr_toolkit.utils.data_loader import load_raw_data_paths_and_labels, create_class_maps
from mapocr_toolkit.utils.cnn_preprocessor import prepare_cnn_data

# пути к моделям
CNN_MODEL_PATH = os.path.join(project_root, 'models', 'demo', 'cnn', 'cnn_model.keras')
SAVE_DIR       = os.path.join(project_root, 'models', 'demo', 'crnn')
CKPT_PATH      = os.path.join(SAVE_DIR, 'crnn_best.keras')

# те же настройки что у CNN и RNN - чтобы val сет совпадал
VAL_SPLIT    = 0.35
RANDOM_STATE = 42
BATCH_SIZE   = 32

# этап 1: CNN заморожена, учим только BiLSTM + Dense
EPOCHS_PHASE1 = 10
LR_PHASE1     = 1e-3

# этап 2: размораживаем верхние слои CNN и дообучаем вместе
EPOCHS_PHASE2 = 15
LR_PHASE2     = 1e-4  # в 10 раз меньше чем на этапе 1, чтобы не сломать перенесённые веса

# хотим раздуть city_major до 70 примеров через аугментацию, как в Issue #9
CITY_MAJOR_TARGET = 70


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Обучение CRNN-классификатора (Issue #13)')
    p.add_argument('--skip-transfer', action='store_true',
                   help='Не переносить веса из cnn_model.keras')
    p.add_argument('--epochs1', type=int, default=EPOCHS_PHASE1,
                   help=f'Эпох на этапе 1 (default: {EPOCHS_PHASE1})')
    p.add_argument('--epochs2', type=int, default=EPOCHS_PHASE2,
                   help=f'Эпох на этапе 2 (default: {EPOCHS_PHASE2})')
    return p.parse_args()


def augment_city_major(x_train, y_train, city_major_idx, target=CITY_MAJOR_TARGET):
    """раздуваем city_major до target примеров - та же стратегия что в Issue #9."""
    mask = (np.argmax(y_train, axis=1) == city_major_idx)
    x_cm = x_train[mask]
    y_cm = y_train[mask]

    current = len(x_cm)
    print(f'[INFO] city_major в train до аугментации: {current}')

    if current >= target:
        print('[INFO] аугментация не нужна')
        return x_train, y_train

    aug = ImageDataGenerator(
        rotation_range=5,            # ±5° - текст на картах иногда чуть наклонён
        width_shift_range=0.08,      # ±8% по горизонтали - имитируем смещение рамки кропа
        height_shift_range=0.05,     # ±5% по вертикали
        brightness_range=[0.8, 1.2], # ±20% яркость - разное качество сканирования
        fill_mode='reflect',
        # horizontal_flip намеренно выключен - текст станет зеркальным
    )

    needed = target - current
    it = aug.flow(x_cm, y_cm, batch_size=1, shuffle=True)
    aug_imgs, aug_lbls = [], []
    for _ in range(needed):
        img_b, lbl_b = next(it)
        aug_imgs.append(img_b[0])
        aug_lbls.append(lbl_b[0])

    x_aug = np.array(aug_imgs)
    y_aug = np.array(aug_lbls)

    x_out = np.concatenate([x_train, x_aug], axis=0)
    y_out = np.concatenate([y_train, y_aug], axis=0)

    # перемешиваем, чтобы аугментированные не шли пачкой в конце
    idx = np.random.permutation(len(x_out))
    print(f'[INFO] city_major после аугментации: {target}')
    return x_out[idx], y_out[idx]


def compute_weights(y_int):
    weights = compute_class_weight('balanced', classes=np.unique(y_int), y=y_int)
    return dict(enumerate(weights))


def save_confusion_matrix(y_true, y_pred, class_names, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap='Blues')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[INFO] Confusion matrix → {path}')


def main():
    args = parse_args()

    # импортируем здесь, чтобы TF не грузился раньше времени при --help
    from mapocr_toolkit.crnn.crnn_model import (
        create_crnn_model,
        freeze_cnn_backbone,
        transfer_weights_from_cnn,
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # загружаем данные
    print('[INFO] загрузка данных...')
    data_items, raw_labels = load_raw_data_paths_and_labels()
    if not data_items:
        print('[ERROR] нет данных')
        return

    class_to_int, int_to_class = create_class_maps(raw_labels)
    num_classes = len(class_to_int)
    class_names = [int_to_class[i] for i in range(num_classes)]
    print(f'[INFO] {len(data_items)} примеров, {num_classes} классов: {class_names}')

    (x_train, y_train), (x_val, y_val), proc_info = prepare_cnn_data(
        data_items, class_to_int,
        val_split_size=VAL_SPLIT,
        random_state_value=RANDOM_STATE,
    )
    print(f'[INFO] train: {x_train.shape}, val: {x_val.shape}')

    # city_major всего 9 реальных примеров, без аугментации модель его просто не увидит
    city_major_idx = class_to_int.get('city_major', -1)
    if city_major_idx >= 0:
        x_train, y_train = augment_city_major(x_train, y_train, city_major_idx)

    # пересчитываем веса после аугментации
    y_train_int = np.argmax(y_train, axis=1)
    class_weight_dict = compute_weights(y_train_int)
    print(f'[INFO] веса классов: {class_weight_dict}')

    y_val_int = np.argmax(y_val, axis=1)
    input_shape = x_train.shape[1:]

    # создаём модель
    print('\n[INFO] создание CRNN модели...')
    model = create_crnn_model(input_shape=input_shape, num_classes=num_classes)
    model.summary()

    # переносим веса conv1 и conv2 из уже обученной CNN 
    if not args.skip_transfer and os.path.exists(CNN_MODEL_PATH):
        print(f'\n[INFO] перенос весов из {CNN_MODEL_PATH}')
        transfer_weights_from_cnn(model, CNN_MODEL_PATH)
    else:
        if args.skip_transfer:
            print('[INFO] --skip-transfer: обучаем с нуля')
        else:
            print(f'[WARN] cnn_model.keras не найдена по пути {CNN_MODEL_PATH}, обучаем с нуля')

    # CNN заморожена, обучаем только BiLSTM + Dense
    print(f'\n{"="*60}')
    print(f'  этап 1 - CNN заморожена, обучаем BiLSTM + Dense')
    print(f'  эпох: {args.epochs1}, lr: {LR_PHASE1}')
    print(f'{"="*60}')

    freeze_cnn_backbone(model, freeze=True)
    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE1),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_phase1 = [
        EarlyStopping(monitor='val_loss', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-6),
        ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    history1 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs1,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase1,
    )

    # размораживаем верхние Conv-блоки (conv3, conv4, conv5)
    print(f'\n{"="*60}')
    print(f'  этап 2 - частичная разморозка CNN, дообучение')
    print(f'  эпох: {args.epochs2}, lr: {LR_PHASE2}')
    print(f'{"="*60}')

    # сначала размораживаем всё
    freeze_cnn_backbone(model, freeze=False)
    # потом снова замораживаем conv1 и conv2 - не трогаем перенесённые веса
    for name in ['conv1', 'conv2', 'pool1', 'pool2']:
        try:
            model.get_layer(name).trainable = False
        except ValueError:
            pass

    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE2),  # маленький lr - не ломаем то, что уже хорошо работает
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_loss', patience=7,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, verbose=1, min_lr=1e-7),
        ModelCheckpoint(CKPT_PATH, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    history2 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs2,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks_phase2,
    )

    # смотрим что получилось
    print('\n[INFO] оценка на валидации (703 примера)...')
    y_pred_probs = model.predict(x_val, verbose=0)
    y_pred_int   = np.argmax(y_pred_probs, axis=1)

    print('\n[RESULTS] classification report:')
    print(classification_report(y_val_int, y_pred_int,
                                 target_names=class_names, zero_division=0))

    # сохраняем модель
    model_path = os.path.join(SAVE_DIR, 'crnn_model.keras')
    model.save(model_path)
    print(f'[INFO] модель сохранена → {model_path}')

    save_confusion_matrix(
        y_val_int, y_pred_int, class_names,
        title=f'CRNN Confusion Matrix (val={len(y_val_int)}, Issue #13)',
        path=os.path.join(SAVE_DIR, 'confusion_matrix.png'),
    )

    # тот же формат что у CNN чтобы visualize_map мог подхватить CRNN без изменений
    proc_info['class_to_int'] = class_to_int
    proc_info['int_to_class'] = int_to_class
    info_path = os.path.join(SAVE_DIR, 'crnn_processing_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(proc_info, f, indent=4, ensure_ascii=False)
    print(f'[INFO] processing info → {info_path}')

    # итоговая таблица для crnn_results_issue13.md
    print(f'\n{"="*60}')
    print('  итог - для crnn_results_issue13.md')
    print(f'{"="*60}')

    from sklearn.metrics import accuracy_score, f1_score
    acc     = accuracy_score(y_val_int, y_pred_int)
    macro   = f1_score(y_val_int, y_pred_int, average='macro', zero_division=0)
    per_cls = f1_score(y_val_int, y_pred_int, average=None, zero_division=0)

    reference = {
        'CNN (Issue #9)':          {'acc': 0.851, 'macro': 0.58},
        'RNN (Issue #10)':         {'acc': 0.862, 'macro': 0.35},
        'Ансамбль Soft Voting':    {'acc': 0.905, 'macro': 0.62},
        'CRNN (Issue #13)':        {'acc': acc,   'macro': macro},
    }

    print(f'{"модель":<28} {"accuracy":>10} {"macro f1":>10}')
    print('-' * 52)
    for name, vals in reference.items():
        marker = ' ←' if name == 'CRNN (Issue #13)' else ''
        print(f'{name:<28} {vals["acc"]:>10.3f} {vals["macro"]:>10.3f}{marker}')

    # labels=list(range(num_classes)) гарантирует что per_cls[i] всегда
    # соответствует классу с индексом i, даже если класс не встретился в y_val
    from sklearn.metrics import f1_score as sk_f1_score
    per_cls_fixed = sk_f1_score(
        y_val_int, y_pred_int,
        labels=list(range(num_classes)),
        average=None,
        zero_division=0,
    )
    print(f'\nper-class f1 (CRNN):')
    for i, cls in enumerate(class_names):
        f1 = float(per_cls_fixed[i])
        print(f'  {cls:<14} f1={f1:.2f}')

    print(f'\n[INFO] готово. результаты: {SAVE_DIR}')


if __name__ == '__main__':
    main()