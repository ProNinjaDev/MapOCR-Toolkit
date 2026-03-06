#!/usr/bin/env python3
"""
scripts/visualize_map.py  —  Issue #12
Визуализирует предсказания ансамбля CNN+RNN поверх исходной карты.

Алгоритм:
  1. Читает dataset_LABELED.csv, фильтрует строки по --map (имя TIF-файла).
  2. Для каждой строки запускает CNN-инференс на кропе и RNN-инференс на OCR-тексте.
  3. Объединяет результаты взвешенным ансамблем: p = cnn_w * p_cnn + (1-cnn_w) * p_rnn.
  4. Загружает исходный TIF и уменьшает его масштаб (--scale).
  5. Строит интерактивный Plotly-HTML: карта + цветные боксы + hover с предсказанным классом.

Запуск из корня репозитория:
    python scripts/visualize_map.py --map img20250920_20532456.tif
    python scripts/visualize_map.py --map img20250920_20532456.tif \\
        --scale 0.12 --cnn-weight 0.65 --output outputs/map_annotated.html

Зависимости (добавить в requirements.txt):
    plotly
    Pillow  (обычно уже стоит с tensorflow)
    tqdm
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm

# ── Путь до корня проекта (скрипт лежит в scripts/) ──────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

# ── Пути к артефактам обученных моделей ───────────────────────────────────────
CNN_MODEL_PATH  = PROJECT_ROOT / 'models' / 'demo' / 'cnn' / 'cnn_model.keras'
RNN_MODEL_PATH  = PROJECT_ROOT / 'models' / 'demo' / 'rnn' / 'rnn_model.keras'
CNN_INFO_PATH   = PROJECT_ROOT / 'models' / 'demo' / 'cnn' / 'cnn_processing_info.json'
RNN_INFO_PATH   = PROJECT_ROOT / 'models' / 'demo' / 'rnn' / 'rnn_processing_info.json'

# ── Пути к данным ─────────────────────────────────────────────────────────────
LABELS_CSV  = PROJECT_ROOT / 'data' / 'dataset_LABELED.csv'
CROPS_DIR   = PROJECT_ROOT / 'data' / 'dataset_crops_paddle'
TIFS_DIR    = PROJECT_ROOT / 'data' / 'raw_tifs'

# ── Цветовая схема классов (из label_tool.py для согласованности) ─────────────
CLASS_COLORS: dict[str, str] = {
    'city_major': '#e74c3c',   # красный
    'city':       '#e67e22',   # оранжевый
    'settlement': '#27ae60',   # зелёный
    'hydro':      '#3498db',   # синий
    'region':     '#9b59b6',   # фиолетовый
    'other':      '#7f8c8d',   # серый
}
DEFAULT_COLOR = '#cccccc'

# ── CNN input size (должен совпадать с train_cnn.py) ─────────────────────────
CNN_TARGET_H = 60
CNN_TARGET_W = 200


# ─────────────────────────────────────────────────────────────────────────────
#  Вспомогательные функции
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv(csv_path: Path):
    """Читает CSV с поддержкой разделителей ';' и ',' и BOM."""
    import pandas as pd
    for sep in (';', ','):
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig')
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    raise ValueError(f'Не удалось прочитать CSV: {csv_path}')


def _parse_global_box(raw: str) -> Optional[tuple[int, int, int, int]]:
    """
    Разбирает строку вида '[[x1, y1], [x2, y2]]' → (x1, y1, x2, y2).
    Возвращает None при любой ошибке разбора.
    """
    try:
        pts = ast.literal_eval(raw)          # безопаснее, чем eval()
        x1, y1 = int(pts[0][0]), int(pts[0][1])
        x2, y2 = int(pts[1][0]), int(pts[1][1])
        return x1, y1, x2, y2
    except Exception:
        return None


def _encode_text_for_rnn(
    text: str,
    char_to_int: dict[str, int],
    max_seq_len: int,
    num_chars: int,
) -> np.ndarray:
    """
    Кодирует одну строку текста для RNN-инференса.
    Воспроизводит логику rnn_preprocessor.py:
      - Неизвестные символы → pad_index (0, символ '\0').
      - Последовательность обрезается/дополняется до max_seq_len.
      - Результат: one-hot массив shape (1, max_seq_len, num_chars).
    """
    pad_idx = char_to_int.get('\0', 0)
    encoded = [char_to_int.get(c, pad_idx) for c in str(text).strip()]

    # Усечение или дополнение нулями
    if len(encoded) >= max_seq_len:
        encoded = encoded[:max_seq_len]
    else:
        encoded += [pad_idx] * (max_seq_len - len(encoded))

    # One-hot кодирование
    x = np.zeros((1, max_seq_len, num_chars), dtype='float32')
    for t, idx in enumerate(encoded):
        if 0 <= idx < num_chars:
            x[0, t, idx] = 1.0
    return x


def _load_crop_for_cnn(image_path: Path) -> Optional[np.ndarray]:
    """
    Загружает кроп-изображение и приводит его к формату CNN:
    resize → (CNN_TARGET_H, CNN_TARGET_W), нормализация /255.
    Возвращает массив shape (CNN_TARGET_H, CNN_TARGET_W, 3) или None при ошибке.
    """
    try:
        img = load_img(str(image_path), target_size=(CNN_TARGET_H, CNN_TARGET_W))
        arr = img_to_array(img) / 255.0
        return arr.astype('float32')
    except Exception as e:
        print(f'[WARNING] Не удалось загрузить кроп {image_path.name}: {e}')
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Ансамбль: одна функция — весь inference для всего датасета
# ─────────────────────────────────────────────────────────────────────────────

def run_ensemble_inference(
    records: list[dict],
    cnn_model,
    rnn_model,
    cnn_info: dict,
    rnn_info: dict,
    cnn_weight: float = 0.65,
    batch_size: int = 64,
) -> list[str]:
    """
    Запускает CNN и RNN на всех записях пакетно, возвращает список
    предсказанных классов (по одному на запись).

    Пакетный инференс в ~10-20x быстрее, чем поэлементный, поэтому
    сначала собираем все данные, потом делаем один model.predict().
    """
    int_to_class: dict[int, str] = {
        int(k): v for k, v in cnn_info['int_to_class'].items()
    }
    char_to_int: dict[str, int] = rnn_info['char_to_int_map']
    max_seq_len: int             = int(rnn_info['max_seq_len'])
    num_chars: int               = int(rnn_info['num_chars_vocab'])

    n = len(records)

    # ── 1. Подготовка данных для CNN ─────────────────────────────────────────
    print('[INFO] Загрузка кропов для CNN...')
    cnn_inputs: list[Optional[np.ndarray]] = []
    for rec in tqdm(records, desc='  кропы'):
        crop_path = CROPS_DIR / rec['filename']
        cnn_inputs.append(_load_crop_for_cnn(crop_path))

    # ── 2. Подготовка данных для RNN ─────────────────────────────────────────
    print('[INFO] Кодирование текстов для RNN...')
    rnn_batch = np.zeros((n, max_seq_len, num_chars), dtype='float32')
    for i, rec in enumerate(records):
        encoded = _encode_text_for_rnn(rec['ocr_text'], char_to_int, max_seq_len, num_chars)
        rnn_batch[i] = encoded[0]

    # ── 3. CNN inference (пакетами, пропуская пустые кропы) ──────────────────
    print('[INFO] CNN inference...')
    num_classes = len(int_to_class)
    p_cnn = np.full((n, num_classes), 1.0 / num_classes, dtype='float32')  # prior по умолчанию

    # Собираем только те индексы, у которых есть кроп
    valid_idx = [i for i, x in enumerate(cnn_inputs) if x is not None]
    if valid_idx:
        cnn_valid = np.stack([cnn_inputs[i] for i in valid_idx])
        preds = cnn_model.predict(cnn_valid, batch_size=batch_size, verbose=0)
        for out_i, src_i in enumerate(valid_idx):
            p_cnn[src_i] = preds[out_i]

    missing_crops = n - len(valid_idx)
    if missing_crops:
        print(f'[WARNING] {missing_crops} кропов не найдено — для них CNN использует равномерный prior.')

    # ── 4. RNN inference ─────────────────────────────────────────────────────
    print('[INFO] RNN inference...')
    p_rnn = rnn_model.predict(rnn_batch, batch_size=batch_size, verbose=0)

    # ── 5. Взвешенный ансамбль ───────────────────────────────────────────────
    rnn_weight = 1.0 - cnn_weight
    p_ensemble = cnn_weight * p_cnn + rnn_weight * p_rnn  # (n, num_classes)

    predicted_indices = np.argmax(p_ensemble, axis=1)
    predicted_classes = [int_to_class.get(int(idx), 'unknown') for idx in predicted_indices]

    return predicted_classes


# ─────────────────────────────────────────────────────────────────────────────
#  Визуализация: Plotly HTML
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_figure(
    img_array: np.ndarray,
    annotated_records: list[dict],
    scale: float,
    map_name: str,
) -> go.Figure:
    """
    Строит Plotly-фигуру: фоновое изображение карты + цветные боксы + hover.

    Архитектура трейсов (per class):
      - Trace A: go.Scatter mode='lines'  — видимые прямоугольные контуры боксов.
                 Все боксы одного класса объединены None-разделителями в один трейс
                 (эффективнее, чем по одному трейсу на бокс).
      - Trace B: go.Scatter mode='markers' invisible — точки в центрах боксов
                 с hovertext = класс. Marker opacity=0 делает их невидимыми,
                 но Plotly всё равно регистрирует hover при наведении.
      Оба трейса связаны legendgroup, поэтому клик в легенде скрывает оба.
    """
    h_scaled, w_scaled = img_array.shape[:2]

    fig = go.Figure()

    # ── Фоновое изображение ───────────────────────────────────────────────────
    # go.Image задаёт систему координат: (0,0) — верхний левый угол,
    # x растёт вправо, y — вниз (совпадает с пиксельными координатами TIF).
    fig.add_trace(go.Image(
        z=img_array,
        hoverinfo='skip',
        name='карта',
    ))

    # ── Группировка боксов по классу ─────────────────────────────────────────
    boxes_by_class: dict[str, list[dict]] = {}
    for rec in annotated_records:
        cls = rec['predicted_class']
        boxes_by_class.setdefault(cls, []).append(rec)

    # ── Трейсы боксов ─────────────────────────────────────────────────────────
    for cls_name, cls_records in sorted(boxes_by_class.items()):
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        # --- Trace A: контуры прямоугольников ---
        # Каждый прямоугольник = 5 точек (замкнутый контур) + None-разделитель.
        outline_x: list = []
        outline_y: list = []

        # --- Trace B: невидимые центры для hover ---
        center_x: list[float] = []
        center_y: list[float] = []
        hover_texts: list[str] = []

        for rec in cls_records:
            x1s = rec['x1'] * scale
            y1s = rec['y1'] * scale
            x2s = rec['x2'] * scale
            y2s = rec['y2'] * scale

            # Контур прямоугольника (по часовой стрелке)
            outline_x += [x1s, x2s, x2s, x1s, x1s, None]
            outline_y += [y1s, y1s, y2s, y2s, y1s, None]

            # Центр бокса
            center_x.append((x1s + x2s) / 2)
            center_y.append((y1s + y2s) / 2)
            hover_texts.append(f'<b>{cls_name}</b>')

        # Trace A — видимые линии
        fig.add_trace(go.Scatter(
            x=outline_x,
            y=outline_y,
            mode='lines',
            line=dict(color=color, width=2),
            name=cls_name,
            legendgroup=cls_name,
            hoverinfo='skip',       # hover только через Trace B
            showlegend=True,
        ))

        # Trace B — невидимые маркеры для hover
        # size=16 даёт комфортную зону наведения, opacity=0 скрывает маркер
        fig.add_trace(go.Scatter(
            x=center_x,
            y=center_y,
            mode='markers',
            marker=dict(size=16, color=color, opacity=0),
            name=cls_name,
            legendgroup=cls_name,
            showlegend=False,
            text=hover_texts,
            hoverinfo='text',
        ))

    # ── Легенда и оформление ─────────────────────────────────────────────────
    n_boxes = len(annotated_records)
    fig.update_layout(
        title=dict(
            text=f'<b>MapOCR — предсказания ансамбля CNN+RNN</b>'
                 f'<br><sup>Карта: {map_name} · Боксов: {n_boxes} · '
                 f'Масштаб: {scale:.0%}</sup>',
            x=0.5,
            font=dict(size=15),
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[0, w_scaled],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            range=[h_scaled, 0],   # y идёт сверху вниз (как в изображении)
            scaleanchor='x',       # сохраняем пропорции карты
        ),
        legend=dict(
            title='Класс (клик — скрыть)',
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#cccccc',
            borderwidth=1,
        ),
        margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0'),
        hovermode='closest',
        dragmode='zoom',           # зажать мышь = зум; двойной клик = сброс
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Визуализация предсказаний ансамбля CNN+RNN на исходной карте (Issue #12)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--map', required=True,
        help='Имя TIF-файла (например: img20250920_20532456.tif)',
    )
    parser.add_argument(
        '--scale', type=float, default=0.15,
        help='Коэффициент уменьшения карты (0.15 = 15%% от оригинала)',
    )
    parser.add_argument(
        '--cnn-weight', type=float, default=0.65,
        help='Вес CNN в ансамбле. Вес RNN = 1 - cnn_weight.',
    )
    parser.add_argument(
        '--output', type=Path,
        default=PROJECT_ROOT / 'outputs' / 'map_annotated.html',
        help='Путь для сохранения HTML-файла.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=64,
        help='Размер батча при инференсе моделей.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_name: str = args.map

    # ── Проверка входных файлов ───────────────────────────────────────────────
    tif_path = TIFS_DIR / map_name
    if not tif_path.exists():
        print(f'[ERROR] TIF не найден: {tif_path}')
        sys.exit(1)

    for path, label in [
        (LABELS_CSV,     'dataset_LABELED.csv'),
        (CNN_MODEL_PATH, 'cnn_model.keras'),
        (RNN_MODEL_PATH, 'rnn_model.keras'),
        (CNN_INFO_PATH,  'cnn_processing_info.json'),
        (RNN_INFO_PATH,  'rnn_processing_info.json'),
    ]:
        if not path.exists():
            print(f'[ERROR] Файл не найден: {path} ({label})')
            sys.exit(1)

    # ── Загрузка CSV и фильтрация по карте ───────────────────────────────────
    print(f'[INFO] Загрузка {LABELS_CSV.name}...')
    df = _read_csv(LABELS_CSV)

    required_cols = {'filename', 'ocr_text', 'global_box', 'source_map'}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        print(f'[ERROR] В CSV нет колонок: {sorted(missing_cols)}')
        sys.exit(1)

    df_map = df[df['source_map'] == map_name].copy()
    if df_map.empty:
        print(f'[ERROR] В CSV нет записей для карты "{map_name}".')
        print(f'        Доступные карты: {sorted(df["source_map"].unique())}')
        sys.exit(1)

    print(f'[INFO] Найдено {len(df_map)} записей для карты "{map_name}".')

    # ── Разбор global_box ────────────────────────────────────────────────────
    records: list[dict] = []
    skipped = 0
    for _, row in df_map.iterrows():
        box = _parse_global_box(str(row['global_box']))
        if box is None:
            skipped += 1
            continue
        x1, y1, x2, y2 = box
        records.append({
            'filename': str(row['filename']).strip(),
            'ocr_text': str(row['ocr_text']),
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })

    if skipped:
        print(f'[WARNING] {skipped} записей пропущено из-за невалидного global_box.')
    print(f'[INFO] Записей с корректными координатами: {len(records)}.')

    # ── Загрузка моделей и processing_info ───────────────────────────────────
    print('[INFO] Загрузка CNN модели...')
    cnn_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))

    print('[INFO] Загрузка RNN модели...')
    rnn_model = tf.keras.models.load_model(str(RNN_MODEL_PATH))

    with open(CNN_INFO_PATH, encoding='utf-8') as f:
        cnn_info = json.load(f)
    with open(RNN_INFO_PATH, encoding='utf-8') as f:
        rnn_info = json.load(f)

    # ── Inference + ансамбль ─────────────────────────────────────────────────
    predicted_classes = run_ensemble_inference(
        records, cnn_model, rnn_model, cnn_info, rnn_info,
        cnn_weight=args.cnn_weight,
        batch_size=args.batch_size,
    )

    for rec, cls in zip(records, predicted_classes):
        rec['predicted_class'] = cls

    # Сводка по предсказаниям
    from collections import Counter
    counts = Counter(predicted_classes)
    print('\n[INFO] Распределение предсказаний:')
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * (cnt * 30 // max(counts.values()))
        print(f'  {cls:<14} {cnt:>4}  {bar}')

    # ── Загрузка и масштабирование TIF ───────────────────────────────────────
    print(f'\n[INFO] Загрузка TIF: {tif_path} ...')
    Image.MAX_IMAGE_PIXELS = None          # TIF могут весить >100 МБ
    pil_img = Image.open(tif_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    print(f'[INFO] Оригинальный размер: {orig_w}×{orig_h} px.')

    new_w = max(1, int(orig_w * args.scale))
    new_h = max(1, int(orig_h * args.scale))
    print(f'[INFO] Масштабированный размер: {new_w}×{new_h} px (scale={args.scale:.0%}).')

    # LANCZOS — лучший фильтр для даун-семплинга карт (сохраняет чёткость текста)
    pil_img_small = pil_img.resize((new_w, new_h), Image.LANCZOS)
    img_array = np.array(pil_img_small)

    # ── Построение Plotly-фигуры ─────────────────────────────────────────────
    print('[INFO] Построение интерактивной визуализации...')
    fig = build_plotly_figure(img_array, records, scale=args.scale, map_name=map_name)

    # ── Сохранение HTML ──────────────────────────────────────────────────────
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(args.output),
        include_plotlyjs='cdn',    # ~3 МБ Plotly.js подгружается один раз с CDN
        full_html=True,
    )

    file_size_mb = args.output.stat().st_size / 1024 / 1024
    print(f'\n[OK] Файл сохранён: {args.output}')
    print(f'     Размер: {file_size_mb:.1f} МБ')
    print(f'     Открой в браузере: file://{args.output.resolve()}')
    print('\n  Управление в браузере:')
    print('    Зажать мышь   — выделить область для зума')
    print('    Двойной клик  — сброс зума')
    print('    Hover на боксе — показать предсказанный класс')
    print('    Клик в легенде — скрыть/показать класс')


if __name__ == '__main__':
    main()