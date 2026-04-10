#!/usr/bin/env python3
"""
scripts/visualize_map.py  —  Issue #12
Визуализирует предсказания ансамбля CNN+RNN поверх исходной карты.

Запуск из корня репозитория:
    python scripts/visualize_map.py --map img20250920_20532456.tif
    python scripts/visualize_map.py --map img20250920_20532456.tif \\
        --scale 0.12 --cnn-weight 0.65 --output outputs/map_annotated.html

диагностика
    python scripts/visualize_map.py --map img20250920_20532456.tif --diagnose
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array

CNN_MODEL_PATH = PROJECT_ROOT / 'models' / 'demo' / 'cnn' / 'cnn_model.keras'
RNN_MODEL_PATH = PROJECT_ROOT / 'models' / 'demo' / 'rnn' / 'rnn_model.keras'
CNN_INFO_PATH  = PROJECT_ROOT / 'models' / 'demo' / 'cnn' / 'cnn_processing_info.json'
RNN_INFO_PATH  = PROJECT_ROOT / 'models' / 'demo' / 'rnn' / 'rnn_processing_info.json'

LABELS_CSV = PROJECT_ROOT / 'data' / 'dataset_LABELED.csv'
CROPS_DIR  = PROJECT_ROOT / 'data' / 'dataset_crops_paddle'
TIFS_DIR   = PROJECT_ROOT / 'data' / 'raw_tifs'

CLASS_COLORS: dict[str, str] = {
    'city_major': '#e74c3c',
    'city':       '#e67e22',
    'settlement': '#27ae60',
    'hydro':      '#3498db',
    'region':     '#9b59b6',
    'other':      '#7f8c8d',
}
DEFAULT_COLOR = '#cccccc'

CNN_TARGET_H = 60
CNN_TARGET_W = 200

def _parse_global_box(raw: str) -> Optional[tuple[int, int, int, int]]:
    if not isinstance(raw, str) or not raw.strip():
        return None

    s = raw.strip()

    s = re.sub(r'np\.\w+\((-?\d+)\)', r'\1', s)

    # парсим уже чистый пайтон список 
    try:
        pts = ast.literal_eval(s)

        # формат [[x1, y1], [x2, y2]]
        if (isinstance(pts, list) and len(pts) == 2
                and isinstance(pts[0], (list, tuple)) and len(pts[0]) == 2
                and isinstance(pts[1], (list, tuple)) and len(pts[1]) == 2):
            return int(pts[0][0]), int(pts[0][1]), int(pts[1][0]), int(pts[1][1])

        # формат [x1, y1, x2, y2]
        if isinstance(pts, list) and len(pts) == 4:
            return int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])

    except (ValueError, SyntaxError, TypeError):
        pass

    # только если оба предыдущих шага не сработали
    try:
        nums = [int(m) for m in re.findall(r'-?\d+', s)]
        if len(nums) == 4:
            return nums[0], nums[1], nums[2], nums[3]
    except Exception:
        pass

    return None


def _diagnose_box_format(df, n_samples: int = 5) -> None:
    if 'global_box' not in df.columns:
        print('[WARN] Колонка global_box отсутствует в CSV!')
        return
    print('\n[ДИАГНОСТИКА] Примеры значений global_box:')
    for val in df['global_box'].dropna().head(n_samples):
        parsed = _parse_global_box(str(val))
        status = 'OK' if parsed else 'ОШИБКА'
        print(f'  [{status}]  raw={repr(str(val)[:80])}')
        print(f'         ->  parsed={parsed}')
    print()

def _read_csv(csv_path: Path):
    import pandas as pd
    for sep in (';', ','):
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig')
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    raise ValueError(f'Не удалось прочитать CSV: {csv_path}')


def _encode_text_for_rnn(
    text: str,
    char_to_int: dict[str, int],
    max_seq_len: int,
    num_chars: int,
) -> np.ndarray:
    
    pad_idx = char_to_int.get('\0', 0)
    encoded = [char_to_int.get(c, pad_idx) for c in str(text).strip()]

    if len(encoded) >= max_seq_len:
        encoded = encoded[:max_seq_len]
    else:
        encoded += [pad_idx] * (max_seq_len - len(encoded))

    x = np.zeros((1, max_seq_len, num_chars), dtype='float32')
    for t, idx in enumerate(encoded):
        if 0 <= idx < num_chars:
            x[0, t, idx] = 1.0
    return x


def _load_crop_for_cnn(image_path: Path) -> Optional[np.ndarray]:
    try:
        img = load_img(str(image_path), target_size=(CNN_TARGET_H, CNN_TARGET_W))
        return (img_to_array(img) / 255.0).astype('float32')
    except Exception as e:
        print(f'[WARNING] Не удалось загрузить кроп {image_path.name}: {e}')
        return None


def run_ensemble_inference(
    records: list[dict],
    cnn_model,
    rnn_model,
    cnn_info: dict,
    rnn_info: dict,
    cnn_weight: float = 0.65,
    batch_size: int = 64,
) -> list[str]:
    int_to_class: dict[int, str] = {
        int(k): v for k, v in cnn_info['int_to_class'].items()
    }
    char_to_int: dict[str, int] = rnn_info['char_to_int_map']
    max_seq_len: int             = int(rnn_info['max_seq_len'])
    num_chars: int               = int(rnn_info['num_chars_vocab'])
    num_classes: int             = len(int_to_class)

    n = len(records)

    if n == 0:
        print('[WARNING] Нет записей для inference.')
        return []

    # загрузка кропов 
    print('[INFO] Загрузка кропов для CNN...')
    cnn_inputs: list[Optional[np.ndarray]] = []
    for rec in tqdm(records, desc='  кропы'):
        crop_path = CROPS_DIR / rec['filename']
        cnn_inputs.append(_load_crop_for_cnn(crop_path))

    p_cnn = np.full((n, num_classes), 1.0 / num_classes, dtype='float32')

    valid_idx = [i for i, x in enumerate(cnn_inputs) if x is not None]
    if valid_idx:
        print('[INFO] CNN inference...')
        cnn_batch = np.stack([cnn_inputs[i] for i in valid_idx])
        preds = cnn_model.predict(cnn_batch, batch_size=batch_size, verbose=0)
        for out_i, src_i in enumerate(valid_idx):
            p_cnn[src_i] = preds[out_i]
    else:
        print('[WARNING] Ни одного кропа не загружено — CNN использует равномерный prior.')

    missing = n - len(valid_idx)
    if missing:
        print(f'[WARNING] {missing} кропов не найдено — для них CNN prior = равномерный.')

    # кодирование текстов 
    print('[INFO] Кодирование текстов для RNN...')
    rnn_batch = np.zeros((n, max_seq_len, num_chars), dtype='float32')
    for i, rec in enumerate(records):
        encoded = _encode_text_for_rnn(rec['ocr_text'], char_to_int, max_seq_len, num_chars)
        rnn_batch[i] = encoded[0]

    print('[INFO] RNN inference...')
    p_rnn = rnn_model.predict(rnn_batch, batch_size=batch_size, verbose=0)

    p_ensemble = cnn_weight * p_cnn + (1.0 - cnn_weight) * p_rnn

    predicted_indices = np.argmax(p_ensemble, axis=1)
    return [int_to_class.get(int(idx), 'unknown') for idx in predicted_indices]


# ─────────────────────────────────────────────────────────────────────────────
#  Визуализация: Plotly HTML
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_figure(
    img_array: np.ndarray,
    annotated_records: list[dict],
    scale: float,
    map_name: str,
) -> go.Figure:
    h_scaled, w_scaled = img_array.shape[:2]
    fig = go.Figure()

    # Фоновое изображение карты
    fig.add_trace(go.Image(z=img_array, hoverinfo='skip', name='карта'))

    boxes_by_class: dict[str, list[dict]] = {}
    for rec in annotated_records:
        boxes_by_class.setdefault(rec['predicted_class'], []).append(rec)

    for cls_name, cls_records in sorted(boxes_by_class.items()):
        color = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)

        outline_x: list = []
        outline_y: list = []
        center_x:  list[float] = []
        center_y:  list[float] = []
        hover_texts: list[str] = []

        for rec in cls_records:
            x1s = rec['x1'] * scale
            y1s = rec['y1'] * scale
            x2s = rec['x2'] * scale
            y2s = rec['y2'] * scale

            # 5 точек замкнутого прямоугольника + None = разрыв до следующего
            outline_x += [x1s, x2s, x2s, x1s, x1s, None]
            outline_y += [y1s, y1s, y2s, y2s, y1s, None]

            center_x.append((x1s + x2s) / 2)
            center_y.append((y1s + y2s) / 2)
            hover_texts.append(f'<b>{cls_name}</b>')

        # Trace A — видимые контуры
        fig.add_trace(go.Scatter(
            x=outline_x, y=outline_y,
            mode='lines',
            line=dict(color=color, width=2),
            name=cls_name,
            legendgroup=cls_name,
            hoverinfo='skip',
            showlegend=True,
        ))

        # Trace B — невидимые маркеры для hover
        # size=16 = зона наведения мыши 16 пикселей, opacity=0 = визуально скрыт
        fig.add_trace(go.Scatter(
            x=center_x, y=center_y,
            mode='markers',
            marker=dict(size=16, color=color, opacity=0),
            name=cls_name,
            legendgroup=cls_name,
            showlegend=False,
            text=hover_texts,
            hoverinfo='text',
        ))

    n_boxes = len(annotated_records)
    fig.update_layout(
        title=dict(
            text=(f'<b>MapOCR — предсказания ансамбля CNN+RNN</b>'
                  f'<br><sup>Карта: {map_name} · Боксов: {n_boxes} · '
                  f'Масштаб: {scale:.0%}</sup>'),
            x=0.5, font=dict(size=15),
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0, w_scaled]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[h_scaled, 0],  # y сверху вниз — как в изображении
                   scaleanchor='x'),     # пропорции карты не искажаются
        legend=dict(title='Класс (клик — скрыть)',
                    bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#cccccc', borderwidth=1),
        margin=dict(l=10, r=10, t=80, b=10),
        plot_bgcolor='#1a1a2e',
        paper_bgcolor='#1a1a2e',
        font=dict(color='#e0e0e0'),
        hovermode='closest',
        dragmode='zoom',
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Визуализация предсказаний ансамбля CNN+RNN (Issue #12)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--map', required=True,
                        help='Имя TIF-файла, например: img20250920_20532456.tif')
    parser.add_argument('--scale', type=float, default=0.4,
                        help='Коэффициент уменьшения карты (0.4 = 15%% от оригинала)')
    parser.add_argument('--cnn-weight', type=float, default=0.65,
                        help='Вес CNN в ансамбле. Вес RNN = 1 - cnn_weight.')
    parser.add_argument('--output', type=Path,
                        default=PROJECT_ROOT / 'outputs' / 'map_annotated.html',
                        help='Куда сохранить HTML.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Размер батча при инференсе')
    parser.add_argument('--diagnose', action='store_true',
                        help='Показать примеры raw global_box и выйти без построения карты.')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='порог IoU для NMS-дедупликации боксов (0.5 по умолчанию)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    map_name: str = args.map

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
            print(f'[ERROR] Файл не найден: {path}  ({label})')
            sys.exit(1)

    print(f'[INFO] Загрузка {LABELS_CSV.name}...')
    df = _read_csv(LABELS_CSV)
    print(f'[INFO] Колонки в CSV: {list(df.columns)}')

    if args.diagnose:
        _diagnose_box_format(df, n_samples=10)
        sys.exit(0)

    required_cols = {'filename', 'ocr_text', 'global_box', 'source_map'}
    missing_cols  = required_cols - set(df.columns)
    if missing_cols:
        print(f'[ERROR] В CSV нет колонок: {sorted(missing_cols)}')
        sys.exit(1)

    df_map = df[df['source_map'] == map_name].copy()
    if df_map.empty:
        print(f'[ERROR] В CSV нет записей для карты "{map_name}".')
        print(f'        Доступные карты: {sorted(df["source_map"].dropna().unique())}')
        sys.exit(1)

    print(f'[INFO] Найдено {len(df_map)} записей для карты "{map_name}".')

    _diagnose_box_format(df_map, n_samples=3)

    records: list[dict] = []
    skipped = 0
    for _, row in df_map.iterrows():
        box = _parse_global_box(str(row['global_box']))
        if box is None:
            skipped += 1
            continue
        x1, y1, x2, y2 = box
        
        # global_box и confidence нужны для nms
        records.append({
            'filename': str(row['filename']).strip(),
            'ocr_text': str(row['ocr_text']),
            'global_box': str(row['global_box']),
            'confidence': float(row['confidence']) if 'confidence' in df_map.columns else 0.0,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        })
    
    # внедряем nms
    from mapocr_toolkit.utils.nms import nms_filter
    before = len(records)
    records = nms_filter(records, iou_threshold=args.iou_threshold)
    removed = before - len(records)
    if removed:
        print(f'[NMS] удалено дублей: {removed} '
              f'(осталось: {len(records)}, порог IoU={args.iou_threshold})')

    if skipped:
        print(f'[WARNING] {skipped} записей пропущено (невалидный global_box).')

    print(f'[INFO] Записей с корректными координатами: {len(records)}.')

    if len(records) == 0:
        print('\n[ERROR] Нет ни одной записи с валидным global_box.')
        print(f'        python scripts/visualize_map.py --map {map_name} --diagnose')
        sys.exit(1)

    print('[INFO] Загрузка CNN модели...')
    cnn_model = tf.keras.models.load_model(str(CNN_MODEL_PATH))
    print('[INFO] Загрузка RNN модели...')
    rnn_model = tf.keras.models.load_model(str(RNN_MODEL_PATH))

    with open(CNN_INFO_PATH, encoding='utf-8') as f:
        cnn_info = json.load(f)
    with open(RNN_INFO_PATH, encoding='utf-8') as f:
        rnn_info = json.load(f)

    predicted_classes = run_ensemble_inference(
        records, cnn_model, rnn_model, cnn_info, rnn_info,
        cnn_weight=args.cnn_weight,
        batch_size=args.batch_size,
    )

    for rec, cls in zip(records, predicted_classes):
        rec['predicted_class'] = cls

    counts = Counter(predicted_classes)
    print('\n[INFO] Распределение предсказаний:')
    for cls, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = '█' * max(1, cnt * 30 // max(counts.values()))
        print(f'  {cls:<14} {cnt:>4}  {bar}')

    print(f'\n[INFO] Загрузка TIF: {tif_path} ...')
    Image.MAX_IMAGE_PIXELS = None
    pil_img = Image.open(tif_path).convert('RGB')
    orig_w, orig_h = pil_img.size
    print(f'[INFO] Оригинальный размер: {orig_w}x{orig_h} px.')

    new_w = max(1, int(orig_w * args.scale))
    new_h = max(1, int(orig_h * args.scale))
    print(f'[INFO] Масштабированный: {new_w}x{new_h} px (scale={args.scale:.0%}).')

    pil_img_small = pil_img.resize((new_w, new_h), Image.LANCZOS)
    img_array = np.array(pil_img_small)

    print('[INFO] Построение интерактивной визуализации...')
    fig = build_plotly_figure(img_array, records, scale=args.scale, map_name=map_name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(args.output),
        include_plotlyjs='cdn',
        full_html=True,
    )

    file_size_mb = args.output.stat().st_size / 1024 / 1024
    print(f'\n[OK] Файл сохранён: {args.output}')
    print(f'     Размер: {file_size_mb:.1f} МБ')
    print(f'     Открой в браузере: file://{args.output.resolve()}')
    print('\n  Управление в браузере:')
    print('    Зажать мышь    — выделить область для зума')
    print('    Двойной клик   — сброс зума')
    print('    Hover на боксе — предсказанный класс')
    print('    Клик в легенде — скрыть/показать класс')


if __name__ == '__main__':
    main()