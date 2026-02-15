"""Подготовка CSV для ручной разметки на основе dataset_CLEANED_v2.csv.

Скрипт:
1) добавляет колонку `label`, если её нет;
2) добавляет колонку `priority_for_labeling` на основе confidence;
3) сортирует примеры так, чтобы сначала шли приоритетные для разметки.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path('data/dataset_CLEANED_v2.csv')
DEFAULT_OUTPUT = Path('data/dataset_CLEANED_v2_labeled.csv')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Подготовка очереди для ручной разметки')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Путь до исходного CSV')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Куда сохранить CSV для разметки')
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.8,
        help='Порог confidence для приоритетной разметки (по умолчанию: 0.8)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=0,
        help='Ограничить количество строк в выходном файле (0 = без ограничения)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f'Входной файл не найден: {args.input}')

    df = pd.read_csv(args.input, sep=';', encoding='utf-8-sig')

    required_columns = {'filename', 'ocr_text', 'confidence'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f'В CSV отсутствуют обязательные колонки: {sorted(missing)}')

    if 'label' not in df.columns:
        df['label'] = ''
    else:
        df['label'] = df['label'].fillna('')

    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.0)
    df['priority_for_labeling'] = df['confidence'] >= args.confidence_threshold

    df = df.sort_values(by=['priority_for_labeling', 'confidence'], ascending=[False, False])

    if args.limit > 0:
        df = df.head(args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, sep=';', index=False, encoding='utf-8-sig')

    total = len(df)
    priority = int(df['priority_for_labeling'].sum())
    print(f'Готово: {args.output}')
    print(f'Строк: {total}; приоритетных (confidence >= {args.confidence_threshold}): {priority}')


if __name__ == '__main__':
    main()
