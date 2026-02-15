import os
from typing import List, Set, Tuple

import pandas as pd


DEFAULT_LABELS_FILE_PATH = os.path.join('data', 'dataset_CLEANED.csv')
RAW_IMAGES_DIR = os.path.join('data', 'dataset_crops')
REQUIRED_COLUMNS = {'filename', 'label', 'ocr_text'}


def _resolve_labels_path() -> str:
    """Возвращает путь до CSV с разметкой.

    Приоритет:
    1) переменная окружения MAPOCR_LABELS_PATH
    2) data/dataset_CLEANED_v2_labeled.csv (новый формат после ручной разметки)
    3) data/dataset_CLEANED.csv (legacy)
    """
    env_path = os.environ.get('MAPOCR_LABELS_PATH')
    if env_path:
        return env_path

    v2_labeled = os.path.join('data', 'dataset_CLEANED_v2_labeled.csv')
    if os.path.exists(v2_labeled):
        return v2_labeled

    return DEFAULT_LABELS_FILE_PATH


def _read_dataset_csv(csv_path: str) -> pd.DataFrame:
    """Читает CSV с поддержкой `;` и `,`, а также BOM."""
    for sep in (';', ','):
        try:
            df = pd.read_csv(csv_path, sep=sep, encoding='utf-8-sig')
            if len(df.columns) > 1:
                return df
        except Exception:
            continue

    raise ValueError(f'Не удалось прочитать CSV файл: {csv_path}')


def load_raw_data_paths_and_labels() -> Tuple[List[Tuple[str, str, str]], Set[str]]:
    data_items: List[Tuple[str, str, str]] = []
    class_labels_set: Set[str] = set()

    labels_file_path = _resolve_labels_path()

    if not os.path.exists(labels_file_path):
        print(f"[ERROR] CSV файл не найден: {labels_file_path}")
        return [], class_labels_set

    try:
        df = _read_dataset_csv(labels_file_path)

        if not REQUIRED_COLUMNS.issubset(df.columns):
            print('[ERROR] В CSV нет нужных колонок (filename, label, ocr_text)')
            return [], class_labels_set

        for _, row in df.iterrows():
            filename = str(row['filename']).strip()
            ocr_text = str(row['ocr_text'])
            label = str(row['label']).strip()

            if not filename or filename.lower() == 'nan':
                continue

            if not label or label.lower() == 'nan':
                continue

            image_full_path = os.path.join(RAW_IMAGES_DIR, filename)

            if os.path.exists(image_full_path):
                data_items.append((image_full_path, ocr_text, label))
                class_labels_set.add(label)
            else:
                print(f'[WARNING] Картинка не найдена: {image_full_path}')

    except Exception as e:
        print(f'[ERROR] Ошибка чтения CSV: {e}')

    return data_items, class_labels_set


def create_class_maps(class_labels_set: Set[str]):
    sorted_labels = sorted(list(class_labels_set))

    class_to_int_map = {}
    int_to_class_map = {}

    for i, label in enumerate(sorted_labels):
        class_to_int_map[label] = i
        int_to_class_map[i] = label

    return class_to_int_map, int_to_class_map
