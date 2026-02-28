import os
import cv2
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR
from PIL import Image
from tqdm import tqdm
import logging
import math

# Глушим логи Paddle
logging.getLogger("ppocr").setLevel(logging.ERROR)
os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

# ================= НАСТРОЙКИ =================
INPUT_DIR = os.path.join('data', 'raw_tifs')
OUTPUT_IMG_DIR = os.path.join('data', 'dataset_crops_paddle')
OUTPUT_CSV = os.path.join('data', 'dataset_paddle.csv')

PADDING = 10
CONFIDENCE_THRESHOLD = 0.6

# Настройки нарезки
SLICE_SIZE = 2000
OVERLAP = 400

# Инициализация PaddleOCR v3
ocr = PaddleOCR(
    lang='ru',
    text_det_limit_side_len=SLICE_SIZE + 100,
    text_det_limit_type='max',
)
# =============================================

def process_tiffs():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    dataset_records = []

    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Папка {INPUT_DIR} не найдена!")
        return

    tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"Найдено файлов: {len(tiff_files)}. Режим: Sliding Window.")

    global_counter = 0

    for tiff_file in tiff_files:
        tiff_path = os.path.join(INPUT_DIR, tiff_file)
        print(f"\n===== Обработка: {tiff_file} =====")

        try:
            Image.MAX_IMAGE_PIXELS = None
            pil_img = Image.open(tiff_path).convert('RGB')
            full_img = np.array(pil_img)

            h, w, _ = full_img.shape
            print(f"Размер: {w}x{h}")

            stride = SLICE_SIZE - OVERLAP
            y_steps = math.ceil((h - OVERLAP) / stride) if h > SLICE_SIZE else 1
            x_steps = math.ceil((w - OVERLAP) / stride) if w > SLICE_SIZE else 1
            total_steps = y_steps * x_steps

            pbar = tqdm(total=total_steps, desc="Фрагменты")

            for y_idx in range(y_steps):
                for x_idx in range(x_steps):
                    y_start = y_idx * stride
                    x_start = x_idx * stride

                    y_end = min(y_start + SLICE_SIZE, h)
                    x_end = min(x_start + SLICE_SIZE, w)

                    if (y_end - y_start) < SLICE_SIZE and y_start > 0:
                        y_start = max(0, y_end - SLICE_SIZE)
                    if (x_end - x_start) < SLICE_SIZE and x_start > 0:
                        x_start = max(0, x_end - SLICE_SIZE)

                    slice_img = full_img[y_start:y_end, x_start:x_end]

                    # === РАСПОЗНАВАНИЕ ===
                    try:
                        results = list(ocr.predict(slice_img))
                    except Exception:
                        pbar.update(1)
                        continue

                    if not results:
                        pbar.update(1)
                        continue

                    res_obj = results[0]

                    # Извлечение данных
                    data = {}
                    if hasattr(res_obj, 'json'):
                        raw_json = res_obj.json
                        if isinstance(raw_json, dict):
                            data = raw_json.get('res', raw_json)

                    # rec_boxes — правильные координаты в формате [x_min, y_min, x_max, y_max]
                    # dt_polys — координаты ПОСЛЕ внутреннего поворота, не подходят для кропа
                    boxes  = data.get('rec_boxes', [])
                    texts  = data.get('rec_texts', data.get('rec_text', []))
                    scores = data.get('rec_scores', data.get('rec_score', []))

                    if not boxes or not texts:
                        pbar.update(1)
                        continue

                    for i in range(len(texts)):
                        try:
                            # rec_boxes формат: [x_min, y_min, x_max, y_max]
                            box = np.array(boxes[i])
                            if box.size != 4:
                                continue
                            x_min, y_min, x_max, y_max = box.astype(np.int32)

                            text = texts[i]
                            score = scores[i]
                            if isinstance(score, list):
                                score = score[0]

                            if float(score) < CONFIDENCE_THRESHOLD:
                                continue
                            if len(str(text).strip()) < 2:
                                continue

                            # Вырезаем из слайса с отступом
                            x_min_p = max(0, x_min - PADDING)
                            y_min_p = max(0, y_min - PADDING)
                            x_max_p = min(slice_img.shape[1], x_max + PADDING)
                            y_max_p = min(slice_img.shape[0], y_max + PADDING)

                            crop = slice_img[y_min_p:y_max_p, x_min_p:x_max_p]
                            if crop.size == 0:
                                continue

                            # Глобальные координаты на исходной карте
                            global_box = [
                                [x_min + x_start, y_min + y_start],
                                [x_max + x_start, y_max + y_start],
                            ]

                            # Сохраняем
                            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                            crop_filename = f"{os.path.splitext(tiff_file)[0]}_x{x_start}_y{y_start}_{global_counter}.jpg"
                            save_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
                            cv2.imwrite(save_path, crop_bgr)

                            dataset_records.append({
                                'filename':   crop_filename,
                                'ocr_text':   text,
                                'confidence': round(float(score), 4),
                                'global_box': global_box,
                                'source_map': tiff_file,
                            })
                            global_counter += 1

                        except Exception:
                            continue

                    pbar.update(1)

            pbar.close()

        except KeyboardInterrupt:
            print("\n[STOP] Прервано. Сохраняем...")
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] {tiff_file}: {e}")
            import traceback
            traceback.print_exc()

    if dataset_records:
        df = pd.DataFrame(dataset_records)
        df.to_csv(OUTPUT_CSV, index=False, sep=';', encoding='utf-8-sig')
        print(f"\nГОТОВО! Сохранено: {len(dataset_records)} шт.")
        print(f"Таблица: {OUTPUT_CSV}")
    else:
        print("\nНичего не сохранено.")

if __name__ == '__main__':
    process_tiffs()