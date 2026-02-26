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

# Инициализация PaddleOCR
# Используем параметры, актуальные для новых версий, чтобы минимизировать Warning-и
ocr = PaddleOCR(
    lang='ru', 
    use_angle_cls=True,             # Включаем классификатор углов (для совместимости оставим так)
    det_limit_side_len=SLICE_SIZE + 100, 
    det_limit_type='max',
)
# =============================================

def get_box_crop(img_cv2, box, padding):
    try:
        # box уже должен быть np.array int32
        x_min = max(0, np.min(box[:, 0]) - padding)
        y_min = max(0, np.min(box[:, 1]) - padding)
        x_max = min(img_cv2.shape[1], np.max(box[:, 0]) + padding)
        y_max = min(img_cv2.shape[0], np.max(box[:, 1]) + padding)
        
        return img_cv2[y_min:y_max, x_min:x_max]
    except Exception:
        return np.array([])

def process_tiffs_v3_robust():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    dataset_records = []
    
    if not os.path.exists(INPUT_DIR):
        print(f"[ERROR] Папка {INPUT_DIR} не найдена!")
        return

    tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff'))]
    print(f"Найдено файлов: {len(tiff_files)}. Режим: Sliding Window (Fix Dimensions).")

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
                    
                    # Корректировка границ, чтобы не вылезти за пределы
                    y_end = min(y_start + SLICE_SIZE, h)
                    x_end = min(x_start + SLICE_SIZE, w)
                    
                    if (y_end - y_start) < SLICE_SIZE and y_start > 0:
                        y_start = max(0, y_end - SLICE_SIZE)
                    if (x_end - x_start) < SLICE_SIZE and x_start > 0:
                        x_start = max(0, x_end - SLICE_SIZE)

                    slice_img = full_img[y_start:y_end, x_start:x_end]
                    
                    # === РАСПОЗНАВАНИЕ ===
                    try:
                        # Используем predict() для v3/PaddleX
                        results = list(ocr.predict(slice_img))
                    except Exception:
                        pbar.update(1)
                        continue

                    if not results:
                        pbar.update(1)
                        continue

                    res_obj = results[0]
                    
                    # Извлечение данных из JSON
                    data = {}
                    if hasattr(res_obj, 'json'):
                        # .json иногда парсится как строка, иногда как словарь
                        raw_json = res_obj.json
                        if isinstance(raw_json, dict):
                            data = raw_json.get('res', raw_json)
                        else:
                             # Если это вдруг не dict, пропускаем
                             pass
                    
                    # Получаем списки
                    boxes = data.get('dt_polys', [])
                    texts = data.get('rec_texts', data.get('rec_text', []))
                    scores = data.get('rec_scores', data.get('rec_score', []))

                    if not boxes or not texts:
                        pbar.update(1)
                        continue

                    count_items = len(texts)
                    
                    # Итерируемся по найденным элементам
                    for i in range(count_items):
                        try:
                            # 1. Получаем бокс
                            raw_box = boxes[i]
                            
                            # === ГЛАВНЫЙ ФИКС: Проверка размерности массива ===
                            box_np = np.array(raw_box)
                            
                            # Если массив пустой, пропускаем
                            if box_np.size == 0:
                                continue
                                
                            # Если массив плоский (1D), превращаем его в 2D (4 точки по 2 координаты)
                            if box_np.ndim == 1:
                                # Обычно это 4 точки (x1,y1, x2,y2, x3,y3, x4,y4) -> reshape(-1, 2)
                                box_np = box_np.reshape(-1, 2)
                            
                            # Если это не (N, 2), то мы не знаем, что с этим делать
                            if box_np.shape[1] != 2:
                                continue
                            
                            # Приводим к int
                            box_np = box_np.astype(np.int32)

                            # 2. Получаем текст и уверенность
                            text = texts[i]
                            score = scores[i]
                            if isinstance(score, list): score = score[0]
                            
                            if float(score) < CONFIDENCE_THRESHOLD:
                                continue
                            if len(text) < 2:
                                continue

                            # 3. Пересчет координат в ГЛОБАЛЬНЫЕ
                            # Теперь box_np гарантированно 2D, ошибки IndexError не будет
                            global_box = box_np.copy()
                            global_box[:, 0] += x_start
                            global_box[:, 1] += y_start

                            # 4. Вырезаем (crop) из текущего слайса (быстрее)
                            crop = get_box_crop(slice_img, box_np, PADDING)
                            if crop.size == 0: continue

                            # Сохранение
                            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                            crop_filename = f"{os.path.splitext(tiff_file)[0]}_x{x_start}_y{y_start}_{global_counter}.jpg"
                            save_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
                            
                            cv2.imwrite(save_path, crop_bgr)
                            
                            dataset_records.append({
                                'filename': crop_filename,
                                'ocr_text': text,
                                'confidence': round(float(score), 4),
                                'global_box': global_box.tolist(),
                                'source_map': tiff_file
                            })
                            global_counter += 1
                        
                        except Exception as inner_e:
                            # Если один бокс сбойнул, не роняем весь процесс
                            continue
                    
                    pbar.update(1)
            
            pbar.close()

        except KeyboardInterrupt:
            print("\n[STOP] Прервано пользователем. Сохраняем то, что успели...")
            break
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Файл {tiff_file} пропущен: {e}")
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
    process_tiffs_v3_robust()