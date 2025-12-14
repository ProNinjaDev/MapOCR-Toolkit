import os
import pytesseract
from PIL import Image
import pandas as pd
from tqdm import tqdm

INPUT_DIR = os.path.join('data', 'raw_tifs')
OUTPUT_IMG_DIR = os.path.join('data', 'dataset_crops')
OUTPUT_CSV = os.path.join('data', 'dataset_v1.csv')

# отступы вокруг текста
PADDING = 10 
# минимальная уверенность тессеракта (из ютуба)
CONFIDENCE_THRESHOLD = 40 

# tif 100 мб каждый файл, ограничения не нужны
Image.MAX_IMAGE_PIXELS = None
# =============================================

def process_tiffs():
    if not os.path.exists(OUTPUT_IMG_DIR):
        os.makedirs(OUTPUT_IMG_DIR)

    # Список для формирования CSV
    dataset_records = []
    
    # Получаем список всех tiff файлов
    tiff_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.tif', '.tiff'))]
    
    print(f"Найдено файлов: {len(tiff_files)}. Начинаем нарезку...")

    global_crop_counter = 0

    for tiff_file in tiff_files:
        tiff_path = os.path.join(INPUT_DIR, tiff_file)
        print(f"\nОбработка файла: {tiff_file} (это может занять время...)")
        
        try:
            # Загружаем изображение
            img = Image.open(tiff_path)
            # Конвертируем в RGB, если вдруг там CMYK или Grayscale
            img = img.convert('RGB')
            
            # 1. Прогоняем Tesseract, чтобы получить данные о боксах (data mining)
            # output_type=dict дает словарь со списками координат, текста и conf
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='rus+eng')
            
            n_boxes = len(data['text'])
            
            # Проходимся по всем найденным элементам
            for i in tqdm(range(n_boxes), desc=f"Анализ {tiff_file}"):
                # Берем только если уверенность > порога и текст не пустой
                text = data['text'][i].strip()
                conf = int(data['conf'][i]) if data['conf'][i] != '-1' else 0
                
                if conf > CONFIDENCE_THRESHOLD and len(text) > 1:
                    (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                    
                    # Добавляем отступы (padding)
                    x_new = max(0, x - PADDING)
                    y_new = max(0, y - PADDING)
                    w_new = w + 2 * PADDING
                    h_new = h + 2 * PADDING
                    
                    # Вырезаем кусочек
                    crop = img.crop((x_new, y_new, x_new + w_new, y_new + h_new))
                    
                    # Генерируем имя файла: mapname_counter.jpg
                    crop_filename = f"{os.path.splitext(tiff_file)[0]}_{global_crop_counter}.jpg"
                    crop_path = os.path.join(OUTPUT_IMG_DIR, crop_filename)
                    
                    # Сохраняем картинку
                    crop.save(crop_path, "JPEG", quality=95)
                    
                    # Записываем в список для CSV
                    # label оставляем пустым, ты его заполнишь руками
                    dataset_records.append({
                        'filename': crop_filename,
                        'ocr_text': text,
                        'label': '', 
                        'source_map': tiff_file
                    })
                    
                    global_crop_counter += 1
                    
        except Exception as e:
            print(f"[ERROR] Ошибка при обработке {tiff_file}: {e}")

    # Сохраняем итоговый CSV
    if dataset_records:
        df = pd.DataFrame(dataset_records)
        # Сортируем колонки для удобства
        df = df[['filename', 'ocr_text', 'label', 'source_map']]
        df.to_csv(OUTPUT_CSV, index=False, sep=';', encoding='utf-8-sig') # utf-8-sig чтобы Excel открыл нормально
        print(f"\nГотово! Вырезано {len(dataset_records)} фрагментов.")
        print(f"CSV файл создан: {OUTPUT_CSV}")
        print(f"Картинки лежат в: {OUTPUT_IMG_DIR}")
    else:
        print("Ничего не найдено или возникли ошибки.")

if __name__ == '__main__':
    process_tiffs()