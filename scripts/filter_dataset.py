import pandas as pd
import os
import shutil
import re

# Настройки
INPUT_CSV = os.path.join('data', 'dataset_v1.csv')
OUTPUT_CSV = os.path.join('data', 'dataset_CLEANED.csv')
IMGS_DIR = os.path.join('data', 'dataset_crops')
GARBAGE_DIR = os.path.join('data', 'dataset_crops', 'garbage')

# тех инфа
STOP_WORDS = [
    'тираж', 'заказ', 'цена', 'гугк', 'ссср', 'рсфср', 'картографии', 
    'главное', 'геодезии', 'подписана', 'печати', 'схема', 'редактор',
    'экз', 'копий', 'снятие', 'размножение', 'министров', 'совете'
]

def is_valid_text(text):
    if not isinstance(text, str):
        return False
    
    text = text.strip()
    
    # 1. менее 3 символов
    if len(text) < 3:
        return False
        
    # 2. Состоит только из цифр или спецсимволов 
    clean_text = re.sub(r'[0-9\W_]+', '', text)
    if len(clean_text) < 2:
        return False

    text_lower = text.lower()
    for word in STOP_WORDS:
        if word in text_lower:
            return False

    if not re.search(r'[а-яА-Я]', text):
        pass 

    return True

def clean_dataset():
    if not os.path.exists(INPUT_CSV):
        print("Файл CSV не найден!")
        return

    if not os.path.exists(GARBAGE_DIR):
        os.makedirs(GARBAGE_DIR)

    df = pd.read_csv(INPUT_CSV, sep=';')
    print(f"Всего записей: {len(df)}")

    valid_rows = []
    garbage_count = 0

    for index, row in df.iterrows():
        filename = row['filename']
        text = str(row['ocr_text'])
        
        src_path = os.path.join(IMGS_DIR, filename)
        
        if is_valid_text(text):
            valid_rows.append(row)
        else:
            garbage_count += 1
            dest_path = os.path.join(GARBAGE_DIR, filename)
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dest_path)
                except Exception as e:
                    print(f"Ошибка перемещения {filename}: {e}")

    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv(OUTPUT_CSV, index=False, sep=';', encoding='utf-8-sig')

    print(f"\nГотово!")
    print(f"Осталось полезных записей: {len(valid_rows)}")
    print(f"Перемещено в мусор: {garbage_count}")
    print(f"Работай теперь с файлом: {OUTPUT_CSV}")

if __name__ == '__main__':
    clean_dataset()