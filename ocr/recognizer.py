import pytesseract
from PIL import Image
import Levenshtein
import os



clear_image_path = 'cleared_images/page_1.png'
raw_image_path = 'temp_images/page_1.png'

clear_img = Image.open(clear_image_path)
clear_text_tes = pytesseract.image_to_string(clear_img, lang='rus')
raw_img = Image.open(raw_image_path)
raw_text_tes = pytesseract.image_to_string(raw_img, lang='rus')

ground_truth_path = 'ground_truth/page_1.txt'
text_truth = ''

try:
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        text_truth = f.read()
except FileNotFoundError:
    print('[ERROR] File not found')
    exit()

if text_truth:
    lev_distance_raw = Levenshtein.distance(raw_text_tes, text_truth)
    lev_distance_clear = Levenshtein.distance(clear_text_tes, text_truth)

    cer_raw = lev_distance_raw / len(text_truth)
    cer_clear = lev_distance_clear / len(text_truth)
    print(f'Raw image CER = {cer_raw}')
    print(f'Clear image CER = {cer_clear}')


