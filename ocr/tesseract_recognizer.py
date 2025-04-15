import pytesseract
from PIL import Image
import Levenshtein
import os



clean_image_path = 'cleared_images/page_1.png'
raw_image_path = 'temp_images/page_1.png'

clean_img = Image.open(clean_image_path)
clean_text_tes = pytesseract.image_to_string(clean_img, lang='rus', config='--oem 1 --psm 1')
raw_img = Image.open(raw_image_path)
raw_text_tes = pytesseract.image_to_string(raw_img, lang='rus', config='--oem 1 --psm 1')

ground_truth_path = 'ground_truth/page_1.txt'
text_truth = ''

try:
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        text_truth = f.read()
except FileNotFoundError:
    print('[ERROR] File not found')
    exit()

if text_truth:
    lev_distance_raw_cer = Levenshtein.distance(raw_text_tes, text_truth)
    lev_distance_clean_cer = Levenshtein.distance(clean_text_tes, text_truth)

    cer_raw = lev_distance_raw_cer / len(text_truth)
    cer_clean = lev_distance_clean_cer / len(text_truth)
    print(f'Raw image CER = {cer_raw:.4f}')
    print(f'Clean image CER = {cer_clean:.4f}')

    raw_words = raw_text_tes.split()
    clean_words = clean_text_tes.split()
    truth_words = text_truth.split()

    lev_distance_raw_wer = Levenshtein.distance(raw_words, truth_words)
    lev_distance_clean_wer = Levenshtein.distance(clean_words, truth_words)

    wer_raw = lev_distance_raw_wer / len(truth_words)
    wer_clean = lev_distance_clean_wer / len(truth_words)

    print(f'Raw image WER = {wer_raw:.4f}')
    print(f'Clean image WER = {wer_clean:.4f}')

    #print('Clean text:')
    #print(clean_text_tes)
    #print('Raw text:')
    #print(raw_text_tes)
    
