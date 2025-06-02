import easyocr
from PIL import Image
import Levenshtein

reader = easyocr.Reader(['ru'], gpu=True)

clean_image_path = 'cleared_images/page_1.png'
raw_image_path = 'temp_images/page_1.png'

results_clean_easy = reader.readtext(clean_image_path, paragraph=True)

clean_text_easy = ' '.join([res[1] for res in results_clean_easy])

print("\n--- Clean Image Text (EasyOCR) ---")
print(clean_text_easy)
print("---------------------------------\n")


print(f"Processing Raw Image: {raw_image_path}")
results_raw_easy = reader.readtext(raw_image_path, paragraph=True)

raw_text_easy = ' '.join([res[1] for res in results_raw_easy])

print("\n--- Raw Image Text (EasyOCR) ---")
print(raw_text_easy)
print("--------------------------------\n")

ground_truth_path = 'ground_truth/page_1.txt'

try:
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        text_truth = f.read()

except FileNotFoundError:
    print('[ERROR] Ground truth file not found')

if (text_truth):
    lev_distance_raw_cer = Levenshtein.distance(raw_text_easy, text_truth)
    lev_distance_clean_cer = Levenshtein.distance(clean_text_easy, text_truth)

    cer_raw = lev_distance_raw_cer / len(text_truth)
    cer_clean = lev_distance_clean_cer / len(text_truth)
    print(f'Raw image CER = {cer_raw:.4f}')
    print(f'Clean image CER = {cer_clean:.4f}')

    raw_words = raw_text_easy.split()
    clean_words = clean_text_easy.split()
    truth_words = text_truth.split()

    lev_distance_raw_wer = Levenshtein.distance(raw_words, truth_words)
    lev_distance_clean_wer = Levenshtein.distance(clean_words, truth_words)

    wer_raw = lev_distance_raw_wer / len(truth_words)
    wer_clean = lev_distance_clean_wer / len(truth_words)

    print(f'Raw image WER = {wer_raw:.4f}')
    print(f'Clean image WER = {wer_clean:.4f}')
