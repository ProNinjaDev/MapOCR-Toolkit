import csv
import sys

def edit_distance(s1, s2):
    """расстояние левенштейна — минимум операций для превращения s1 в s2"""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j-1], prev[j], dp[j-1])
    return dp[n]

def calc_cer_wer(csv_path):
    total_chars_gt = 0
    total_char_errors = 0
    total_words_gt = 0
    total_word_errors = 0
    
    rows_with_errors = []

    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            ocr = row['ocr_text'].strip()
            gt  = row['ground_truth'].strip()
            
            if not gt:
                continue
            
            # --- CER ---
            char_err = edit_distance(ocr, gt)
            total_chars_gt    += len(gt)
            total_char_errors += char_err
            
            # --- WER ---
            ocr_words = ocr.split()
            gt_words  = gt.split()
            word_err = edit_distance(ocr_words, gt_words)
            total_words_gt    += len(gt_words)
            total_word_errors += word_err
            
            if char_err > 0:
                rows_with_errors.append({
                    'ocr': ocr,
                    'gt':  gt,
                    'char_err': char_err,
                    'cer_item': round(char_err / len(gt), 3)
                })

    cer = total_char_errors / total_chars_gt if total_chars_gt else 0
    wer = total_word_errors / total_words_gt if total_words_gt else 0

    print(f"\n{'='*50}")
    print(f"Всего записей с ground truth: {total_words_gt} слов, {total_chars_gt} символов")
    print(f"Ошибок символов:  {total_char_errors}")
    print(f"Ошибок слов:      {total_word_errors}")
    print(f"\nCER = {cer:.4f}  ({cer*100:.1f}%)")
    print(f"WER = {wer:.4f}  ({wer*100:.1f}%)")
    print(f"\nПримеры ошибок (первые 10):")
    for r in rows_with_errors[:10]:
        print(f"  OCR: '{r['ocr']}'  →  GT: '{r['gt']}'  (char_err={r['char_err']})")

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'cer_wer_sample.csv'
    calc_cer_wer(path)