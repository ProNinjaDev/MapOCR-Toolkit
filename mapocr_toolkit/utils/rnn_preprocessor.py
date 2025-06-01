import numpy as np
from tensorflow import keras
from keras.utils import to_categorical

PAD_CHAR = '\0'

def create_char_vocabulary(text_list):
    all_chars = set()
    for text in text_list:
        for char in text:
            all_chars.add(char)
    
    sorted_chars = sorted(list(all_chars))

    char_to_int = {PAD_CHAR: 0}
    int_to_char = {0: PAD_CHAR}
    
    for i, char in enumerate(sorted_chars):
        char_to_int[char] = i + 1
        int_to_char[i + 1] = char
        
    num_chars = len(char_to_int)
    return char_to_int, int_to_char, num_chars

def get_max_seq_length(text_list):
    if text_list:
        return max(len(text) for text in text_list)
    else:
        return 0

def texts_to_padded_sequences(text_list, char_to_int_map, max_len):
    sequences = []
    pad_index = char_to_int_map[PAD_CHAR] # пока что ноль

    for text in text_list:
        encoded_sequence = [char_to_int_map.get(char, pad_index) for char in text]
        
        current_len = len(encoded_sequence)

        if current_len > max_len:
            padded_sequence_np = np.array(encoded_sequence[:max_len])
        elif current_len < max_len:
            padding_size = max_len - current_len
            padded_sequence_np = np.pad(np.array(encoded_sequence), (0, padding_size), mode='constant', constant_values=pad_index)
        else:
            padded_sequence_np = np.array(encoded_sequence)
        
        sequences.append(padded_sequence_np)
    
    return np.array(sequences)


if __name__ == '__main__':
    pass