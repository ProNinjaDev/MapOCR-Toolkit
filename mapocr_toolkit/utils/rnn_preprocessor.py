import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

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

def one_hot_encode_padded_sequences(padded_sequences_array, num_chars_vocab):
    return to_categorical(padded_sequences_array, num_classes=num_chars_vocab)

def prepare_rnn_data(data_items, class_to_int_map, val_split_size=0.2, random_state_value=42):
    texts = []
    raw_class_labels = []

    for item_path, item_label in data_items:
        try:
            with open(item_path, 'r', encoding='utf-8') as f:
                texts.append(f.read().strip())
            raw_class_labels.append(item_label)
        except Exception as ex:
            print('[ERROR] Couldnt read the file')
            continue
    
    char_map, _, num_chars_dict = create_char_vocabulary(texts)
    max_len_seq = get_max_seq_length(texts)

    padded_sequences_indexes = texts_to_padded_sequences(texts, char_map, max_len_seq)
    x_one_hot_sequences = one_hot_encode_padded_sequences(padded_sequences_indexes, num_chars_dict)

    int_class_labels = [class_to_int_map[label] for label in raw_class_labels]
    num_classes_overall = len(class_to_int_map)
    y_one_hot_labels = to_categorical(np.array(int_class_labels), num_classes=num_classes_overall)

    x_train, x_val, y_train, y_val = train_test_split(x_one_hot_sequences,
                                                      y_one_hot_labels,
                                                      test_size=val_split_size,
                                                      random_state=random_state_value,
                                                      stratify=y_one_hot_labels)
    processing_info = {'char_to_int_map': char_map,
                       'max_seq_len': max_len_seq,
                       'num_chars_vocab': num_chars_dict,
                       'class_to_int_map': class_to_int_map,
                       'int_to_class_map': {int_label: str_label for str_label, int_label in class_to_int_map.items()}}
    
    return (x_train, y_train), (x_val, y_val), processing_info


if __name__ == '__main__':
    pass