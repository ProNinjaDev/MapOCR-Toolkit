from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.utils import to_categorical
import numpy as np

def create_char_level_lstm_model(max_seq_len, num_chars, num_classes=6):
    model = Sequential()

    model.add(LSTM(128, input_shape=(max_seq_len, num_chars)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == '__main__':
    example_max_seq_len = 20
    example_num_chars = 35
    example_num_classes = 6

    model = create_char_level_lstm_model(max_seq_len=example_max_seq_len, 
                                         num_chars=example_num_chars, 
                                         num_classes=example_num_classes)
    model.summary()
