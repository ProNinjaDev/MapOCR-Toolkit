import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, BatchNormalization,
    Bidirectional, LSTM, Dense, Dropout, Lambda,
)


def create_crnn_model(input_shape=(60, 200, 3), num_classes=5):
    inputs = Input(shape=input_shape, name='input')
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1')(inputs)
    x = MaxPooling2D(pool_size=(2, 2), name='pool1')(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    # уменьшаем высоту, ширину сохраняем
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = BatchNormalization(name='bn1')(x)
    x = MaxPooling2D(pool_size=(2, 1), name='pool3')(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv4')(x)
    x = BatchNormalization(name='bn2')(x)
    x = MaxPooling2D(pool_size=(2, 1), name='pool4')(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv5')(x)
    x = MaxPooling2D(pool_size=(3, 1), name='pool5')(x)

    x = Lambda(lambda t: tf.squeeze(t, axis=1), name='squeeze')(x)

    x = Bidirectional(LSTM(128, return_sequences=False), name='bilstm')(x)
    x = Dropout(0.5, name='dropout')(x)

    outputs = Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs, outputs, name='crnn_classifier')
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def transfer_weights_from_cnn(crnn_model, cnn_model_path: str) -> int:
    cnn = tf.keras.models.load_model(cnn_model_path)

    cnn_conv_layers = [l for l in cnn.layers if isinstance(l, tf.keras.layers.Conv2D)]

    transferred = 0
    for crnn_name, cnn_layer in zip(['conv1', 'conv2'], cnn_conv_layers[:2]):
        crnn_layer = crnn_model.get_layer(crnn_name)
        crnn_w_shape = [w.shape for w in crnn_layer.get_weights()]
        cnn_w_shape  = [w.shape for w in cnn_layer.get_weights()]

        if crnn_w_shape == cnn_w_shape:
            crnn_layer.set_weights(cnn_layer.get_weights())
            print(f'[TRANSFER] {crnn_name}: {cnn_layer.name} → OK  {crnn_w_shape}')
            transferred += 1
        else:
            print(f'[TRANSFER] {crnn_name}: shape mismatch {cnn_w_shape} ≠ {crnn_w_shape}, skip')

    print(f'[TRANSFER] Перенесено слоёв: {transferred}/2')
    return transferred


def freeze_cnn_backbone(crnn_model, freeze: bool = True):
    cnn_layer_names = {'conv1', 'conv2', 'conv3', 'conv4', 'conv5',
                       'bn1', 'bn2',
                       'pool1', 'pool2', 'pool3', 'pool4', 'pool5',
                       'squeeze'}

    for layer in crnn_model.layers:
        if layer.name in cnn_layer_names:
            layer.trainable = not freeze

    status = 'заморожена' if freeze else 'разморожена'
    trainable_params = sum(
        tf.size(w).numpy() for w in crnn_model.trainable_variables
    )
    print(f'[INFO] CNN-часть {status}. Обучаемых параметров: {trainable_params:,}')


if __name__ == '__main__':
    model = create_crnn_model(input_shape=(60, 200, 3), num_classes=5)
    model.summary()