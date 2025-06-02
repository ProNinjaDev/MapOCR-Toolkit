import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

def prepare_cnn_data(data_items, class_to_int_map, target_size=(60, 200), val_split_size=0.2, random_state_value=42):

    images = []
    raw_class_labels = []

    for img_path, _, str_label in data_items:
        if not os.path.exists(img_path):
            print(f"[WARNING] Image file not found: {img_path}")
            continue
        try:
            img = load_img(img_path, target_size=target_size)
            
            img_array = img_to_array(img)
            
            img_array = img_array / 255.0
            
            images.append(img_array)
            raw_class_labels.append(str_label)
        except Exception as e:
            print(f"[ERROR] Could not load or process image {img_path}: {e}")
            continue
    
    if not images:
        empty_np_array = np.array([])
        return (empty_np_array, empty_np_array), (empty_np_array, empty_np_array), None

    x_images_np = np.array(images)
    
    int_class_labels = [class_to_int_map[label] for label in raw_class_labels]
    
    num_classes_overall = len(class_to_int_map)
    y_one_hot_labels = to_categorical(np.array(int_class_labels), num_classes=num_classes_overall)


    x_train, x_val, y_train, y_val = train_test_split(x_images_np, 
                                                      y_one_hot_labels, 
                                                      test_size=val_split_size, 
                                                      random_state=random_state_value,
                                                      stratify=y_one_hot_labels)

    processing_info = {
        'target_size': target_size,
        'class_to_int_map': class_to_int_map,
        'int_to_class_map': {int_label: str_label for str_label, int_label in class_to_int_map.items()}
    }
    
    return (x_train, y_train), (x_val, y_val), processing_info

if __name__ == '__main__':
    pass