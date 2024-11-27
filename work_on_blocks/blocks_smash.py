import numpy as np



def split_into_blocks(image):
    height, weight = image.shape[:2]

    horizontal_border = height // 2
    vertical_border = weight // 2

    # Два портретных блока
    left_block = image[:horizontal_border, :vertical_border]
    right_block = image[:horizontal_border, vertical_border:]

    # Альбомный блок
    bottom_block = image[horizontal_border:, :]

    return left_block, right_block, bottom_block

def merge_into_image(left_block, right_block, bottom_block, image_size):
    height, weight = image_size

    horizontal_border = height // 2
    vertical_border = weight // 2

    # Пустое изображение
    image = np.zeros((height, weight), dtype="uint8")

    image[:horizontal_border, :vertical_border] = left_block
    image[:horizontal_border, vertical_border:] = right_block
    image[horizontal_border:, :] = bottom_block

    return image