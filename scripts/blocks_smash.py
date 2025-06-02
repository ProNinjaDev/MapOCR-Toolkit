import numpy as np



def split_into_blocks(image):
    height, weight = image.shape[:2]

    horizontal_border = height // 2
    vertical_border = weight // 2

    # Два портретных блока
    left_top_block = image[:horizontal_border, :vertical_border]
    right_top_block = image[:horizontal_border, vertical_border:]

    # Альбомный блок
    left_bottom_block = image[horizontal_border:, :vertical_border]
    right_bottom_block = image[horizontal_border:, vertical_border:]

    return left_top_block, right_top_block, left_bottom_block, right_bottom_block

def merge_into_image(left_top_block, right_top_block, left_bottom_block, right_bottom_block, image_size):
    height, weight = image_size

    horizontal_border = height // 2
    vertical_border = weight // 2

    # Пустое изображение
    image = np.zeros((height, weight), dtype="uint8")

    image[:horizontal_border, :vertical_border] = left_top_block
    image[:horizontal_border, vertical_border:] = right_top_block
    image[horizontal_border:, :vertical_border] = left_bottom_block
    image[horizontal_border:, vertical_border:] = right_bottom_block


    return image