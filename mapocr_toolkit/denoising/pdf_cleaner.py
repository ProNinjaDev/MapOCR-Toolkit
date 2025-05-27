import os
from config import cleaning_image_config as config
from clean_image import load_model, process_image
import cv2
import fitz
import numpy as np
from work_on_blocks.blocks_smash import split_into_blocks, merge_into_image




# При dpi > 50 качество изображение, которое пока мне не нужно
def convert_pdf_to_images(pdf_path, output_folder, dpi=75):
    print("[INFO] Converting PDF to images...")
    pdf_document = fitz.open(pdf_path)
    image_paths = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image = page.get_pixmap(dpi=dpi)

        image_path = os.path.join(output_folder, f"page_{page_number + 1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    print(f"[INFO] PDF converted into {len(image_paths)} images.")
    return image_paths

def process_pdf_pages(image_paths, model, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in image_paths:
        print(f"[INFO] Processing {image_path}")
        image = cv2.imread(image_path)
        left_top_block, right_top_block, left_bottom_block, right_bottom_block = split_into_blocks(image)

        # Очищаем изображение
        processed_left_top = process_image(left_top_block, model)[1]
        processed_right_top = process_image(right_top_block, model)[1]
        processed_left_bottom = process_image(left_bottom_block, model)[1]
        processed_right_bottom = process_image(right_bottom_block, model)[1]

        final_image = merge_into_image(processed_left_top, processed_right_top, processed_left_bottom, processed_right_bottom, image.shape[:2])

        # Проверка данных
        print(f"[DEBUG] output.shape: {final_image.shape}, output.dtype: {final_image.dtype}")
        if final_image.size == 0:
            raise ValueError("Output image is empty.")
        if np.all(final_image == 0):
            print(f"[WARNING] Output image for {image_path} is completely black.")
        if np.isnan(final_image).any():
            raise ValueError("Output image contains NaN values.")

        if not os.access(output_folder, os.W_OK):
            raise PermissionError(f"Cannot write to the folder: {output_folder}")
        # Сохраняем изображение
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, final_image)

        if os.path.exists(output_path):
            print(f"[INFO] Successfully saved: {output_path}")
        else:
            print(f"[ERROR] Failed to save image: {output_path}")

    print(f"[INFO] All cleaned images saved to {output_folder}")

def main():
    pdf_path = "megaTest.pdf"
    temp_images_folder = config.TEMP_PATH
    cleaned_images_folder = config.PDF_PATH

    print("[INFO] Loading model...")
    model = load_model(config.MODEL_PATH)

    try:
        image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)

        process_pdf_pages(image_paths, model, cleaned_images_folder)

        print("[INFO] PDF cleaning process completed successfully.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        print("[INFO] Cleaning up resources...")


if __name__ == "__main__":
    main()