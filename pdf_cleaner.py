from pdf2image import convert_from_path
import os
from config import cleaning_image_config as config
from clean_image import load_model, process_image
import cv2




def convert_pdf_to_images(pdf_path, output_folder, dpi=300):
    print("[INFO] Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=dpi)
    image_paths = []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, page in enumerate(pages):
        image_path = os.path.join(output_folder, f"page_{i + 1}.png")
        page.save(image_path, "PNG")
        image_paths.append(image_path)

    print(f"[INFO] PDF converted into {len(image_paths)} images.")
    return image_paths

def process_pdf_pages(image_paths, model, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_path in image_paths:
        print(f"[INFO] Processing {image_path}")
        image = cv2.imread(image_path)

        # Очищаем изображение
        trash, output = process_image(image, model)

        # Сохраняем изображение
        output_path = os.path.join(output_folder, os.path.basename(image_path))
        cv2.imwrite(output_path, output)

    print(f"[INFO] All cleaned images saved to {output_folder}")

def main():
    pdf_path = "Cartography file.pdf"
    temp_images_folder = config.TEMP_PATH
    cleaned_images_folder = config.PDF_PATH

    print("[INFO] Loading model...")
    model = load_model(config.MODEL_PATH)

    try:
        image_paths = convert_pdf_to_images(pdf_path, temp_images_folder)

        process_pdf_pages(image_paths, model, cleaned_images_folder)


    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        print("[INFO] Cleaning up resources...")


if __name__ == "__main__":
    main()