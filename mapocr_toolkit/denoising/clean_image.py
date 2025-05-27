from config import cleaning_image_config as config
from image_processing.blur_and_threshold import blur_and_threshold
from imutils import paths
import pickle
import random
import cv2
import matplotlib.pyplot as plt




def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

def process_image(image, model):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig = gray.copy()
    gray = cv2.copyMakeBorder(gray, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    gray = blur_and_threshold(gray)

    roi_features = [
        gray[y:y + 5, x:x + 5].flatten()
        for y in range(gray.shape[0] - 4)
        for x in range(gray.shape[1] - 4)
    ]

    pixels = model.predict(roi_features)
    output = (pixels.reshape(orig.shape) * 255).astype("uint8") # Преобразование одномерного в двумерный массив

    return orig, output


def visualize_results(images, titles, fig_size=(12, 6)):
    plt.figure(figsize=fig_size)
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    testing_path = "denoising-dirty-documents/test"
    sample_size = 10

    print("[INFO] Loading model...")
    model = load_model(config.MODEL_PATH)

    image_paths = list(paths.list_images(testing_path))
    random.shuffle(image_paths)
    image_paths = image_paths[:sample_size]

    try:
        for image_path in image_paths:
            print(f"[INFO] Processing {image_path}")
            image = cv2.imread(image_path)

            # Process the image
            orig, output = process_image(image, model)

            # Visualize original and processed images side by side
            visualize_results([orig, output], ["Original", "Output"])
    except KeyboardInterrupt:
        print("\n[INFO] Program interrupted by user. Exiting the program...")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
    finally:
        print("[INFO] Cleaning up resources...")


if __name__ == "__main__":
    main()