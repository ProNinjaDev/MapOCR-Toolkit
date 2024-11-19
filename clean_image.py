from config import cleaning_image_config as config
from image_processing.blur_and_threshold import blur_and_threshold
from imutils import paths
import pickle
import random
import cv2



testing_path = "denoising-dirty-documents/test"
sample_size = 10

model = pickle.loads(open(config.MODEL_PATH, "rb").read())

imagePaths = list(paths.list_images(testing_path))
random.shuffle(imagePaths)
imagePaths = imagePaths[:sample_size]

for imagePath in imagePaths:
    print("[INFO] processing {}".format(imagePath))
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orig = image.copy()
    image = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    image = blur_and_threshold(image)

    roiFeatures = []

    for y in range(0, image.shape[0]):
        for x in range(0, image.shape[1]):
            roi = image[y:y + 5, x:x + 5]
            (rH, rW) = roi.shape[:2]

            if rW != 5 or rH != 5:
                continue

            features = roi.flatten()
            roiFeatures.append(features)

    pixels = model.predict(roiFeatures)

    pixels = pixels.reshape(orig.shape)
    output = (pixels * 255).astype("uint8")

    cv2.imshow("Original", orig)
    cv2.imshow("Output", output)
    cv2.waitKey(0)