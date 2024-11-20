import numpy as np
import cv2

def blur_and_threshold(image, eps=1e-7):
    blur = cv2.medianBlur(image, 5)
    foreground = image.astype("float") - blur

    foreground[foreground > 0] = 0

    min_val = np.min(foreground)
    max_val = np.max(foreground)
    foreground = (foreground - min_val) / (max_val - min_val + eps)

    return foreground