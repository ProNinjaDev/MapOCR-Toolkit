import os

BASE_PATH = "denoising-dirty-documents"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
CLEANED_PATH = os.path.join(BASE_PATH, "train_cleaned")
TEST_PATH = os.path.join(BASE_PATH, "test")
MODEL_PATH = "cleaner.pickle"
FEATURES_PATH = "features.csv"
SAMPLE_PROB = 0.02

PDF_PATH = "cleared_images"
TEMP_PATH = "temp_images"