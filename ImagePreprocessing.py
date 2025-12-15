import os
import cv2
import numpy as np

TRAIN_PATH = r"C:\Users\user\Desktop\ML Project\Image Dataset\dataset\fruits-360_original-size\fruits-360-original-size\Training"
IMG_SIZE = 64

CLASSES = ["Banana", "Cucumber", "Apple", "Peach", "Tomato"]

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray))
    norm_uint8 = np.uint8(norm * 255)

    equalized = cv2.equalizeHist(norm_uint8)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(equalized)

    filtered = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    resized = cv2.resize(filtered, (IMG_SIZE, IMG_SIZE))
    resized = resized / 255.0

    return resized.flatten()

def load_dataset_from_classes(base_path, classes):
    X, y = [], []
    class_labels = {cls: idx for idx, cls in enumerate(classes)}

    for class_name in classes:
        print(f"Loading class: {class_name}")
        for folder in os.listdir(base_path):
            if not folder.startswith(class_name):
                continue

            folder_path = os.path.join(base_path, folder)
            if not os.path.exists(folder_path):
                continue

            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                try:
                    features = preprocess_image(img_path)
                    X.append(features)
                    y.append(class_labels[class_name])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue

    return np.array(X), np.array(y)

print("Loading data from TRAINING folder...")
X, y = load_dataset_from_classes(TRAIN_PATH, CLASSES)

print("Total images:", X.shape[0])
print("Feature vector size:", X.shape[1])

print("Combining features and labels into one CSV...")

y = y.reshape(-1, 1)
full_data = np.hstack((X, y))

np.savetxt("dataset_full.csv", full_data, delimiter=",")

print("Single CSV file saved as: dataset_full.csv")
print("Full dataset shape:", full_data.shape)
