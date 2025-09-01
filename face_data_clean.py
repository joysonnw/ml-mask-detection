import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

img_size = 128
data = []
labels = []

with_mask_path = r"C:\Users\nwjoy\Downloads\archive\data\with_mask"
without_mask_path = r"C:\Users\nwjoy\Downloads\archive\data\without_mask"

def load_images(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (img_size, img_size))
            data.append(img)
            labels.append(label)

load_images(with_mask_path, 1)
load_images(without_mask_path, 0)

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

