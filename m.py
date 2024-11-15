import cv2
import os
import numpy as np

def load_data(data_path):
    images = []
    labels = []
    label_map = {}

    label_id = 0
    for person_name in os.listdir(data_path):
        person_path = os.path.join(data_path, person_name)
        label_map[label_id] = person_name

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0  # Chuẩn hóa
            images.append(img)
            labels.append(label_id)

        label_id += 1

    return np.array(images), np.array(labels), label_map

X, y, label_map = load_data("data")
np.save("X.npy", X)
np.save("y.npy", y)
np.save("label_map.npy", label_map)
