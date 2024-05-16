import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import pickle
import pathlib
data_dir = pathlib.Path("fyp test data set")

leaf_images_dict = {
    'almond': list(data_dir.glob('almond/*')),
    'hibiscus': list(data_dir.glob('hibiscus/*')),
    'java plum': list(data_dir.glob('java plum/*')),
    'money plant': list(data_dir.glob('money plant/*')),
    'rose': list(data_dir.glob('rose/*')),
    'guava': list(data_dir.glob('guava/*')),
    'jasmine': list(data_dir.glob('jasmine/*')),
    'lemon': list(data_dir.glob('lemon/*')),
    'mango': list(data_dir.glob('mango/*')),
    'sapota': list(data_dir.glob('sapota/*')),
}
leaf_labels_dict = {
    'almond': 0,
    'hibiscus': 1,
    'java plum': 2,
    'money plant': 3,
    'rose': 4,
    'guava': 5,
    'jasmine': 6,
    'lemon': 7,
    'mango': 8,
    'sapota': 9
}

X, y = [], []

for leaf_name, images in leaf_images_dict.items():
    for image in images:
        img = cv2.imread(str(image))
        resized_img = cv2.resize(img, (100, 100))
        X.append(resized_img)
        y.append(leaf_labels_dict[leaf_name])


X = np.array(X)
y = np.array(y)

dataset_vkr_10p = {
    "X": X,
    "y": y
}

with open('datasets_pickle/dataset_vkr_10p.pickle', 'wb') as f:
    pickle.dump(dataset_vkr_10p, f)

dataaug1 = Sequential([
    keras.layers.RandomZoom(0.1, input_shape=(100, 100, 3)),
    keras.layers.RandomRotation(0.4)
])
dataaug2 = Sequential([
    keras.layers.RandomZoom(0.3, input_shape=(100, 100, 3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomFlip("horizontal")
])

X1 = dataaug1(X).numpy().astype("uint8")

up_X = np.concatenate((X, X1))
up_y = np.concatenate((y, y))

X1 = dataaug2(up_X).numpy().astype("uint8")

up_X = np.concatenate((up_X, X1))
up_y = np.concatenate((up_y, up_y))

dataset_vkr_10p = {
    "X": up_X,
    "y": up_y
}

with open('datasets_pickle/10p_after-data-aug.pickle', 'wb') as f:
    pickle.dump(dataset_vkr_10p, f)

infile = open("datasets_pickle/10p_after-data-aug.pickle", "rb")
new_dataset = pickle.load(infile)
infile.close()

X = new_dataset["X"]
y = new_dataset["y"]

dataaug3 = Sequential([
    keras.layers.RandomZoom(0.3, input_shape=(100, 100, 3)),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomFlip("horizontal")
])
X1 = dataaug3(X).numpy().astype("uint8")
X = np.concatenate((X, X1))
y = np.concatenate((y, y))

dataset_vkr_10p = {
    "X": X,
    "y": y
}
with open('datasets_pickle/10p_3000.pickle', 'wb') as f:
    pickle.dump(dataset_vkr_10p, f)
