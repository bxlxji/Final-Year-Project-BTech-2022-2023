from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential
import pickle

infile = open("datasets_pickle/10p_3000.pickle", "rb")
new_dataset = pickle.load(infile)
infile.close()

X = new_dataset["X"]
y = new_dataset["y"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train_scaled = X_train / 255
X_test_scaled = X_test / 255

num_classes = 10
#
model1 = Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.evaluate(X_test_scaled, y_test)

model2 = Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3),
                  input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=32, kernel_size=(3, 3),
                  input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=64, kernel_size=(3, 3),
                  input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.fit(X_train_scaled, y_train, epochs=20)

model2.evaluate(X_test_scaled, y_test)

with open('models/model-kk-ta85_71.pickle', 'wb') as f:
    pickle.dump(model2, f)

num_classes = 10

model3 = Sequential([
    layers.Conv2D(filters=16, kernel_size=(3, 3),
                  input_shape=(100, 100, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(filters=256, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
model3.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model3.fit(X_train_scaled, y_train, epochs=25)

model3.evaluate(X_test_scaled, y_test)

with open('models/model-kk-10p-ta90_48.pickle', 'wb') as f:
    pickle.dump(model3, f)
