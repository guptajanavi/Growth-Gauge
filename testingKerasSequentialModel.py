import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
from keras.models import load_model
import numpy as np

model = load_model("./anaemicvsnonanaemic.h5")
model.summary()

image_size = (180, 180)
img = keras.preprocessing.image.load_img("./newModelTestInput2/img_1_0.jpg", target_size = image_size)
print(img)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = np.reshape(img_array,[1,180,180,3])
predictions = model.predict(img_array)
score = predictions[0]

print("This image is %.2f anaemic and %.2f percent non-anaemic."% (100 * (1- score), 100 * score))

