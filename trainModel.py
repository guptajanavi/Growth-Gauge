#Train Model For Eyes

import cv2 as cv
import numpy as np
from PIL import Image
import os 

path = "./Dataset/"

recognizer = cv.face.LBPHFaceRecognizer_create()
detector = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_eye.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    eyeSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array (PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split("_")[1])
        eyes = detector.detectMultiScale(img_numpy)

        for (x,y,w,h) in eyes:
            eyeSamples.append(img_numpy[y:y+h, x:x+h])
            ids.append(id)
    return eyeSamples, ids

print ('Training on prvoided eye dataset. Please wait')
eyes, ids = getImagesAndLabels(path)
recognizer.train(eyes, np.array(ids))
recognizer.write("Trainer.yml")

print("Model training is completed on {0} sample types". format(len(np.unique(ids))))
