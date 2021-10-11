#Model Testing

import numpy as np
import cv2 as cv
import time as t
from datetime import datetime
import json
import os, os.path

userName = input("Enter your name")
userAge = int(input("Enter your age"))
userGender = input("Enter your gender: Please enter 'M' for male, 'F' for female and 'O' for other")

def getTime():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    return timestamp

cap = cv.VideoCapture(0)
print('Loading face model')
faceCas = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
print('Done')
print('Loading eye model')
eyeCas = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_eye.xml")
print('Done')

recognizer = cv.face.LBPHFaceRecognizer_create()
fname = "./Trainer/Trainer.yml"
if not os.path.isfile(fname):
    print('No such file found')
    exit(0)

recognizer.read(fname)
id = 0
name = ['None', 'Anaemic', 'Non-Anaemic']

start = getTime()
xAxis = [] #Time
yAxis = [] #Amplitude
zLabel = []

while True:    
    ret, img = cap.read()
    if ret == True:
        faces = faceCas.detectMultiScale(img, 1.3, 5)
        for (x,y,w,h) in faces:
            cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            #print('Face count is', faces.shape[0])

            roi_face = img[y:y+h, x:x+w]
            eyes = eyeCas.detectMultiScale(roi_face)
        
            for (a,b,c,d) in eyes:                
                roi_eye = roi_face[b:b+d, a:a+c]
                cv.imwrite("./Sample Results/Img.jpg", roi_eye)
                cv.rectangle(roi_face, (a,b), (a+c, b+d), (0,255,0), 2)
                roi_eye = cv.resize(roi_eye, (224, 224))
                    
                roi_eye = cv.cvtColor(roi_eye, cv.COLOR_BGR2GRAY)
                id, conf = recognizer.predict(roi_eye)
                id = name[id]
                if x//2 < a:
                    if str(id) == 'Anaemic':
                        cv.putText (img, str(id), (0,20), 4, 1, (0,0,255), 2)
                    else:
                        cv.putText (img, str(id), (0,20), 4, 1, (0,255,0), 2)   
                    print(id, int(100.00 - conf))
                if x//2 > a:
                    if str(id) == 'Anaemic':
                        cv.putText (img, str(id), (450,20), 4, 1, (0,0,255), 2)
                    else:
                        cv.putText (img, str(id), (450,20), 4, 1, (0,255, 0), 2)
                    print(id, int(100.00 - conf))
                    
                
                xAxis.append(getTime())
                yAxis.append(100 - int(conf))
                zLabel.append(id)
                
                #cv.imshow('Result', roi_eye)
                
        #cv.imshow('ROI face', roi_face)
                
        cv.imshow("Img," , img)
        
        k = cv.waitKey(27) & 0xff
        if k == ord('q'):
            break
        
            

data = {
    'name': userName,
    'age': userAge,
    'result': id,
    'gender': userGender,
    'xaxis': xAxis,
    'yaxis': yAxis,
    'labels': zLabel
    }
json_object = json.dumps(data, indent = 6)
with open("./Sample Results/Sample.json", "w") as outFile:
    outFile.write(json_object)

print("File write completed")

print(xAxis, yAxis)
cap.release()
cv.destroyAllWindows()
