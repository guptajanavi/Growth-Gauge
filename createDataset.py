import numpy as np
import cv2 as cv
import os

cap = cv.VideoCapture(0)
print('Loading face model')
faceCas = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
print('Done')
print('Loading eye model')
eyeCas = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_eye.xml")
print('Done')

print('Data entry begins here')
object_name = input('Enter class name to generate dataset')
user = int(input("Enter data value i.e [1 => disease ot 2 => normal]: "))
path = './Dataset/'
os.mkdir(path + object_name, mode= 0o666)

pathl = './Dataset/'+ object_name+ '/Left'
os.mkdir(pathl, mode= 0o666)

pathr = './Dataset/'+ object_name+ '/Right'
os.mkdir(pathr, mode= 0o666)

countl = 0
countr = 0
threshold = 30

while True:    
    ret, img = cap.read()
    faces = faceCas.detectMultiScale(img, 1.3, 5)
        
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        print('Face count is', faces.shape[0])

        roi_face = img[y:y+h, x:x+w]
        eyes = eyeCas.detectMultiScale(roi_face)
        
        for (a,b,c,d) in eyes:
            cv.rectangle(roi_face, (a,b), (a+c, b+d), (0,255,0), 2)
            roi_eye = roi_face[b:b+d, a:a+c]
            roi_eye = cv.resize(roi_eye, (224, 224))
            if x//2 < a and (countl < threshold):                
                cv.imshow('Left Eye', roi_eye)
                cv.imwrite(pathl + '/img_'+str(user)+"_"+str(countl)+'.jpg', roi_eye)
                countl += 1
                
            if x//2 > a and (countr < threshold):
                cv.imshow('Right Eye', roi_eye)
                cv.imwrite(pathr + '/img_'+str(user)+"_"+str(countr+30)+'.jpg', roi_eye)
                countr += 1
        print (countl, "   ", countr)
        cv.imshow('ROI face', roi_face)            
        cv.imshow("img," , img)
        k = cv.waitKey(27) & 0xff
        if k == ord('q') or (countl >= threshold) and (countr >= threshold):
            break   
        
cap.release()
cv.destroyAllWindows()

