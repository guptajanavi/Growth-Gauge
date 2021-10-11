import os, os.path
import cv2 as cv

inPath = "./Anaemic Images/"
outPath = "./savedImages/"

c = 0
for files in os.listdir(inPath):
    img  = cv.imread(inPath + files)
    cv.imshow("live", img)
    cv.waitKey(27)
    name = "img_1_"+str(c)+".jpg"
    cv.imwrite(outPath+name, img)
    print("processed on", c, "no of images")    
    c+=1
