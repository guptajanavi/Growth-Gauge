import cv2
import numpy as np
import json as j
import matplotlib.pyplot as plt

with open("./Sample Results/Sample.json") as f:
    data = j.load(f)

def textMode(img, text, location, fontsize):
    img = cv2.putText(img, text, location, 4, fontsize, (0,0,0), 1, cv2.LINE_AA)

def picMode(path, size, location):
    img = cv2.imread(path)
    img = cv2.resize(img, (size))
    h,w,d = img.shape
    y,x = location
    background[y:y+h, x:x+w] = img
    return img

background = np.zeros([800,900,3],dtype=np.uint8)
background.fill(255) # white background

textMode(background, "Growth Gauge", (280, 40), 1)
textMode(background, "Report", (340, 80), 1)

picMode("./Resized Logo.png", (100, 98), (15,780))
picMode("./Sample Results/img.jpg", (224, 224), (100,25))

textMode(background, "Name: ", (280, 150), 0.70)
textMode(background, str(data['name']), (415, 150), 0.70)
textMode(background, "Age: ", (280, 190), 0.70)
textMode(background, str(data['age']), (415, 190), 0.70)
textMode(background, "Gender: ", (280, 230), 0.70)
textMode(background, str(data['gender']), (415, 230), 0.70)
textMode(background, "Result: ", (280, 270), 0.70)
textMode(background, str(data['result']), (415, 270), 0.70)

picMode("./Sample Results/Figure_1.png", (880, 250), (350,15))
textMode(background, "*Values above the x axis indicate detection of anaemia.", (400, 605), 0.40)
textMode(background, "Values below the x axis indicate no detection of anaemia", (407, 618), 0.40)

textMode(background, "Recommendations to increase or maintain iron levels in blood:", (25, 680), 0.55)
textMode(background, "- Consume local foods like dates, nuts, leafy greens, beans, beetroot and jaggery.", (25, 700), 0.55)
textMode(background, "- Avoid tea, coffee, dairy, grapes and other tannin rich food.", (25, 720), 0.55)
textMode(background, "- Consult a doctor if you have yellow skin, irregular heartbeat or cold hands and feet.", (25, 740), 0.55)

textMode(background, "Growth Gauge- An initiative to eradicate Anaemia by Janavi Gupta", (165, 780), 0.40)

cv2.imshow('Result', background)
cv2.imwrite("Sample Results/FinalResult.png", background)
cv2.waitKey(0)
cv2.destroyAllWindows()
