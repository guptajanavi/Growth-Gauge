import json
import time
import os, os.path

name = input("Enter your name")
age = int(input("Enter your age"))
gender = input("Enter your gender: Please enter 'M' for male, 'F' for female and 'O' for other")

time = [] #works as x axis, used to obtain real time iterations
amp = [] #works as y axis, confidence
id = "Normal"

data = {
    'name': name,
    'age': age,
    'result': id,
    'gender': gender,
    'xaxis': time,
    'yaxis': amp
    }
json_object = json.dumps(data, indent = 6)
with open("sample.json", "x") as outFile:
    outFile.write(json_object)

print("File write completed")



