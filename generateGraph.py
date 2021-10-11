import json as j
import matplotlib.pyplot as plt

with open("./Sample Results/Sample.json") as f:
    data = j.load(f)

zLabel = data['labels']
xAxis = data['xaxis']
yAxis = data['yaxis']

plt.bar(xAxis, yAxis, tick_label = zLabel, width = 0.1, color = ['blue'])
plt.xlabel("Time")
plt.ylabel("Condidence")
plt.title("Report")
plt.show()


