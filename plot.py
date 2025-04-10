import matplotlib.pyplot as plt
import numpy as np

file = open('lidar-scans/lidar-box-scan.txt', 'r')

file.readline()
file.readline()
file.readline()

x = []
y = []

for i in range (0, 300):
    line = file.readline()
    numbers = line.split()
    x.append(float(numbers[1]) * np.cos(float(numbers[0])))
    y.append(float(numbers[1]) * np.sin(float(numbers[0])))



plt.plot(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.show()