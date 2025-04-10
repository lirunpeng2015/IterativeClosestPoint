import matplotlib.pyplot as plt
import numpy as np

# open the file containing the lidar scan data
# change to the file which you want to plot
file = open('lidar-scans/lidar-box-scan.txt', 'r')

# skip the first line of file
file.readline()

# get the number of points contained in the file from the COUNT
count_line = file.readline()
count_line_split = count_line.split('=')
num_of_points = int(count_line_split[1])
# print('Number of points in file: ', num_of_points)

# skip third line of file
file.readline()

# lists to contain x, y values
x = []
y = []

for i in range (0, 300):
    line = file.readline()
    numbers = line.split()
    x.append(float(numbers[1]) * np.cos(float(numbers[0])))
    y.append(float(numbers[1]) * np.sin(float(numbers[0])))



plt.scatter(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.show()