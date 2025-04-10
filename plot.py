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

# convert angle + distance to (x,y) and add to lists
for i in range (0, num_of_points):
    line = file.readline()
    numbers = line.split()

    # np.cos/np.sin expects angle in radians
    angle_rad = np.radians(float(numbers[0]))
    x.append(float(numbers[1]) * np.cos(angle_rad))
    y.append(float(numbers[1]) * np.sin(angle_rad))
    print('Angle: ', numbers[0], ' Distance: ', numbers[1], ' X-value: ',
           (float(numbers[1]) * (float(np.cos(float(numbers[0]))))), ' Y-value: ', (float(numbers[1]) * (float(np.sin(float(numbers[0]))))))


file.close()

plt.scatter(x, y)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Simple Line Plot")
plt.show()