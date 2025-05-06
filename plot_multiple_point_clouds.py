# This script plots multiple point clouds onto one scatter plot


import matplotlib.pyplot as plt
import numpy as np

num_of_data_files = 3

file_path_1 = 'lidar-scans/lidar-floor-scan-1.txt'
file_path_2 = 'lidar-scans/lidar-floor-scan-2.txt'
file_path_3 = 'lidar-scans/lidar-floor-scan-3.txt'

for i in range(num_of_data_files):

    file_path = ''

    if i == 0:
        file_path = file_path_1
    elif i == 1:
        file_path = file_path_2
    elif i == 2:
        file_path = file_path_3


    # open the file containing the lidar scan data
    # change to the file which you want to plot
    file = open(file_path, 'r')

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
    for i in range (num_of_points):
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
plt.title("Lidar Point-Cloud plot")
plt.show()