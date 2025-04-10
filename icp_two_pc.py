# This script performs iterative closets point for two point clouds


import matplotlib.pyplot as plt
import numpy as np

# reference point cloud file
ref_pc_file = open('lidar-scans/lidar-data.txt', 'r')

# source point cloud file
src_pc_file = open('lidar-scans/lidar-data-3.txt', 'r')


#----------------------------------------------------------#
#     gather cartesian coordinates from reference file     #
#----------------------------------------------------------#


# skip the first line of file
ref_pc_file.readline()

# get the number of points contained in the file from the COUNT
count_line = ref_pc_file.readline()
count_line_split = count_line.split('=')
num_of_points = int(count_line_split[1])

# skip third line of file
ref_pc_file.readline()

x_ref = []
y_ref = []

# convert angle + distance to (x,y) and add to lists
for i in range (num_of_points):
    line = ref_pc_file.readline()
    numbers = line.split()

    # np.cos/np.sin expects angle in radians
    angle_rad = np.radians(float(numbers[0]))
    x_ref.append(float(numbers[1]) * np.cos(angle_rad))
    y_ref.append(float(numbers[1]) * np.sin(angle_rad))

ref_pc_file.close()


#-------------------------------------------------------#
#     gather cartesian coordinates from source file     #
#-------------------------------------------------------#


# skip the first line of file
src_pc_file.readline()

# get the number of points contained in the file from the COUNT
count_line = src_pc_file.readline()
count_line_split = count_line.split('=')
num_of_points = int(count_line_split[1])

# skip third line of file
src_pc_file.readline()

x_src = []
y_src = []

# convert angle + distance to (x,y) and add to lists
for i in range (num_of_points):
    line = src_pc_file.readline()
    numbers = line.split()

    # np.cos/np.sin expects angle in radians
    angle_rad = np.radians(float(numbers[0]))
    x_src.append(float(numbers[1]) * np.cos(angle_rad))
    y_src.append(float(numbers[1]) * np.sin(angle_rad))

src_pc_file.close()


#--------------------------------------------------------------#
#     calculate distances between each point in the src pc     #
#--------------------------------------------------------------#


for i in range(len(x_src)):
    smallest_distance = -1

    for j in range(len(x_ref)):
        distance = np.sqrt(np.pow((x_src[i] - x_ref[j]), 2) + np.pow((y_src[i] - y_ref[j]), 2))

        if smallest_distance == -1:
            smallest_distance = distance
            smallest_distance_index = j

        elif smallest_distance > distance:
            smallest_distance = distance
            smallest_distance_index = j




plt.scatter(x_ref, y_ref)
plt.scatter(x_src, y_src)

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Lidar Point-Cloud plot")
plt.show()