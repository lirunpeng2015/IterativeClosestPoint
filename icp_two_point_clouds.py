# This script performs iterative closets point for two point clouds


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import time

reference_file_path = 'lidar-scans/classroom/pos1-1.txt'
source_file_path = 'lidar-scans/classroom/pos2-1.txt'

ref_pc_file = open(reference_file_path, 'r')
src_pc_file = open(source_file_path, 'r')


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


# for i in range(len(x_src)):
#     smallest_distance = -1

#     for j in range(len(x_ref)):
#         distance = np.sqrt(np.pow((x_src[i] - x_ref[j]), 2) + np.pow((y_src[i] - y_ref[j]), 2))

#         if smallest_distance == -1:
#             smallest_distance = distance
#             smallest_distance_index = j

#         elif smallest_distance > distance:
#             smallest_distance = distance
#             smallest_distance_index = j


plt.ion()

plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Lidar Point-Cloud plot")

plt.scatter(x_ref, y_ref, color = 'blue')
plt.scatter(x_src, y_src, color = 'red')
plt.draw()


#---------------------------------------#
#           algorithm for ICP           #
#---------------------------------------#


def icp_2d(x_ref, y_ref, x_src, y_src, max_iterations=100, tolerance=1e-6):

    # Convert input lists to numpy arrays
    ref_points = np.vstack((x_ref, y_ref)).T  # Shape (N, 2)
    print(ref_points)
    src_points = np.vstack((x_src, y_src)).T  # Shape (M, 2)

    # Initialize transformation: rotation matrix R and translation vector t
    R_total = np.eye(2)
    t_total = np.zeros((2, 1))

    prev_error = float(1)

    for i in range(max_iterations):

        # Step 1: Find closest points
        tree = cKDTree(ref_points)
        distances, indices = tree.query(src_points)
        closest_ref = ref_points[indices]

        # Step 2: Compute centroids
        centroid_src = np.mean(src_points, axis=0)
        centroid_ref = np.mean(closest_ref, axis=0)

        # Step 3: Subtract centroids
        src_centered = src_points - centroid_src
        ref_centered = closest_ref - centroid_ref

        # Step 4: Compute optimal rotation using SVD
        H = src_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Ensure R is a proper rotation (det(R)=1)
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_ref.reshape(2, 1) - R @ centroid_src.reshape(2, 1)

        # Step 5: Apply transformation
        src_points = (R @ src_points.T).T + t.T

        x_aligned, y_aligned = src_points[:, 0].tolist(), src_points[:, 1].tolist()

        plt.clf()
        plt.scatter(x_ref, y_ref, color='blue')
        plt.scatter(x_aligned, y_aligned)
        plt.draw()

        plt.pause(0.5)

        # Accumulate transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        # Step 6: Check convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Final transformed source points
    x_aligned, y_aligned = src_points[:, 0].tolist(), src_points[:, 1].tolist()

    return x_aligned, y_aligned, R_total, t_total

x_aligned, y_aligned, R_total, t_total = icp_2d(x_ref, y_ref, x_src, y_src)

x_ref = x_ref + x_aligned
y_ref = y_ref + y_aligned

print(len(x_ref))
print(len(y_ref))

print(R_total)
print(t_total)

#plt.scatter(x_ref, y_ref, color='blue')
#plt.scatter(x_aligned, y_aligned, color='orange')

plt.ioff()
plt.show()