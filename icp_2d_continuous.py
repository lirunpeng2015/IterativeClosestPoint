import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import time

# Specify the folder where your lidar scan files are located
folder_path = 'lidar-scans/'

# Function to convert polar coordinates to cartesian (x, y)
def to_cartesian(file_path):
    with open(file_path, 'r') as file:
        file.readline()  # Skip the first line
        count_line = file.readline()  # Read the second line for the number of points
        count_line_split = count_line.split('=')
        num_of_points = int(count_line_split[1])  # Get number of points

        file.readline()  # Skip the third line

        x = []
        y = []

        for _ in range(num_of_points):
            line = file.readline()
            numbers = line.split()
            angle_rad = np.radians(float(numbers[0]))
            x.append(float(numbers[1]) * np.cos(angle_rad))
            y.append(float(numbers[1]) * np.sin(angle_rad))

    return x, y

# ICP Algorithm
def icp_2d(x_ref, y_ref, x_src, y_src, max_iterations=500, tolerance=1e-6):
    ref_points = np.vstack((x_ref, y_ref)).T  # Shape (N, 2)
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

        # Visualization
        plt.clf()
        plt.scatter(x_ref, y_ref, color='blue')
        plt.scatter(x_aligned, y_aligned, color='orange')
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

    x_aligned, y_aligned = src_points[:, 0].tolist(), src_points[:, 1].tolist()

    return x_aligned, y_aligned, R_total, t_total

# Main loop to process multiple files
def process_lidar_files():
    # Get all LIDAR scan files in the directory
    lidar_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    # Initialize reference point cloud (empty at first)
    x_ref, y_ref = [], []

    for file_name in lidar_files:
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name}...")

        # Convert source LIDAR scan to Cartesian coordinates
        x_src, y_src = to_cartesian(file_path)

        # If there is no reference yet, use the current file as the reference
        if len(x_ref) == 0 and len(y_ref) == 0:
            x_ref, y_ref = x_src, y_src
            continue  # Skip ICP if the reference is just set

        # Perform ICP to align current scan (source) with the reference
        x_aligned, y_aligned, R_total, t_total = icp_2d(x_ref, y_ref, x_src, y_src)

        # Update the reference for the next iteration
        x_ref = x_ref + x_aligned
        y_ref = y_ref + y_aligned

        # Print the accumulated transformation (for debugging purposes)
        print(f"Transformation (R):\n{R_total}")
        print(f"Transformation (t):\n{t_total}")

    # Final plot after all files have been processed
    plt.ioff()  # Turn off interactive mode
    plt.scatter(x_ref, y_ref, color='blue')
    plt.title('Final Aligned Point Cloud')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()

# Run the main function to process the files
process_lidar_files()
