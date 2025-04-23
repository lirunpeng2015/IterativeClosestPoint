import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# Specify the folder where your lidar scan files are located
folder_path = 'lidar-scans/outside-hallway'

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
            if len(numbers) < 2:
                continue
            angle_rad = np.radians(float(numbers[0]))
            x.append(float(numbers[1]) * np.cos(angle_rad))
            y.append(float(numbers[1]) * np.sin(angle_rad))

    return x, y

# ICP Algorithm
def icp_2d(x_ref, y_ref, x_src, y_src, max_iterations=500, tolerance=1e-6):
    ref_points = np.vstack((x_ref, y_ref)).T
    src_points = np.vstack((x_src, y_src)).T

    R_total = np.eye(2)
    t_total = np.zeros((2, 1))

    prev_error = float('inf')

    for i in range(max_iterations):
        tree = cKDTree(ref_points)
        distances, indices = tree.query(src_points)
        closest_ref = ref_points[indices]

        centroid_src = np.mean(src_points, axis=0)
        centroid_ref = np.mean(closest_ref, axis=0)

        src_centered = src_points - centroid_src
        ref_centered = closest_ref - centroid_ref

        H = src_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        t = centroid_ref.reshape(2, 1) - R @ centroid_src.reshape(2, 1)

        src_points = (R @ src_points.T).T + t.T

        # Accumulate transformation
        R_total = R @ R_total
        t_total = R @ t_total + t

        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    x_aligned, y_aligned = src_points[:, 0].tolist(), src_points[:, 1].tolist()
    return x_aligned, y_aligned, R_total, t_total

# Main loop to process multiple files
def process_lidar_files():
    lidar_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    
    x_map, y_map = [], []

    poses = [np.eye(2)]  # Rotation matrices
    translations = [np.zeros((2, 1))]  # Translation vectors

    x_ref, y_ref = [], []

    for i, file_name in enumerate(lidar_files):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name}...")

        x_src, y_src = to_cartesian(file_path)

        if i == 0:
            x_ref, y_ref = x_src, y_src

            # First scan goes directly into the map
            x_map += x_src
            y_map += y_src
            continue

        # Align current scan to reference
        x_aligned, y_aligned, R_rel, t_rel = icp_2d(x_ref, y_ref, x_src, y_src)

        # Get previous global pose
        R_prev = poses[-1]
        t_prev = translations[-1]

        # Compute current global pose
        R_curr = R_rel @ R_prev
        t_curr = R_rel @ t_prev + t_rel

        poses.append(R_curr)
        translations.append(t_curr)

        # Transform current scan to global coordinates
        aligned_points = np.vstack((x_aligned, y_aligned))
        transformed_points = R_curr @ aligned_points + t_curr

        x_trans, y_trans = transformed_points[0, :], transformed_points[1, :]

        x_map += x_trans.tolist()
        y_map += y_trans.tolist()

        # Optional: update reference to current scan (or you could use the map itself)
        x_ref, y_ref = x_aligned, y_aligned

        # Print transformation
        print(f"Rotation matrix:\n{R_curr}")
        print(f"Translation vector:\n{t_curr}")

    # Plot final map
    plt.figure()
    plt.scatter(x_map, y_map, s=1, color='blue')
    plt.title('Final Aligned LIDAR Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

    # Save map
    with open('final_map.txt', 'w') as f:
        for xi, yi in zip(x_map, y_map):
            f.write(f"{xi:.3f} {yi:.3f}\n")

    # Save poses
    with open('poses.txt', 'w') as f:
        for R, t in zip(poses, translations):
            angle = np.arctan2(R[1, 0], R[0, 0])
            f.write(f"{t[0, 0]:.3f} {t[1, 0]:.3f} {angle:.6f}\n")

# Run it
process_lidar_files()
