import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# convert polar coordinates to cartesian coordinates
# return np.arrays for x and y values
def to_cartesian(file_path):
    with open(file_path, 'r') as file:
        file.readline()  # Skip header
        count_line = file.readline()
        num_of_points = int(count_line.split('=')[1])
        file.readline()  # Skip line

        x, y = [], []
        for _ in range(num_of_points):
            line = file.readline()
            numbers = line.split()
            angle_rad = np.radians(float(numbers[0]))
            distance = float(numbers[1])

            if distance > 10000:  # Filter outliers
                continue

            x.append(distance * np.cos(angle_rad))
            y.append(distance * np.sin(angle_rad))

    return np.array(x), np.array(y)

# performs icp for two point clouds
def icp_2d(x_ref, y_ref, x_src, y_src, max_iterations=100, tolerance=1e-6):
    ref_points = np.vstack((x_ref, y_ref)).T
    src_points = np.vstack((x_src, y_src)).T

    # Initial transformation
    R = np.eye(2)
    t = np.zeros((2, 1))
    prev_error = float('inf')

    for i in range(max_iterations):
        # Find closest points
        tree = cKDTree(ref_points)
        distances, indices = tree.query(src_points)
        closest_ref = ref_points[indices]

        # Compute centroids
        centroid_src = np.mean(src_points, axis=0)
        centroid_ref = np.mean(closest_ref, axis=0)

        # Center the point sets
        src_centered = src_points - centroid_src
        ref_centered = closest_ref - centroid_ref

        # Compute optimal rotation
        H = src_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R_opt = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R_opt) < 0:
            Vt[1, :] *= -1
            R_opt = Vt.T @ U.T

        # Compute translation
        t_opt = centroid_ref.reshape(2, 1) - R_opt @ centroid_src.reshape(2, 1)

        # Apply transformation to source points
        src_points = (R_opt @ src_points.T).T + t_opt.T

        # Update transformation
        R = R_opt @ R
        t = R_opt @ t + t_opt

        # Check convergence
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return src_points[:, 0], src_points[:, 1], R, t

# apply transformation to a point cloud
def transform_points(points, R, t):
    """Apply transformation (R, t) to points"""
    return (R @ points.T + t).T
