import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

folder_path = 'lidar-scans/classroom'

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

def transform_points(points, R, t):
    """Apply transformation (R, t) to points"""
    return (R @ points.T + t).T

def process_lidar_files():
    lidar_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    if not lidar_files:
        print("No scan files found.")
        return

    # Load first scan as reference
    x_first, y_first = to_cartesian(os.path.join(folder_path, lidar_files[0]))
    
    # Initialize global map with first scan
    global_map_x, global_map_y = x_first.copy(), y_first.copy()
    
    # Store robot poses (x, y, theta)
    poses = [(0.0, 0.0, 0.0)]  # initial pose
    
    # Initialize global transformation as identity
    R_global = np.eye(2)
    t_global = np.zeros((2, 1))
    
    # Use the first scan as the initial reference
    x_ref, y_ref = x_first.copy(), y_first.copy()
    
    for i, file_name in enumerate(lidar_files[1:], start=1):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name} ({i}/{len(lidar_files)-1})...")

        # Load next scan
        x_src, y_src = to_cartesian(file_path)
        
        # Align current scan to reference scan using ICP
        x_aligned, y_aligned, R_local, t_local = icp_2d(x_ref, y_ref, x_src, y_src)
        
        # Update global transformation
        R_global = R_local @ R_global  # Accumulate rotation
        t_global = R_local @ t_global + t_local  # Accumulate translation
        
        # Transform the original source points to global frame
        src_points = np.vstack((x_src, y_src))
        global_points = transform_points(src_points.T, R_global, t_global)
        
        # Add transformed points to global map
        global_map_x = np.append(global_map_x, global_points[:, 0])
        global_map_y = np.append(global_map_y, global_points[:, 1])
        
        # Update reference for next iteration
        x_ref, y_ref = x_src.copy(), y_src.copy()
        
        # Extract robot orientation (theta) from rotation matrix
        theta = np.arctan2(R_global[1, 0], R_global[0, 0])
        poses.append((float(t_global[0]), float(t_global[1]), float(theta)))

    # Plot final map and robot path
    plt.figure(figsize=(10, 8))
    plt.scatter(global_map_x, global_map_y, color='blue', s=1, alpha=0.5, label='LIDAR Points')
    
    # Plot robot path
    pose_x = [p[0] for p in poses]
    pose_y = [p[1] for p in poses]
    plt.plot(pose_x, pose_y, color='red', linewidth=2, linestyle='-', label='Robot Path')
    
    # Add robot orientations as arrows
    arrow_scale = 0.5  # Scale factor for arrows
    for i, (x, y, theta) in enumerate(poses):
        if i % 5 == 0:  # Draw every 5th orientation for clarity
            dx = arrow_scale * np.cos(theta)
            dy = arrow_scale * np.sin(theta)
            plt.arrow(x, y, dx, dy, head_width=0.2, head_length=0.3, fc='green', ec='green')
    
    plt.title('Global LIDAR Map with Robot Path')
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')  # Equal aspect ratio
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_lidar_files()