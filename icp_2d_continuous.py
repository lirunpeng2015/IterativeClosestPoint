import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
import my_icp_helper as my_icp
from scipy.spatial.transform import Rotation as R

folder_path = 'lidar-scans/classroom'

def process_lidar_files():
    lidar_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    if not lidar_files:
        print("No scan files found.")
        return

    # Load first scan as reference
    x_first, y_first = my_icp.to_cartesian(os.path.join(folder_path, lidar_files[0]))
    
    # Initialize global map with first scan
    global_map_x, global_map_y = x_first.copy(), y_first.copy()
    
    # Store robot poses (x, y, theta)
    poses = [(0.0, 0.0, 0.0)]  # initial pose
    
    # Initialize global transformation as identity
    R_global = np.eye(2)
    t_global = np.zeros((2, 1))

    rotation_error = np.eye(2)
    translation_error = np.zeros((2, 1))
    
    # Use the first scan as the initial reference
    x_ref, y_ref = x_first.copy(), y_first.copy()
    
    for i, file_name in enumerate(lidar_files[1:], start=1):
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing {file_name} ({i}/{len(lidar_files)-1})...")

        # Load next scan
        x_src, y_src = my_icp.to_cartesian(file_path)
        
        # Align current scan to reference scan using ICP
        x_aligned, y_aligned, R_local, t_local = my_icp.icp_2d(x_ref, y_ref, x_src, y_src)

        # Convert to rotation objects
        rot_local = R.from_matrix(R_local)
        rot_error = R.from_matrix(rotation_error)

        # Get the rotation magnitudes (in radians)
        angle_local = rot_local.magnitude()
        angle_error = rot_error.magnitude()

        # Compare which one has more rotation
        if angle_local > angle_error:
            rotation_error = R_local

        # Compute magnitudes (Euclidean distances)
        dist_local = np.linalg.norm(t_local)
        dist_error = np.linalg.norm(translation_error)

        # Compare distances
        if dist_local > dist_error:
            translation_error = t_local
        
        # Update global transformation
        R_global = R_local @ R_global  # Accumulate rotation
        t_global = R_local @ t_global + t_local  # Accumulate translation
        
        # Transform the original source points to global frame
        src_points = np.vstack((x_src, y_src))
        global_points = my_icp.transform_points(src_points.T, R_global, t_global)
        
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
    print('Rotation Error: ', rotation_error)
    print('Translation Error: ', translation_error)

if __name__ == "__main__":
    process_lidar_files()
