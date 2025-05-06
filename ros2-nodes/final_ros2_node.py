import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from scipy.spatial import KDTree
from geometry_msgs.msg import PoseStamped
import math
import time

class PersistentMapBuilder(Node):
    def __init__(self):
        super().__init__('persistent_map_builder')

        self.subscription = self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.map_pub = self.create_publisher(PointCloud2, '/global_map', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/robot_pose', 10)


        self.global_map = None  # Accumulated map points as Nx2 array
        self.global_rotation = np.eye(2)  # Cumulative rotation
        self.global_translation = np.zeros((2, 1))  # Cumulative translation

        self.max_num_of_stored_scans = 5 # Optimized number of scans for ICP to be fast enough
        self.prev_scans_transformed = [] # List of previous scans
        self.prev_n_points = None # Array of points from previous scans


    def laser_callback(self, msg):
        # Testing for time
        start_time = time.perf_counter()

        # get angles and ranges from message
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)

        # convert ranges and angles to cartesian coordinates
        new_points = self.polar_to_cartesian(ranges, angles)

        # filter out non-finite ranges and angles, NAN and INF
        valid = np.isfinite(new_points[:, 0]) & np.isfinite(new_points[:, 1])
        new_points = new_points[valid]

        # First scan
        if len(self.prev_scans_transformed) == 0:
            # No ICP required, just initialize map and previous scans list
            self.global_map = new_points
            self.prev_scans_transformed.append(new_points)
            self.prev_n_points = new_points
        # Subsequent scans
        else:
            # Get previous n scans as array of points
            self.prev_n_points = np.vstack(self.prev_scans_transformed)
            # self.get_logger().info(f'Length of prev n points {self.prev_n_points.size}')

            # Transform new points using global transform before ICP
            new_points_transformed = (self.global_rotation @ new_points.T).T + self.global_translation.T
            # Perform ICP
            rotation, translation = self.icp(self.prev_n_points, new_points_transformed)

            # Update global transform
            self.global_rotation = rotation @ self.global_rotation
            self.global_translation = rotation @ self.global_translation + translation

            # Apply global transformation to the new scan points
            new_points_transformed = (self.global_rotation @ new_points.T).T + self.global_translation.T

            # Update previous points
            # First n scans
            if len(self.prev_scans_transformed) < self.max_num_of_stored_scans:
                # Add new scan transformed
                self.prev_scans_transformed.append(new_points_transformed)
            # Subsequent scans
            else:
                # Remove oldest scan and add new scan transformed
                self.prev_scans_transformed.pop()
                self.prev_scans_transformed.append(new_points_transformed)

            # Append the transformed points to the global map
            self.global_map = np.vstack((self.global_map, new_points_transformed))

            # Debugging
            # self.get_logger().info(f'Length of previous scans list: {len(self.prev_scans_transformed)}')
            

        # Publish global map
        self.publish_pointcloud(self.global_map, msg.header.stamp)
        # Publish pose
        self.publish_robot_pose(msg.header.stamp)


        # Testing for time
        end_time = time.perf_counter()
        scan_process_time = end_time - start_time
        self.get_logger().info(f'Scan process time: {scan_process_time}')

    def publish_pointcloud(self, points, stamp):
        header = Header()
        header.stamp = stamp
        header.frame_id = 'map'

        cloud_data = [(p[0], p[1], 0.0) for p in points]
        cloud_msg = pc2.create_cloud_xyz32(header, cloud_data)
        self.map_pub.publish(cloud_msg)

    def publish_robot_pose(self, stamp):
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = 'map'

        # Translation
        pose.pose.position.x = self.global_translation[0, 0]
        pose.pose.position.y = self.global_translation[1, 0]
        pose.pose.position.z = 0.0

        # Rotation (2D to quaternion)
        theta = math.atan2(self.global_rotation[1, 0], self.global_rotation[0, 0])
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        self.pose_pub.publish(pose)


    def polar_to_cartesian(self, ranges, angles):
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        points = np.vstack((x, y)).T

        return points

    def icp(self, reference_points, source_points, max_iterations=100, tolerance=1e-8):
        R = np.eye(2)
        t = np.zeros((2, 1))
        prev_error = float('inf')

        for _ in range(max_iterations):
            tree = KDTree(reference_points)
            distances, indices = tree.query(source_points)
            closest_ref = reference_points[indices]

            centroid_src = np.mean(source_points, axis=0)
            centroid_ref = np.mean(closest_ref, axis=0)

            src_centered = source_points - centroid_src
            ref_centered = closest_ref - centroid_ref

            H = src_centered.T @ ref_centered
            U, _, Vt = np.linalg.svd(H)
            R_opt = Vt.T @ U.T

            if np.linalg.det(R_opt) < 0:
                Vt[1, :] *= -1
                R_opt = Vt.T @ U.T

            t_opt = centroid_ref.reshape(2, 1) - R_opt @ centroid_src.reshape(2, 1)
            source_points = (R_opt @ source_points.T).T + t_opt.T

            R = R_opt @ R
            t = R_opt @ t + t_opt

            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        return R, t


def main(args=None):
    rclpy.init(args=args)
    node = PersistentMapBuilder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
