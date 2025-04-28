import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from scipy.spatial import cKDTree
import math


class PersistentMapBuilder(Node):
    def __init__(self):
        super().__init__('persistent_map_builder')

        # Subscriber to LaserScan messages
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)

        # Publisher for the global map
        self.map_pub = self.create_publisher(PointCloud2, '/global_map', 10)

        # Initialize the map and pose
        self.global_map = None  # This will store the persistent map (Nx2)
        self.current_pose = np.eye(3)  # SE(2) pose (3x3 matrix)

    def laser_callback(self, msg):
        # Convert the LaserScan data to 2D points
        scan_points = self.laserscan_to_points(msg)

        # If it's the first scan, initialize the global map
        if self.global_map is None:
            self.global_map = scan_points
            self.publish_map()
            return

        # Align the current scan to the global map using ICP
        T = self.icp(scan_points, self.global_map)
        self.current_pose = T @ self.current_pose  # Update the robot's pose

        # Transform the current scan into the global map frame
        homog = np.hstack((scan_points, np.ones((scan_points.shape[0], 1))))
        transformed = (self.current_pose @ homog.T).T[:, :2]

        # Add the transformed scan to the global map
        self.global_map = np.vstack((self.global_map, transformed))

        # Publish the updated map
        self.publish_map()

    def laserscan_to_points(self, msg):
        """Convert LaserScan to a 2D point cloud."""
        angles = msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment
        ranges = np.array(msg.ranges)
        mask = (ranges > msg.range_min) & (ranges < msg.range_max)
        ranges = ranges[mask]
        angles = angles[mask]
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.stack((x, y), axis=-1)

    def icp(self, src_points, dst_points, max_iter=25, tolerance=1e-4):
        """Perform ICP to align src_points to dst_points."""
        src = src_points.copy()
        dst = dst_points.copy()
        T = np.eye(3)

        for _ in range(max_iter):
            # Nearest neighbor search
            tree = cKDTree(dst)
            dists, indices = tree.query(src)
            dst_matched = dst[indices]

            # Compute centroids
            mu_src = np.mean(src, axis=0)
            mu_dst = np.mean(dst_matched, axis=0)

            # Center the points
            src_centered = src - mu_src
            dst_centered = dst_matched - mu_dst

            # Compute the best-fit rotation matrix using SVD
            W = dst_centered.T @ src_centered
            U, _, VT = np.linalg.svd(W)
            R = U @ VT
            if np.linalg.det(R) < 0:
                R[1, :] *= -1
            t = mu_dst - R @ mu_src

            # Construct the transformation matrix (SE(2) matrix)
            T_step = np.eye(3)
            T_step[:2, :2] = R
            T_step[:2, 2] = t

            # Apply the transformation to the source points
            homog_src = np.hstack((src, np.ones((src.shape[0], 1))))
            src = (T_step @ homog_src.T).T[:, :2]

            # Update the transformation matrix
            T = T_step @ T

            # Check for convergence
            if np.linalg.norm(t) < tolerance:
                break

        return T

    def publish_map(self):
        """Publish the global map as a PointCloud2."""
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'

        # Convert global map to PointCloud2 format
        points = [(x, y, 0.0) for x, y in self.global_map]
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        cloud = pc2.create_cloud(header, fields, points)
        self.map_pub.publish(cloud)


def main(args=None):
    rclpy.init(args=args)
    node = PersistentMapBuilder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
