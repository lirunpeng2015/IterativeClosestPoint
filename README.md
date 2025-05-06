# Iterative Closest Point

This project is a collection of python scripts and ROS2 nodes that perform iterative closest point.

## Running the code

All python files in the root directory can be run in normal python environment. 

All python files in the ros2-node directory are ROS2 nodes and need to be placed inside a custom ROS2 package. They also need a LiDAR to be publishing LaserScan messages to the /scan topic to work.

## Python Scripts in root directory

These python scripts process point clouds and perform ICP in different ways using point cloud data stored in static files. 

These are the important files:
1. plot_one_point_cloud.py: plots one point cloud. change line 7, file_path = 'lidar-scans/lidar-floor-scan-2.txt', to another file inside the 'lidar-scans' directory to view different point clouds.
2. plot_multiple_point_clouds.py: plots 3 point clouds. change lines 9, 10, 11 to view different point clouds.
3. icp_two_point_clouds.py: performs ICP with two point clouds. Displays each iteration.
4. icp_many_point_clouds.py: performs ICP with all files inside a subfolder of 'lidar-scans'. change line 6 to look at different groups of scans.

The following files are ROS2 nodes inside the 'ros2-nodes' subfolder
1. ros2_w_occupancy_grid.py: performs ICP for a stream of point clouds and publishes an occupancy grid.
  To view: run the rplidar package for the C1. Close rviz2. In another terminal run the node. Open rviz2 in another terminal. Add a display for a map and change the topic to 'occupancy_grid'. Update topic should automatically get set to 'occupancy_grid_update'. Change it if not.

The rest of the ros2 nodes all have problems and are left for documentation purposes.
