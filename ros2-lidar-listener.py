import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

class LidarListener(Node):
    def __init__(self):
        super().__init__('lidar_listener')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription

    def listener_callback(self, msg):
        self.get_logger().info(f'Ranges: {msg.ranges}')

def main(args=None):
    rclpy.init(args=args)
    node = LidarListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
