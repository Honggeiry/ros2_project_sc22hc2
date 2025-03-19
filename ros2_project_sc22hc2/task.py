from __future__ import division
import threading
import sys, time, signal
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException

class GoToPose(Node):
    def __init__(self):
        super().__init__('navigation_goal_action_client')
        self.action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.goal_completed = False

    def send_goal(self, x, y, yaw):
        self.goal_completed = False
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        # Convert yaw to quaternion (assuming roll=pitch=0)
        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2)

        self.get_logger().info(f"Sending goal: x={x}, y={y}, yaw={yaw}")
        self.action_client.wait_for_server()
        self.send_goal_future = self.action_client.send_goal_async(goal_msg)
        self.send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        self.goal_handle = future.result()
        if not self.goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info("Goal accepted")
        self.get_result_future = self.goal_handle.get_result_async()
        self.get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        self.goal_completed = True
        self.get_logger().info('Goal completed')

    def cancel_goal(self):
        if self.goal_handle:
            future = self.goal_handle.cancel_goal_async()
            self.get_logger().info('Canceling current goal')

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.blue_detected = False
        self.sensitivity = 10
        self.move_forward = False
        self.move_backward = False
        self.stop_moving = False

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # Waypoints list: (x, y, yaw)
        self.waypoints = [
            (-2.97, -3.9, -0.00134),
            (4.18, -5.49, -0.00134)
        ]
        self.current_waypoint = 0
        # Create the navigator node
        self.navigator = GoToPose()

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        # Convert frame to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color detection thresholds
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours of the blue regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset flags each callback
        self.blue_detected = False
        self.move_forward = False
        self.move_backward = False
        self.stop_moving = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                self.blue_detected = True
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    # You can use the centroid (cx) if needed for further control
                    cx = int(M["m10"] / M["m00"])
                    area = cv2.contourArea(largest)
                    # Decide motion based on the area of the blue object
                    if area > 15000:  # Too close
                        self.move_backward = True
                    elif area < 8000:  # Too far
                        self.move_forward = True
                    else:  # Within desired range
                        self.stop_moving = True

        cv2.imshow('Camera View', frame)
        cv2.waitKey(1)

    def approach_blue(self):
        cmd = Twist()
        if self.move_forward:
            cmd.linear.x = 0.25
            self.get_logger().info("Moving forward")
        elif self.move_backward:
            cmd.linear.x = -0.25
            self.get_logger().info("Moving backward")
        elif self.stop_moving:
            cmd.linear.x = 0.0
            self.get_logger().info("Stopping")
        self.publisher.publish(cmd)

    def search_blue(self):
        cmd = Twist()
        cmd.angular.z = 0.4
        self.get_logger().info("Searching for blue")
        self.publisher.publish(cmd)

def main():
    rclpy.init()
    # Create our nodes
    robot = Robot()
    navigator = robot.navigator  # Access the navigator node

    # Create a MultiThreadedExecutor and add both nodes so that all callbacks are handled
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(navigator)

    # Define a signal handler for a graceful shutdown
    def signal_handler(sig, frame):
        robot.get_logger().info("Shutting down")
        rclpy.shutdown()
    signal.signal(signal.SIGINT, signal_handler)

    # Start the executor in a separate thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Phase 1: Navigate through waypoints until a blue object is detected
        while robot.current_waypoint < len(robot.waypoints) and rclpy.ok():
            x, y, yaw = robot.waypoints[robot.current_waypoint]
            robot.get_logger().info(f"Navigating to waypoint {robot.current_waypoint}: x={x}, y={y}, yaw={yaw}")
            navigator.send_goal(x, y, yaw)
            
            # Wait until either the navigation goal is completed or blue is detected.
            while not navigator.goal_completed and not robot.blue_detected and rclpy.ok():
                time.sleep(0.1)
            
            if robot.blue_detected:
                navigator.cancel_goal()
                robot.get_logger().info("Blue detected - interrupting navigation")
                break
            else:
                robot.current_waypoint += 1

        # Phase 2: Handle blue box detection
        if robot.blue_detected:
            robot.get_logger().info("Approaching blue box")
            # Continue approaching until the blue box is within the desired range (stop_moving flag set)
            while not robot.stop_moving and rclpy.ok():
                robot.approach_blue()
                time.sleep(0.1)
        else:
            robot.get_logger().info("Searching for blue box")
            # If no blue was detected during waypoint navigation, search for it
            while not robot.blue_detected and rclpy.ok():
                robot.search_blue()
                time.sleep(0.1)
            
            if robot.blue_detected:
                robot.get_logger().info("Blue found - approaching")
                while not robot.stop_moving and rclpy.ok():
                    robot.approach_blue()
                    time.sleep(0.1)

        # Final stop command
        robot.publisher.publish(Twist())
        robot.get_logger().info("Mission complete")

    except Exception as e:
        robot.get_logger().error(f"Error: {str(e)}")
    finally:
        cv2.destroyAllWindows()
        # Shutdown the executor and nodes
        executor.shutdown()
        robot.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
