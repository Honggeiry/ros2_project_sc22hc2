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

        goal_msg.pose.pose.orientation.z = np.sin(yaw / 2)
        goal_msg.pose.pose.orientation.w = np.cos(yaw / 2)

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
            _ = self.goal_handle.cancel_goal_async()
            self.get_logger().info('Canceling current goal')

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.blue_detected = False

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # Waypoints list: (x, y, yaw)
        self.waypoints = [
            (-2.97, -3.9, -0.00134),
            (4.18, -5.49, -0.00134),
            (1.71, -10.1, -0.00134)
        ]
        self.current_waypoint = 0
        # Create the navigator node
        self.navigator = GoToPose()

        # For smoothing blue area detection
        self.area_buffer = []
        self.buffer_size = 5
        self.avg_area = 0

        # New variables for centroid tracking
        self.cx = None
        self.cy = None
        self.frame_width = None
        self.frame_height = None

        # Parameters for approach control:
        # target_area corresponds to the blue box's apparent area when the robot is ~1 m away.
        # Increase this value if the robot stops too early.
        self.target_area = 200000    # Increased target area so that the robot drives closer
        self.tolerance = 5000        # Acceptable range around the target area
        self.Kp = 0.00001            # Proportional gain for forward velocity

        # Parameters for centering control
        self.center_threshold = 20   # pixel error tolerated for centering
        self.Kp_angle = 0.005        # Proportional gain for angular velocity

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Blue color detection thresholds (adjust if needed)
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours of blue regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Update frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Reset blue detection and centroid variables
        self.blue_detected = False
        self.cx = None
        self.cy = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > 500:
                self.blue_detected = True
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    self.cx = int(M["m10"] / M["m00"])
                    self.cy = int(M["m01"] / M["m00"])
                # Update the buffer and compute the running average area
                self.area_buffer.append(area)
                if len(self.area_buffer) > self.buffer_size:
                    self.area_buffer.pop(0)
                self.avg_area = sum(self.area_buffer) / len(self.area_buffer)
        else:
            self.area_buffer = []
            self.avg_area = 0

        cv2.imshow('Camera View', frame)
        cv2.waitKey(1)

    def control_blue_approach(self):
        """
        If the blue box is not centered, rotate to center it.
        Once centered, use a proportional controller based on the area error to drive forward.
        """
        cmd = Twist()
        if not self.blue_detected or self.avg_area == 0 or self.cx is None or self.frame_width is None:
            self.get_logger().info("No blue detected during approach")
            return

        center_x = self.frame_width / 2.0
        error_x = self.cx - center_x

        # If blue box is not centered, command an angular correction.
        if abs(error_x) > self.center_threshold:
            cmd.angular.z = -self.Kp_angle * error_x
            self.get_logger().info(f"Rotating: error_x = {error_x:.2f}, angular velocity = {cmd.angular.z:.2f}")
        else:
            # Blue box is centered; now use area error to drive forward.
            if self.avg_area < self.target_area:
                error_area = self.target_area - self.avg_area
                vel = self.Kp * error_area
                if vel > 0.25:
                    vel = 0.25  # cap the velocity
                cmd.linear.x = vel
                self.get_logger().info(f"Moving forward: area error = {error_area:.2f}, velocity = {cmd.linear.x:.2f}")
            else:
                cmd.linear.x = 0.0
                self.get_logger().info(f"Target reached (avg_area = {self.avg_area:.2f}), stopping")
        self.publisher.publish(cmd)

    def search_blue(self):
        cmd = Twist()
        cmd.angular.z = 0.4
        self.get_logger().info("Searching for blue")
        self.publisher.publish(cmd)

def main():
    rclpy.init()
    # Create nodes
    robot = Robot()
    navigator = robot.navigator  # Access the navigator node

    # Create a MultiThreadedExecutor and add both nodes so that all callbacks are handled
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(navigator)

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        robot.get_logger().info("Shutting down")
        rclpy.shutdown()
    signal.signal(signal.SIGINT, signal_handler)

    # Start executor in a separate thread
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Phase 1: Navigate through waypoints until a blue object is detected.
        while robot.current_waypoint < len(robot.waypoints) and rclpy.ok():
            x, y, yaw = robot.waypoints[robot.current_waypoint]
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

        # Phase 2: Approach the blue box using centering and proportional control.
        if robot.blue_detected:
            robot.get_logger().info("Approaching blue box")
            while rclpy.ok():
                # If blue is lost during approach, switch to search mode.
                if not robot.blue_detected:
                    robot.get_logger().info("Lost blue target, searching...")
                    robot.search_blue()
                # Check if the area error is within tolerance.
                if abs(robot.target_area - robot.avg_area) < robot.tolerance:
                    robot.get_logger().info("Reached desired distance from blue box.")
                    break
                robot.control_blue_approach()
                time.sleep(0.1)
        else:
            robot.get_logger().info("Searching for blue box")
            while not robot.blue_detected and rclpy.ok():
                robot.search_blue()
                time.sleep(0.1)
            if robot.blue_detected:
                robot.get_logger().info("Blue found - approaching")
                while rclpy.ok():
                    if abs(robot.target_area - robot.avg_area) < robot.tolerance:
                        robot.get_logger().info("Reached desired distance from blue box.")
                        break
                    if robot.blue_detected:
                        robot.control_blue_approach()
                    else:
                        robot.get_logger().info("Lost blue target, searching...")
                        robot.search_blue()
                    time.sleep(0.1)

        # Final stop command
        robot.publisher.publish(Twist())
        robot.get_logger().info("Mission complete")

    except Exception as e:
        robot.get_logger().error(f"Error: {str(e)}")
    finally:
        # Do not close the camera window so it remains open.
        executor.shutdown()
        robot.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
