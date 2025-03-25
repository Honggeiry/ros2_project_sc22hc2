from __future__ import division
import threading
import sys
import time
import signal
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

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
        self.subscription_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        
        self.waypoints = [
            (-2.97, -3.9, -0.00134),
            (6.9, -5.13, -0.00134),
            (7.27, -11.4, -0.00134),
            (-6.22, -12.4, -0.00134)]
        self.current_waypoint = 0
        self.navigator = GoToPose()

        # Centroid tracking
        self.cx = None
        self.frame_width = None

        # LiDAR parameters
        self.current_distance = float('inf')
        self.target_distance = 0.8
        self.distance_tolerance = 0.15
        self.Kp_linear = 0.3
        self.max_linear_speed = 0.2
        self.min_safe_distance = 0.7
        self.lidar_initialized = False

        # Centering parameters
        self.center_threshold = 20
        self.Kp_angle = 0.005

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            self.get_logger().error(f"Image error: {str(e)}")
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue detection for control
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
        contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.frame_width = frame.shape[1]
        self.blue_detected = False
        self.cx = None

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 500:
                self.blue_detected = True
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    self.cx = int(M["m10"] / M["m00"])

        # Colour filtering
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # Green mask
        lower_green = np.array([40, 70, 70])
        upper_green = np.array([80, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        # Combine masks: blue, red, and green
        combined_mask = cv2.bitwise_or(mask_blue, mask_red)
        combined_mask = cv2.bitwise_or(combined_mask, mask_green)

        # Apply the combined mask to show only the filtered colours
        filtered_image = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Display the filtered image
        cv2.imshow('Filtered Colours', filtered_image)
        cv2.waitKey(1)

    def scan_callback(self, msg):
        try:
            if not self.lidar_initialized:
                self.lidar_initialized = True

            angle_increment = msg.angle_increment
            start_angle = -np.deg2rad(15)
            end_angle = np.deg2rad(15)
            
            start_idx = int((start_angle - msg.angle_min) / angle_increment)
            end_idx = int((end_angle - msg.angle_min) / angle_increment)
            
            if start_idx < 0:
                start_idx += len(msg.ranges)
            if end_idx < 0:
                end_idx += len(msg.ranges)

            if start_idx < end_idx:
                readings = msg.ranges[start_idx:end_idx]
            else:
                readings = msg.ranges[start_idx:] + msg.ranges[:end_idx]

            valid_readings = [r for r in readings 
                              if (msg.range_min < r < msg.range_max) 
                              and not np.isnan(r)]
            
            if valid_readings:
                self.current_distance = min(valid_readings)
                self.get_logger().debug(f"New LiDAR distance: {self.current_distance:.2f}m")
            else:
                self.current_distance = float('inf')
                
        except Exception as e:
            self.get_logger().error(f"LiDAR processing error: {str(e)}")

    def control_blue_approach(self):
        cmd = Twist()
        if not self.blue_detected or self.cx is None:
            self.get_logger().info("No blue detected")
            return

        center_x = self.frame_width / 2.0
        error_x = self.cx - center_x
        angular_z = -self.Kp_angle * error_x

        if self.current_distance <= self.min_safe_distance:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn(f"EMERGENCY STOP! Distance: {self.current_distance:.2f}m")
        elif abs(error_x) > self.center_threshold:
            cmd.angular.z = angular_z
            self.get_logger().info(f"Centering: X error {error_x:.1f}px, Angular Z: {cmd.angular.z:.2f}")
        else:
            distance_error = self.current_distance - self.target_distance
            if distance_error > self.distance_tolerance:
                cmd.linear.x = min(self.Kp_linear * distance_error, self.max_linear_speed)
                self.get_logger().info(f"Approaching: {self.current_distance:.2f}m, Speed: {cmd.linear.x:.2f}m/s")
            else:
                cmd.linear.x = 0.0
                self.get_logger().info("Reached target distance!")

        self.publisher.publish(cmd)

    def search_blue(self):
        cmd = Twist()
        cmd.angular.z = 0.4
        self.publisher.publish(cmd)
        self.get_logger().info("Searching for blue box...")

def main():
    rclpy.init()
    robot = Robot()
    navigator = robot.navigator

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(robot)
    executor.add_node(navigator)

    def signal_handler(sig, frame):
        robot.get_logger().info("Shutting down")
        rclpy.shutdown()

    signal.signal(signal.SIGINT, signal_handler)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # Phase 1: Waypoint navigation
        while robot.current_waypoint < len(robot.waypoints) and rclpy.ok():
            x, y, yaw = robot.waypoints[robot.current_waypoint]
            navigator.send_goal(x, y, yaw)
            
            while not navigator.goal_completed and not robot.blue_detected and rclpy.ok():
                time.sleep(0.1)
            
            if robot.blue_detected:
                navigator.cancel_goal()
                robot.get_logger().info("Blue detected - interrupting navigation")
                break
            else:
                robot.current_waypoint += 1

        # Phase 2: Box approach
        if robot.blue_detected:
            robot.get_logger().info("Starting approach procedure")
            last_distance = None
            stable_count = 0
            
            while rclpy.ok():
                if not robot.blue_detected:
                    robot.search_blue()
                    time.sleep(0.1)
                    continue
                
                # Check for stable distance reading
                if last_distance is not None:
                    if abs(robot.current_distance - last_distance) < 0.01:
                        stable_count += 1
                    else:
                        stable_count = 0
                
                if stable_count > 5:
                    robot.get_logger().info("Stable distance achieved")
                    break
                
                if abs(robot.current_distance - robot.target_distance) <= robot.distance_tolerance:
                    robot.get_logger().info("Successfully reached target distance")
                    break
                
                robot.control_blue_approach()
                last_distance = robot.current_distance
                time.sleep(0.1)
        else:
            robot.get_logger().info("Initiating search pattern")
            while not robot.blue_detected and rclpy.ok():
                robot.search_blue()
                time.sleep(0.1)

        # Final stop
        robot.publisher.publish(Twist())
        robot.get_logger().info("Mission complete")

    except Exception as e:
        robot.get_logger().error(f"Critical error: {str(e)}")
    finally:
        executor.shutdown()
        robot.destroy_node()
        navigator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
