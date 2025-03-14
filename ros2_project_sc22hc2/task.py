from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal

class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.blue_flag = False
        self.sensitivity = 10
        self.move_forward_flag = False
        self.move_backward_flag = False
        self.stop_flag = False
        self.search_flag = True  # Flag to indicate if the robot is searching for the blue box

        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define range for blue color in HSV
        lower_blue = np.array([110 - self.sensitivity, 100, 100])
        upper_blue = np.array([130 + self.sensitivity, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:  # Adjust this threshold as needed
                self.blue_flag = True
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if cv2.contourArea(c) > 10000:  # Adjust this threshold as needed
                        self.move_backward_flag = True
                        self.move_forward_flag = False
                    elif cv2.contourArea(c) < 5000:  # Adjust this threshold as needed
                        self.move_forward_flag = True
                        self.move_backward_flag = False
                    else:
                        self.stop_flag = True
                        self.move_forward_flag = False
                        self.move_backward_flag = False
            else:
                self.blue_flag = False
        else:
            self.blue_flag = False
            self.move_forward_flag = False
            self.move_backward_flag = False
            self.stop_flag = False

        # Show the camera feed
        cv2.imshow('Camera Feed', image)
        cv2.waitKey(3)

    def search_for_blue(self):
        twist = Twist()
        twist.angular.z = 0.5  # Rotate the robot to search for the blue box
        self.publisher.publish(twist)

    def move_towards_blue(self):
        twist = Twist()
        if self.move_forward_flag:
            twist.linear.x = 0.2  # Move forward
        elif self.move_backward_flag:
            twist.linear.x = -0.2  # Move backward
        elif self.stop_flag:
            twist.linear.x = 0.0  # Stop
        self.publisher.publish(twist)

def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()

    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            if robot.blue_flag:
                robot.move_towards_blue()
            else:
                robot.search_for_blue()
            time.sleep(0.1)
    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()