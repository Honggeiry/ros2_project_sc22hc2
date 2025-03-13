# Exercise 1 - Display an image of the camera feed to the screen

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from rclpy.exceptions import ROSInterruptException
import signal


class colourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'camera/image_raw',
            self.callback,
            10
        )
        self.subscription
        
        self.sensitivity = 30
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Failed: {e}")
            return

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        filtered_image = cv2.bitwise_and(cv_image, cv_image, mask=green_mask)

        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', cv_image)
        cv2.resizeWindow('Original Image', 320, 240)

        cv2.namedWindow('Filtered Image (Green)', cv2.WINDOW_NORMAL)
        cv2.imshow('Filtered Image (Green)', filtered_image)
        cv2.resizeWindow('Filtered Image (Green)', 320, 240)

        cv2.waitKey(3)
        

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    rclpy.init(args=None)
    cI = colourIdentifier()
    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(cI,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            continue
    except ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    
# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
