# Exercise 3 - If green object is detected, and above a certain size, then send a message (print or use lab2)

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



class ColourIdentifier(Node):
    def __init__(self):
        super().__init__('cI')
        # Initialise any flags that signal a colour has been detected (default to false)
        self.green_found = False
        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        self.sensitivity = 10
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning
        
        # Set the upper and lower bounds for the colour you wish to identify - green
        self.hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        self.hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])

    def callback(self, data):
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed: {e}")
            return

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Filter out everything but a particular colour using the cv2.inRange() method
        green_mask = cv2.inRange(hsv_image, self.hsv_green_lower, self.hsv_green_upper)
        # Apply the mask to the original image using the cv2.bitwise_and() method
        filtered_image = cv2.bitwise_and(cv_image, cv_image, mask=green_mask)

        # Find the contours that appear within the certain colour mask using the cv2.findContours() method
        # For <mode> use cv2.RETR_LIST for <method> use cv2.CHAIN_APPROX_SIMPLE
        contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)

            contour_area = cv2.contourArea(largest_contour)

            area_threshold = 1000

            if contour_area > area_threshold:
                self.green_found = True

                M = cv2.moments(largest_contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)

                # Draw the contour on the original image
                cv2.circle(cv_image, center, radius, (0, 255, 0), 2)

                # Draw the contour on the filtered image
                cv2.circle(filtered_image, center, radius, (0, 255, 0), 2)

                self.get_logger().info(f"Green object detected，centre position: ({cx}, {cy}), Area: {contour_area}")
            else:
                self.green_found = False
        else:
            self.green_found = False

        # Show the original image with the contour
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', cv_image)
        cv2.resizeWindow('Original Image', 320, 240)

        # Show the filtered image with the contour
        cv2.namedWindow('Filtered Image (Green)', cv2.WINDOW_NORMAL)
        cv2.imshow('Filtered Image (Green)', filtered_image)
        cv2.resizeWindow('Filtered Image (Green)', 320, 240)

        cv2.waitKey(3)

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
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

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()


# Check if the node is executing in the main path
if __name__ == '__main__':
    main()
