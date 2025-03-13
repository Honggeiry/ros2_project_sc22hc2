# Exercise 2 - detecting two colours, and filtering out the third colour and background.



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

        # Initialise the value you wish to use for sensitivity in the colour detection (10 should be enough)
        # Remember to initialise a CvBridge() and set up a subscriber to the image topic you wish to use
        # We covered which topic to subscribe to should you wish to receive image data
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.subscription  # prevent unused variable warning

        self.sensitivity = 10

        # HSV range for red and blue
        self.hsv_red_lower1 = np.array([0, 100, 100])
        self.hsv_red_upper1 = np.array([self.sensitivity, 255, 255])
        self.hsv_red_lower2 = np.array([180 - self.sensitivity, 100, 100])
        self.hsv_red_upper2 = np.array([180, 255, 255])

        self.hsv_blue_lower = np.array([110 - self.sensitivity, 100, 100])
        self.hsv_blue_upper = np.array([130 + self.sensitivity, 255, 255])

    def callback(self, data):

        # Convert the received image into a opencv image
        # But remember that you should always wrap a call to this conversion method in an exception handler
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Failed: {e}")
            return

        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Filter out everything but particular colours using the cv2.inRange() method
        # Do this for each colour


        # To combine the masks you should use the cv2.bitwise_or() method
        # You can only bitwise_or two images at once, so multiple calls are necessary for more than two colours
        red_mask1 = cv2.inRange(hsv_image, self.hsv_red_lower1, self.hsv_red_upper1)
        red_mask2 = cv2.inRange(hsv_image, self.hsv_red_lower2, self.hsv_red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        blue_mask = cv2.inRange(hsv_image, self.hsv_blue_lower, self.hsv_blue_upper)
        combined_mask = cv2.bitwise_or(red_mask, blue_mask)

        # Apply the mask to the original image using the cv2.bitwise_and() method
        # As mentioned on the worksheet the best way to do this is to...
        #bitwise and an image with itself and pass the mask to the mask parameter (rgb_image,rgb_image, mask=mask)
        # As opposed to performing a bitwise_and on the mask and the image.
        filtered_image = cv2.bitwise_and(cv_image, cv_image, mask=combined_mask)

        #Show the resultant images you have created. You can show all of them or just the end result if you wish to.
        cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
        cv2.imshow('Original Image', cv_image)
        cv2.resizeWindow('Original Image', 320, 240)

        cv2.namedWindow('Filtered Image (Red & Blue)', cv2.WINDOW_NORMAL)
        cv2.imshow('Filtered Image (Red & Blue)', filtered_image)
        cv2.resizeWindow('Filtered Image (Red & Blue)', 320, 240)

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
