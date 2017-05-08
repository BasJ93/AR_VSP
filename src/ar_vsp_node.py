#!/usr/bin/env python
from __future__ import print_function
import roslib
import sys
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import imutils
import numpy as np

#Author: Bas Janssen
#Fontys Mechatronica
#Minro AR 2017

# Class that converts the image/video obtained from the image_raw topic to a image/video which
# can be edited in python with use of openCV.
#The class uses this to track a yellow ball in the frame, using HSV color space, color filtering and contour detection.
#A circle is drawn around the detected ball
class image_converter:    
    def __init__(self):
        self.bridge = CvBridge()
        # The image/video is received from the bluefox2_single/image_raw topic
        self.image_sub = rospy.Subscriber("usb_cam/image_raw", Image,self.callback)

    def callback(self,data):
        try:
            # cv_image = converted image/video from ROS to openCV
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        yellow_min = np.array([25,70,120])
        yellow_max = np.array([60,255,255])
        
        mask = cv2.inRange(hsv, yellow_min, yellow_max)        
        
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)        
        
        thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(blurred.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        
        c = max(cnts, key=cv2.contourArea)        	
        ((x, y), rad) = cv2.minEnclosingCircle(c)
         
        cv2.circle(cv_image, (int(x), int(y)), int(rad), (0, 0, 0), 2)
        
        cv2.imshow("hsv", hsv)
        cv2.imshow("blurred", blurred)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Detection", cv_image)
        cv2.waitKey(3)

def main(args):
    rospy.init_node('prj7_vision_test_bluefox', anonymous=True)
    image_converter()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)