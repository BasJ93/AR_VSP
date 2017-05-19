#!/usr/bin/env python
from __future__ import print_function
#import roslib
import sys
#import rospy

import numpy as np
import cv2

import os

stop_cascade = cv2.CascadeClassifier('stopsign_20stage.xml')

if(len(sys.argv) > 1):
    if(sys.argv[1] == "images"):
        print("Loading test images")
        for files in os.listdir("opencv-haar-classifier-training/positive_images/uncropped"):
                if files.endswith(".jpg"):
                    print(os.path.join("opencv-haar-classifier-training/positive_images/uncropped", files))
                    img = cv2.imread(os.path.join("opencv-haar-classifier-training/positive_images/uncropped", files),cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    signs = stop_cascade.detectMultiScale(gray, 1.3, 5)   
                
                    for (x,y,w,h) in signs:
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                
                        
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]
                
                    cv2.imshow('img',img)
                    cv2.imwrite("processed/" + files, img)
                    k = cv2.waitKey(30) & 0xff
                    if k == 27:
                        break

    elif(sys.argv[1] == "video"):
        print("Loading test video")
        cap = cv2.VideoCapture("test_video.mp4")
        
        while 1:
            ret, img = cap.read()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            signs = stop_cascade.detectMultiScale(gray, 1.3, 5)   
            
            for (x,y,w,h) in signs:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
                
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
        
            cv2.imshow('img',img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        
        cap.release()
        
else:
    print("Processing LIVE video stream")
    cap = cv2.VideoCapture(0)
    while 1:
       ret, img = cap.read()
        
       gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       signs = stop_cascade.detectMultiScale(gray, 1.3, 5)   
       
       for (x,y,w,h) in signs:
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                    
           roi_gray = gray[y:y+h, x:x+w]
           roi_color = img[y:y+h, x:x+w]
    
       cv2.imshow('img',img)
       k = cv2.waitKey(30) & 0xff
       if k == 27:
           break
    
    cap.release()
    

cv2.destroyAllWindows()