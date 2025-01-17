# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 11:14:59 2016

@author: bob
"""

import numpy as np
import cv2

cap = cv2.VideoCapture(0)
while(True):
# Capture frame-by-frame
    ret, frame = cap.read()
 
# Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Display the resulting frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
   
   
## saving a video file

## opening a video file
cap = cv2.VideoCapture('vtest.avi')
    
while(cap.isOpened()):
    ret, frame = cap.read() 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
cap.release()
cv2.destroyAllWindows()
