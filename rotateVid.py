# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 12:19:03 2017

@author: jct1c13
"""

import cv2
import numpy as np
import sys

video = cv2.VideoCapture("FORE_A8S12.mp4")

if not video.isOpened():
    print("Could not open video")
    sys.exit()
    
ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

rows,cols,col = frame.shape
M = cv2.getRotationMatrix2D((cols/2,rows/2),3.88,1)
dst = cv2.warpAffine(frame,M,(cols,rows))
cv2.imwrite('output.png',dst)

cv2.imshow("Tracking", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#
#while True:
## Read a new frame
#    ok, frame = video.read()
#    if not ok:
#        break
#        # Display result
#        cv2.imshow("Tracking", frame)
# 
#        # Exit if ESC pressed
#        k = cv2.waitKey(1) & 0xff
#        if k == 27 : break
#    