# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:03:00 2017

@author: jct1c13
"""
# import the necessary packages
# detection based algorithm based on the pyimage tutorial

from collections import deque
import numpy as np
import imutils
import cv2
import csv
import progressbar
import math  # for NaN's

print('processing')
alpha = -0.355869831
inputFile = 'E:/aftereffets/a8d_04.mp4'
outputFile = inputFile[0:-4] + '_output_rotateOnly.avi'
csvFilename = inputFile[0:-4] + '_output_rotateOnly.csv'
print('output file is ' + outputFile)
print('input file is ' + inputFile)
buffer = 32  # buffer is used to draw line, it could also be used for on-the-fly filtering or tracking
pts = []
# open a csv file in overwrite mode
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (45, 75, 75)
greenUpper = (75, 255, 255)

# initialize the list of tracked points, the frame counter,
# and the coordinate deltas
pts = deque(maxlen=buffer)
counter = 0
(dX, dY) = (0, 0)
direction = ""
camera = cv2.VideoCapture(inputFile)
width = int(camera.get(3))
height = int(camera.get(4))
fps = camera.get(5)
outputVideo = cv2.VideoWriter(outputFile, fourcc, fps, (width, height))
totalFrames = camera.get(cv2.CAP_PROP_FRAME_COUNT)
widgets = [progressbar.Percentage(), progressbar.Bar(), progressbar.ETA()]
bar = progressbar.ProgressBar(widgets=widgets, max_value=totalFrames).start()
        
# keep looping through frames
with open(csvFilename,'w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    while True:
        if counter == totalFrames:
            break
        # grab the current frame
        (grabbed, frame) = camera.read()
        rows,cols,col = frame.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-alpha,1)
        frame = cv2.warpAffine(frame,M,(cols,rows))
#==============================================================================
#         h,  w = frame.shape[:2]

        # resize the frame, blur it, and convert it to the HSV
        # color space
        # frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
     
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = [math.nan, math.nan]
        
    # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
     
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # then update the list of tracked points
                xx = c[:,:,0]
                yy = c[:,:,1]
                pts.appendleft((int(x), int(y)))
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                cv2.drawContours(frame, c, -1, (0, 0, 255),3)
    
        # loop over the set of tracked points
        for i in np.arange(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue
     
            # check to see if enough points have been accumulated in
            # the buffer
            if counter >= 10 and i == 1 and pts[-10] is not None:
                # compute the difference between the x and y
                # coordinates and re-initialize the direction
                # text variables
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")
     
                # ensure there is significant movement in the
                # x-direction
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX) == 1 else "West"
     
                # ensure there is significant movement in the
                # y-direction
                if np.abs(dY) > 20:
                    dirY = "North" if np.sign(dY) == 1 else "South"
     
                # handle when both directions are non-empty
                if dirX != "" and dirY != "":
                    direction = "{}-{}".format(dirY, dirX)
     
                # otherwise, only one direction is non-empty
                else:
                    direction = dirX if dirX != "" else dirY
    
            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            # thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 255), 3)
     
        # show the movement deltas and the direction of movement on
        # the frame
        cv2.putText(frame, 'FRAME ' + (str(counter) + " - " + str(float(np.round(100*(counter/totalFrames),2))) + ' % complete'), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)
     
        # show the frame to our screen and increment the frame counter
        # cv2.imshow("Frame", frame)
#        key = cv2.waitKey(1) & 0xFF
        # if the 'q' key is pressed, stop the loop
#        if key == ord("q"):
#            break
        outputVideo.write(frame)
        # print(str(float(np.round(100*(counter/totalFrames),2))) + ' % complete')
        bar.update(counter)
        counter +=1

#  cleanup the camera and close any open windows
print("Complete!")
bar.finish()
camera.release()
outputVideo.release()
cv2.destroyAllWindows()