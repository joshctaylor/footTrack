# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:03:00 2017

@author: jct1c13
"""

# import the necessary packages
# detection based algorithm based on the pyimage tutorial

from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import csv
# http://jgardiner.co.uk/blog/jupyter_progress_bar
import progressbar
import math

print('processing')

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the video file")

ap.add_argument("-a", "--alpha", type=float, default=32,
	help="video rotation angle (deg)")

ap.add_argument("-c", "--centreline", type=float, default=32,
	help="point for centreline reference")

args = vars(ap.parse_args())

alpha = args["alpha"]
inputFile = args["video"]
centreline = int(args["centreline"])

outputFile = inputFile[0:-4] + '_output_withoffset.avi'
fileName = inputFile[0:-4] + '_output_withoffset.csv'
print('output file is ' + outputFile)
print('input file is ' + inputFile)
print(alpha)


buffer = 32
pts = []
# open a csv file in overwrite mode
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# define the lower and upper boundaries of the "green"
# ball in the HSV color space
greenLower = (45, 75, 75)
greenUpper = (75, 255, 255)

# coeffs for the front camera
D = np.array([0.03564257, -0.04109487, 0.00167179, -0.0011342, 0.01131636])
K =  np.array([[736.39241878, 0.0, 954.85240702],
            [   0.0, 739.57526582, 589.60675824],
            [   0.0, 0.0, 1.0]])
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

# keep looping through all frames
with open(fileName,'w') as f1:
    writer=csv.writer(f1, delimiter=',',lineterminator='\n',)
    while True:
        if counter == totalFrames:
            break

        # grab the current frame
        (grabbed, frame) = camera.read()
        h,  w = frame.shape[:2]
        rows,cols,col = frame.shape
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, K, D, None, newcameramtx)
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-alpha,1)
        frame = cv2.warpAffine(frame,M,(cols,rows))

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
        radius = math.nan
        
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
            if counter >= 10 and i == 1 and len(pts) >= 10: #is not None:
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
        offset = -(center[0] - centreline)
        cv2.putText(frame, 'FRAME ' + (str(counter) + " - " + str(float(np.round(100*(counter/totalFrames),2))) + ' % complete, offset = ' + str(offset) + ' radius = ' + str(np.round(radius,0))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.65, (0, 0, 255), 3)
        cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.35, (0, 0, 255), 1)
        cv2.line(frame, (centreline, 0), (centreline, 1080), (255, 0,0), 4)
    
        outputVideo.write(frame)
#        print("data is", repr(center), repr(radius))
        writer.writerow([counter, center[0], center[1], radius, offset])
        bar.update(counter)
#==============================================================================
#         show the frame to our screen and increment the frame counter
#        if the 'q' key is pressed, stop the loop
#        cv2.imshow("Frame", frame)
#        key = cv2.waitKey(1) & 0xFF
#        if key == ord("q"):
#            break
#==============================================================================
        counter += 1

#  cleanup the camera and close any open windows
print("Complete!")
bar.finish()
camera.release()
outputVideo.release()
cv2.destroyAllWindows()