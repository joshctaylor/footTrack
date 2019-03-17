import cv2
import numpy as np
import csv
import sys
from matplotlib import pyplot as plt 

videos = ['E:/data/008 Uncertainty Study/Static/VideoData/J8D01_FWD.mp4']

['E:/data/008 Uncertainty Study/video/undistorted video/K8D02_AFT.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/K8S01_AFT.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/L8D01_AFT.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/L8S01_AFT.mp4']

fwd = ['E:/data/008 Uncertainty Study/video/undistorted video/L8S01_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/I8S01_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/J8D01_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/J8S01_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/K8D02_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/K8S01_FWD.mp4',
'E:/data/008 Uncertainty Study/video/undistorted video/L8D01_FWD.mp4']


times = [[0,0],]
footdown = [[0,0],]
for i in range(0,len(videos)):
    # mouse callback function
    pts = []
    centreline = []
    footPt = []
    videoFile = videos[i]
    print(videoFile)
    video = cv2.VideoCapture(videoFile)
    alpha = 0  # -2.309062788999995  # angle to rotate by (deg)#
    # set the feet up section of the video to rotate and define the centreline
    minute = times[i][0]
    second = times[i][1]
    
    # set the feet down part of the video to define the point where the feet lie
    min2 = footdown[i][0]
    sec2 = footdown[i][1]
    undist = True
    
    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global pts
            print(x,y)
            pts.append((x,y))
            cv2.circle(frame, (x,y), 6, (0,0,255), -1)
    
    def mouse_callback2(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global centreline
            print(x,y)
            centreline.append((x,y))
            cv2.circle(dst, (x,y), 6, (255,0,0), -1)
    
    def mouse_callback3(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global footPt
            print(x,y)
            footPt.append((x,y))
            cv2.circle(dst, (x,y), 6, (0,255,0), -1)
    
    fileName = 'videoData_aftDyn.csv'     
    csvfile = open(fileName,'a')        # open the csv file in append mode
    # get info about the video 
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(5)
    counter = 0
    targetFrame = int(((minute*60)+second) * fps)
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frameID = targetFrame / totalFrames  # frame id is range 0-1
    video.set(cv2.CAP_PROP_POS_FRAMES, targetFrame)
    
    D = np.array([0.03564257, -0.04109487, 0.00167179, -0.0011342, 0.01131636])
    K =  np.array([[736.39241878, 0.0, 954.85240702],
                [   0.0, 739.57526582, 589.60675824],
                [   0.0, 0.0, 1.0]])
    
    ok, frame = video.read()
    h,  w = frame.shape[:2]
    if undist:
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        frame = cv2.undistort(frame, K, D, None, newcameramtx)
    
    if not ok:
        print('Cannot read video file')
        sys.exit()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)  # Mouse callback
    
    while True:
        cv2.imshow("image", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord('x'):
            pts=np.array(pts)
            alpha = np.rad2deg( np.arctan((pts[1,0]-pts[0,0]) / (pts[1,1]-pts[0,1])) )
            break
        if key == ord('h'):
            pts=np.array(pts)
            alpha = -np.rad2deg( np.arctan((pts[1,1]-pts[0,1]) / (pts[1,0]-pts[0,0])) )
            break
        if key == ord(','):
            ok, frame = video.read()
            frame = cv2.undistort(frame, K, D, None, newcameramtx)
            cv2.imshow("image", frame)
    
    cv2.destroyAllWindows()
    print(alpha)
    
    targetFrame = int(((minute*60)+second) * fps)
    totalFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    frameID = targetFrame / totalFrames  # frame id is range 0-1
    video.set(cv2.CAP_PROP_POS_FRAMES, targetFrame)
    
    # re-open a new frame with camera correction and rotation
    
    ok, frame = video.read()
    rows,cols,col = frame.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-alpha,1)
    dst = cv2.warpAffine(frame,M,(cols,rows))
    h,  w = dst.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    dst = cv2.undistort(dst, K, D, None, newcameramtx)
    
    cv2.namedWindow('image2')
    cv2.setMouseCallback('image2', mouse_callback2)  # Mouse callback
    
    while True:
        cv2.imshow("image2", dst)
        key = cv2.waitKey(1) & 0xff
        if key == ord('x'):
            break
        elif key == ord('y'):
            cv2.line(dst, centreline[0], centreline[-1], (255, 0,0), 4)
        elif key == ord(','):
            ok, dst = video.read()
            frame = cv2.undistort(dst, K, D, None, newcameramtx)
            dst = cv2.warpAffine(frame,M,(cols,rows))
            cv2.imshow("image2", dst)
            
    cv2.imwrite(videoFile+'rot_image.png', dst)
    
    # open a third frame showing all previous points with additional point for the foot position
    
    targetFrame2 = int(((min2*60)+sec2) * fps)
    video.set(cv2.CAP_PROP_POS_FRAMES, targetFrame2)
    
    cv2.namedWindow('image3')
    cv2.setMouseCallback('image3', mouse_callback3)  # Mouse callback
    
    ok, frame = video.read()
    frame = cv2.undistort(frame, K, D, None, newcameramtx)
    dst = cv2.warpAffine(frame,M,(cols,rows))
    
    cv2.imshow("image3", dst)
    cv2.line(dst, centreline[0], centreline[-1], (255, 0,0), 4)
    
    while True:
        cv2.imshow("image3", dst)
        key = cv2.waitKey(1) & 0xff
        if key == ord('x'):
            break
        elif key == ord(','):
            ok, dst = video.read()
            frame = cv2.undistort(dst, K, D, None, newcameramtx)
            dst = cv2.warpAffine(frame,M,(cols,rows))
            cv2.line(dst, centreline[0], centreline[-1], (255, 0,0), 4)
            cv2.imshow("image3", dst)
    
    fileWriter = csv.writer(csvfile, delimiter=',',lineterminator='\n',)
    fileWriter.writerow([videoFile, alpha, centreline[0][0], centreline[0][1], centreline[1][0], centreline[1][1], footPt[0][0], footPt[0][1]])
    print([videoFile, alpha, centreline, footPt])
    video.release()
    cv2.destroyAllWindows()
    csvfile.close()


