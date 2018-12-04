#!/usr/bin/env python

import numpy as np
import cv2
from matplotlib import pyplot as plt

# local modules
from common import splitfn

# built-in modules
import os

import sys
import getopt
from glob import glob

# define image path

imagePath = 'E:/Calibrations/fwdcam/'
square_size = 1.0
pattern_size = (9, 6)

img_mask = imagePath + '*.jpg'  
img_names = glob(img_mask)
img_names = img_names[0:4]
debug_dir = imagePath + 'output/'
if not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = 0, 0
img_names_undistort = []
for fn in img_names:
    print('processing %s... ' % fn, end='')
    img = cv2.imread(fn, 0)
    if img is None:
        print("Failed to load", fn)
        continue

    h, w = img.shape[:2]
    found, corners = cv2.findChessboardCorners(img, pattern_size)
    if found:
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
        
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawChessboardCorners(vis, pattern_size, corners, found)
    path, name, ext = splitfn(fn)
    outfile = debug_dir + name + '_chess.png'
    cv2.imwrite(outfile, vis)
    if found:
        img_names_undistort.append(outfile)

    if not found:
        print('chessboard not found')
        continue

    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)

    print('ok')

# calculate camera distortion
n = np.product(pattern_size)
obj_points2=np.asarray([obj_points],dtype='float64').reshape(-1,1,n,3)
img_points2=np.asarray([img_points],dtype='float64').reshape(-1,1,n,2)
camera_matrix = np.eye(3)
dist_coeffs = np.zeros(4)
rvecs=np.asarray([[[np.zeros(3).tolist() for i in range(len(img_points))]]],dtype='float64').reshape(-1,1,1,3)
tvecs=np.asarray([[[np.zeros(3).tolist() for i in range(len(img_points))]]],dtype='float64').reshape(-1,1,1,3)

calib_flags=None #cv2.fisheye.CALIB_USE_INTRINSIC_GUESS  # set the flags

ret, camera_matrix, dist_coeffs, rvecs, tvecs=cv2.fisheye.calibrate(obj_points2, img_points2, (w, h), camera_matrix, dist_coeffs)

print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coeffs.ravel())

# undistort the image with the calibration
print('')
#for img_found in img_names_undistort:
img = cv2.imread(img_names_undistort[1])

#Knew = camera_matrix.copy()
#Knew[(0,1), (0,1)] = 0.4 * Knew[(0,1), (0,1)]
Knew = np.eye(3)
h,  w = img.shape[:2]
K = camera_matrix
D = dist_coeffs
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), .5)
dst = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)

# crop and save the image
x, y, w, h = roi
#dst = dst[y:y+h, x:x+w]
outfile = img_found + '_undistorted.png'
print('Undistorted image written to: %s' % outfile)
cv2.namedWindow('image')

while 1:
    cv2.imshow('image', dst)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
