#!/usr/bin/env python

import numpy as np
import cv2

# local modules
from common import splitfn

# built-in modules
import os
import sys
import getopt
from glob import glob

# define image path
imagePath = 'E:/Calibrations/no fish eye/'
debug_dir = imagePath + 'output/'
img_mask = imagePath + '*.jpg'  
img_names = glob(img_mask)
debug_dir = imagePath + 'output/'
if not os.path.isdir(debug_dir):
    os.mkdir(debug_dir)
    
square_size = 1.0
pattern_size = (9, 6)
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

    if debug_dir:
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
rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())

# undistort the image with the calibration
print('')
for img_found in img_names_undistort:
    img = cv2.imread(img_found)
    h,  w = img.shape[:2]
    
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    
    # crop and save the image
    x, y, w, h = roi
#    dst = dst[y:y+h, x:x+w]
    path, name, ext = splitfn(fn)
    outfile = img_found +  '_undistorted.png'
    print('Undistorted image written to: %s' % outfile)
    cv2.imwrite(outfile, dst)

cv2.destroyAllWindows()
