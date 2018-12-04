# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:51:19 2016

@author: bob
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Load an color image in grayscale
img = cv2.imread('messi5.jpg', 0) 


# using pyplot to show the image
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# using open cv to show the image
cv2.namedWindow('image', cv2.WINDOW_NORMAL) # creates a named window
cv2.imshow('image', img) # displays the picture
cv2.waitKey(0)      # waits for a key stroke on the windeow
cv2.destroyAllWindows()

# writing an image to file
cv2.imwrite('messigray.png',img)
