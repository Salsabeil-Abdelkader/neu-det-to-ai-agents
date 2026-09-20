"""
Stage 1.3 — Classical Edge Detection (Canny & Sobel) from First Principles
Dataset: NEU-DET (sample: crazing_1.jpg)

Author: Salsabeil Abdelkader
"""

import cv2
import numpy as np


img = cv2.imread('crazing_1.jpg')
assert img is not None, "could not load the image, check the path or file extension"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
gray = cv2.resize(gray,(600,600))
Gx_kernel = np.outer([1,2,1],[-1,0,1])
Gy_kernel = np.outer([-1,0,1],[1,2,1])


def apply_kernel(gray, kernel):
    output=np.zeros_like(gray)
    padded= np.pad(gray, pad_width=1, mode='constant',constant_values=0)
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            patch=padded[i:i+3, j:j+3]
            output[i,j] = np.sum(patch*kernel)

    return output
value_x = apply_kernel(gray, Gx_kernel)
value_y = apply_kernel(gray, Gy_kernel)
magnitude = np.hypot(value_x, value_y)


direction = np.arctan2(value_y,value_x)

th=np.percentile(magnitude, [90, 95, 99])
_,thresh = cv2.threshold(magnitude, th[0],255, cv2.THRESH_BINARY)


canny=cv2.Canny(gray.astype(np.uint8), 80, 170)
cv2.imshow('gray',gray.astype(np.uint8))

cv2.imshow('canny',canny)
cv2.imshow('thresh',thresh.astype(np.uint8))

cv2.waitKey(0)
cv2.destroyAllWindows()
