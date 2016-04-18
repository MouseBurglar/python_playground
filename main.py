import numpy as np
import cv2
import matplotlib as plt


"""
hessian = 400
img = cv2.imread('pp.jpg', 1)
surf = cv2.SURF(hessian)
kp, des = surf.detectAndCompute(img, None)

print len(kp)

img2 = cv2.drawKeypoints(img, kp, None, (0, 255, 0), 4)
cv2.namedWindow('Guatemala', cv.WINDOW_NORMAL)
cv2.imshow('Guatemala', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

"""
cv.namedWindow('Guatemala', cv.WINDOW_NORMAL)
cv.imshow('Guatemala', img)
cv.waitKey(0)
cv.destroyAllWindows()
"""

img1 = cv2.imread('template.jpg', 0)          # queryImage
img2 = cv2.imread('watch.jpg', 0) # trainImage
hessian = 400

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print len(good)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, flags=2)

cv2.imshow('matches', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
