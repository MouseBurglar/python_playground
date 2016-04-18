import cv2 as cv
import matplotlib as plt


hessian = 8000
img = cv.imread('messi.jpg', 1)
surf = cv.SURF(hessian)
kp, des = surf.detectAndCompute(img, None)

print len(kp)

img2 = cv.drawKeypoints(img, kp, None, (0, 255, 0), 4)
cv.imshow('CV', img2)
cv.waitKey(0)
cv.destroyAllWindows()

"""
cv.namedWindow('Guatemala', cv.WINDOW_NORMAL)
cv.imshow('Guatemala', img)
cv.waitKey(0)
cv.destroyAllWindows()
"""