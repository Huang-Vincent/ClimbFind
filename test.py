#using opencv version 4.1.2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

#create a simple gui to select photo
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

#load colored image, resize to increase speed
img = cv.imread(file_path, 1)
scale = 1000.0/img.shape[0]
height = int(img.shape[1] * scale)
img = cv.resize(img, (height,1000))

#apply blur to remove noise
blur = cv.GaussianBlur(img,(5,5),0)

#image must be grayscale for OTSU's thresholding to work
gray = cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
ret,th = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

#edge detection
edges = cv.Canny(blur,ret,ret*2,L2gradient = True)
contours, heir = cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

#apply a convex to each contour to close it
hull_list = []
for i in range(len(contours)):
    hull = cv.convexHull(contours[i])
    hull_list.append(hull)

detector = cv.SimpleBlobDetector()
keypoints = detector.detect(img)

#display the outlines of each hold
mask = np.zeros(gray.shape, np.uint8)
hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)

for i in range(0,len(contours)):
    mask[...]=0
    cv.drawContours(mask,hull_list,i,255,-1)
    mean = cv.mean(hsv, mask=mask)
    #display holds of certain color by comparing hsv values
    if mean[0] > 99 and mean[0] < 119 and mean[1] > 50 and mean[2] > 100:
        cv.drawContours(img, hull_list, i,255 , 2)


    # hsv = cv.cvtColor(mean, cv.COLOR_GRAY2HSV)
    # mask = cv.inRange(hsv, lower_color, upper_color)
    # final = cv.bitwise_and(img, img, mask=mask)

cv.imshow("final", img)
cv.waitKey(0)
