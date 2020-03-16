#using opencv version 4.1.2
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import math

def openImage():
    #create a simple gui to select photo
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    #load colored image, resize to increase speed
    img = cv.imread(file_path, 1)
    scale = 1000.0/img.shape[0]
    height = int(img.shape[1] * scale)
    img = cv.resize(img, (height,1000))
    return img

#returns the hull list (outlines) of each hold detected
def getHolds(img, detector):
    #image must be grayscale for Otsu's thresholding to work
    #apply blur to remove noise
    #convert to hsv and increase the hue to increase contrast between colors
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    #using Otsu's thresholding to determine holds
    #this was determined by experimenting with various binarization methods
    ret, bin_img = cv.threshold(blur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, th = cv.threshold(bin_img, 127,255, 0)

    contours, heir = cv.findContours(th,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape, np.uint8)
    cv.drawContours(mask, contours, -1, (255, 255, 255), -1)

    cv.imshow("mask",mask)
    cv.waitKey(0)

    #use a simple blob detector to ensure the area and inertia of each hold
    #make sure to run makeBlobDetector() to create the detector
    points = detector.detect(mask)
    return points

#this is used to remove edges create from the wall or faint shadows
#look at the inertia and minimum area to remove any noise
def makeBlobDetector():
    default = cv.SimpleBlobDetector_Params()
    # Change thresholds
    default.filterByColor = False

    # Filter by Area.
    default.filterByArea = True
    default.minArea = 25

    # Filter by Circularity
    default.filterByCircularity = False

    # Filter by Convexity
    default.filterByConvexity = False
    default.minConvexity = 0.03

    # Filter by Inertia
    default.filterByInertia = True
    default.minInertiaRatio = 0.05

    ret = cv.SimpleBlobDetector_create(default)
    return ret

def showHolds(img, keypoints):
    #draw rectangles around each keypoint
    for i, key in enumerate(keypoints):
        x = int(key.pt[0])
        y = int(key.pt[1])

        size = int(math.ceil(key.size))

        #Finds a rectangular window in which the keypoint fits
        top = (x + size, y + size)
        bot = (x - size, y - size)
        cv.rectangle(img,top,bot,(0,0,255),2)

    # Display the resulting frame
    cv.imshow("final",img)
    cv.waitKey(0)

#get all the colors of each holdDetector
#This does not work for holds on volumes
def getColors(img):
    #image must be grayscale for Otsu's thresholding to work
    #apply blur to remove noise
    #convert to hsv and increase the hue to increase contrast between colors
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),0)

    #using Otsu's thresholding to determine holds
    #this was determined by experimenting with various binarization methods
    ret, bin_img = cv.threshold(blur,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, th = cv.threshold(bin_img, 127,255, 0)

    contours, heir = cv.findContours(th,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)

    #use mask as a temp to get the color of each contour
    final = np.zeros(img.shape,np.uint8)
    mask = np.zeros(gray.shape, np.uint8)
    cv.drawContours(mask, contours, -1, (255, 255, 255), -1)

    #sort contours by size
    sortedCnt = sorted(contours, key=lambda x: cv.contourArea(x))
    hull = []
    for i in range(0, len(sortedCnt)):
        hull.append(cv.convexHull(sortedCnt[i]))

    #close each contour and find the average color of it
    for i in range(len(hull)-1, -1, -1):
        mask[...]=0
        cv.drawContours(mask,hull, i, (255, 255, 255), -1)
        cv.drawContours(final,hull,i,cv.mean(img,mask),-1)




    cv.imshow('img',img)
    cv.imshow('final',final)
    cv.waitKey(0)
