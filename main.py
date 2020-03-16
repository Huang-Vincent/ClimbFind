import findHold
#import colorSort

img = findHold.openImage()
detect = findHold.makeBlobDetector()
pts = findHold.getHolds(img, detect)
findHold.showHolds(img, pts)
findHold.getColors(img)
