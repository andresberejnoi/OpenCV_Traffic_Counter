import cv2
#import numpy as np

cap = cv2.VideoCapture(0)

#cv2.startWindowThread()

while (cap.isOpened()):
    #BGR image feed from camera
    ret, img = cap.read()
    cv2.imshow('output', img)
    
    #BGR to gray conversion
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayscale', img2)
    
    #BGR to binary (RED) thresholded image
  #  imgthreshold = cv2.inRange(img, cv2.cv.Scalar(3, 3, 125), cv2.cv.Scalar(40, 40, 255), )
    imgthreshold = cv2.inRange(img, cv2.cv.Scalar(160, 160, 80), cv2.cv.Scalar(200, 200, 160), )
    cv2.imshow('thresholded', imgthreshold)
    
    
    k = cv2.waitKey(10) & 0xFF                              # The & 0xFF is apparently necessary in 64 bit systems in order for this thing (waitKey) to work .
    if k==27:
#        cv2.destroyAllWindows()
        break
    
        
cap.release()
cv2.destroyAllWindows()
