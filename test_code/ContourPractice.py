import cv2
import numpy as np
def imageDiff(a, b, c):
    '''Substracts one image from the next and returns the difference.
    The difference can be used as a mask image that only shows the motion.'''
    t0 = cv2.absdiff(a, b)
    t1 = cv2.absdiff(b, c)
        
    t2 = cv2.bitwise_and(t0, t1)
    return t2

def cont2():
    cap = cv2.VideoCapture(0)
    
    bgmask = cv2.BackgroundSubtractorMOG()
    
    while cap.isOpened():
        img = cap.read()[1]
        fmask = bgmask.apply(img)
        thresh = cv2.Canny(fmask, 200, 201)
        
        #Show images:
        cv2.imshow('Original', img)
        cv2.imshow('fmask',  fmask)
        cv2.imshow('thresh', thresh)
        
        #Processing the images to test something
        #Mmask = cv2.bitwise_and(img, fmask)
        #cv2.imshow('Mmask', Mmask)
        
        
        
        
        
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or k == ord('q'):
            cap.release()
    
    cv2.destroyAllWindows()
        
        
    










def contourTest():
    #cap = cv2.VideoCapture('Video11.avi')
    cap = cv2.VideoCapture(0)

    i = 0
    while True:

        img = cap.read()[1]
        
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgray = cv2.bilateralFilter(imgray, 11, 17, 17)
        cv2.imshow('filter', imgray)
        
        #ret,  thresh = cv2.threshold(imgray, 127, 255, 0)
        thresh = cv2.Canny(imgray, 200, 201)                                        #Canny seems to give better results than threshold
        cv2.imshow('Thresh Before', thresh)
        
        #print thresh
       # cv2.imshow('threshold', thresh)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]             #sorts the list of contours and only keeps the biggest 10
        
        #cv2.drawContours(thresh, contours, -1, (0, 0, 255), 10)
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
        cv2.imshow('Test Contour', img)
    #
    #    cnt = contours[0]               #contours is a list with all the contours detected. So, cnt is one of the contours in the frame
    #    M = cv2.moments(cnt)
    #
    #   # Getting the centroid coordinates:
    #    cx = int(M['m10']/M['m00'])
    #    cy = int(M['m01']/M['m00'])
    #    
    #   # Getting a frame with the center detected
    #    detected = cv2.circle(img, (cx, cy), 10, (0, 0, 255), 1)
    #    
    #   # Displaying the image:
    #    cv2.imshow('Experiment',  detected)
    #    cv2.imshow('Experiment',  thresh)
    #    if i % 1000 == 0:
    #        print (type(detected))
    #    
        k = cv2.waitKey(10) & 0xff
        if k == 27 or k == ord('q'):
            break
            
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    
contourTest()
    
    
