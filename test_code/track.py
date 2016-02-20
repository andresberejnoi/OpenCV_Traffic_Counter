import cv2
import numpy as np

def imageDiff(a, b, c):
    '''Substracts one image from the next and returns the difference.
    The difference can be used as a mask image that only shows the motion.'''
    t0 = cv2.absdiff(a, b)
    t1 = cv2.absdiff(b, c)
        
    t2 = cv2.bitwise_and(t0, t1)
    return t2
    
def findCenter(a, b, c, d):
    '''Find the center of a rectangular area, (for rotated rectangles).
    a: the bottommost point; b: leftmost point; c: uppermost point; d: rightmost point'''
    
cap = cv2.VideoCapture(0)

bgSub = cv2.BackgroundSubtractorMOG()

while True:
    img = cap.read()[1]
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgray = cv2.bilateralFilter(imgray, 11, 17, 17)
    fmask = bgSub.apply(imgray)
    
    cv2.imshow('fmask', fmask)
    cv2.imshow('Original', imgray)
    
    #Another threshold using the canny algorithm to detect edges.
    thresh = cv2.Canny(fmask, 0, 1)
    cv2.imshow('thresh', thresh)
    
    #Find contours:
#    contours,  hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
#    cv2.imshow('fmask after', fmask)
    
    contours, _ = cv2.findContours(fmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    im = img.copy()
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100: continue
        print cv2.contourArea(c)
        x,y,w,h = rect
        cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(im,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
    cv2.imshow("Show",im)
        
#    cnts,  hchy =cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    cv2.imshow('thresh after', thresh)
    
    #Draw contours:
    img2 = img.copy()
    cv2.drawContours(img2, contours, -1, (0, 255, 0), -1)
    cv2.imshow('Contours', img2)
    
#    img3 = img.copy()
#    cv2.drawContours(img3, cnts, -1, (255, 0, 0), 1)
#    cv2.imshow('Cont', img3)
    
    
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q') or k == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
    
    
    
    

#cap = cv2.VideoCapture(0)
#
#backSub = cv2.BackgroundSubtractorMOG2()
#
##img = cap.read()[1]
####-----------------------------------------------------------------------------------------------------------------------------------
#
###------------------------------------------------------------------------------------------------------------------------------------
#
#t = cap.read()[1]
#tp = cap.read()[1]
#tpp = cap.read()[1]
#
#t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
#tp = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
#tpp = cv2.cvtColor(tpp, cv2.COLOR_BGR2GRAY)
#    
#while True:
#    img = imageDiff(t, tp, tpp)
#    cv2.imshow('Motion', img)
#    
#    img = cap.read()[1]
#    
#    t = tp
#    tp = tpp
#    tpp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    
#    #Applying a background subtraction on the image
#    fmask = backSub.apply(img)
#    cv2.imshow('fmask', fmask)
#    
##    ret,  thresh = cv2.threshold(fmask, 127, 255, 0)
#    thresh = cv2.Canny(fmask, 200, 201)
#    cv2.imshow('thresh', thresh)
#    
#    k = cv2.waitKey(1) & 0xFF
#    if k == 27 or k == ord('q'):
#        break
#
#cap.release()
#cv2.destroyAllWindows()

