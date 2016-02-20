import cv2
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Finds the contours on a video file')          #creates a parser object
parser.add_argument('-p','--path',type=str,help="""A video filename or path.
Works better with .avi files.
If no path or name is provided, the camera will be used instead.""")

args=vars(parser.parse_args())                  #args is a dictionary with the variables as keys


##------------Initializing the Video Source------------##
if args.get('path',None) is None:           #when a path is not provided, use camera
    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
else:
    cap = cv2.VideoCapture(args['path'])    #otherwise, use the given path or namecap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

_,img = cap.read()


hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_img,np.array((0,60,32)),np.array((180,255,255)))
roi_hist = cv2.calcHist([hsv_img],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

#Set Termination Criteria for the camshift function
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,1)
                   
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

aver = np.float32(img)
rate = 0.02

while True:
    _,img = cap.read()
    img2 = img.copy()
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    backProj = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    img = cv2.blur(img,(2,2))

    cv2.accumulateWeighted(img,aver,rate)
    result = cv2.convertScaleAbs(aver)
    fmask = cv2.absdiff(result,img)
    fmask = cv2.blur(fmask,(21,21))
    
    _,thresh = cv2.threshold(fmask,50,255,0)

        
    dimg = cv2.erode(thresh,None)
    dimg = cv2.erode(dimg,None)
    dimg = cv2.dilate(dimg,None)
    dimg = cv2.dilate(dimg,None)
    

    cnts,hier = cv2.findContours(dimg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]

    for c in cnts:
        if cv2.contourArea(c) < 120:
            continue
        window = cv2.boundingRect(c)                    #this will return 4 points
        rect,window = cv2.CamShift(backProj,window,term_crit)

        pts = cv2.cv.BoxPoints(rect)
        pts = np.int0(pts)

        cv2.drawContours(img2,[pts],0,(0,255,0),1)
    
    #hsv = cv2.cvtColor(img, cv2.BGR2HSV)
    #dst = cv2.calcBackProject([hsv],[0],

##------------------------------------------------------
    cv2.imshow('CamShift',img2)
    cv2.imshow('average',result)
    cv2.imshow('absdiff',fmask)
    cv2.imshow('thresh',thresh)
#    cv2.imshow('noise Redu',dimg)

    k = cv2.waitKey(10) & 0xFF
    if k ==27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
