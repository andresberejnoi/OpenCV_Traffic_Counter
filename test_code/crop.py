##--------------------------Import Statements-----------------------------##
import cv2			#Python bindings for OpenCV
import argparse                 #Allows the program to take arguments from the command line
import numpy as np		#API for matrix operations
import math
import multiprocessing as mp

##--------------------------Useful Information----------------------------##
    #flags for the threshold function, with their respective indices:
    # i.e: cv2.threshold(frame,127,255, 0)  where 0 is the flag below.
    #     The 127 - 255 are the threshold range
    #
    # 0 - cv2.THRESH_BINARY
    # 1 - cv2.THRESH_BINARY_INV
    # 2 - cv2.THRESH_TRUNC
    # 3 - cv2.THRESH_TOZERO
    # 4 - cv2.THRESH_TOZERO_INV


    #Example use of image denoising function dilate:
    # d_img = cv2.dilate(frame,kernel=None [,...]) It seems to work



##-------------------------Function Definitions----------------------------##
def clickEvent(event,x,y,flags,userdata):
    global rect
    if event==cv2.EVENT_LBUTTONDOWN:
        rect.append((x,y))
        

##----------------------Setting up the Argument Parser----------------------##

parser = argparse.ArgumentParser(description='Finds the contours on a video file')          #creates a parser object
parser.add_argument('-p','--path',type=str,help="""A video filename or path.
Works better with .avi files.
If no path or name is provided, the camera will be used instead.""")        #instead of using metavar='--path', just type '--path'. For some reason the metavar argument was causing problems
parser.add_argument('-a','--minArea',type=int,help='The minimum area (in pixels) to draw a bounding box',
                    default=120)
parser.add_argument('-d','--direction', type=str,default='H',help="""A character: H or V
representing the orientation of the count line. H is horizontal, V is vertical.
If not provided, the default is horizontal.""")
parser.add_argument('-n','--numCount',type=int,default=10,help="""The number of contours to be detected by the program.""")
parser.add_argument('-w','--webcam',type=int,default=0,help="""Allows the user to specify which to use as the video source""")
args=vars(parser.parse_args())

##------------Initializing the Video Source------------##
if args.get('path',None) is None and args['webcam']==0:           #when a path is not provided, use camera
    cap = cv2.VideoCapture(0)
    #cap.set(3,320)
    #cap.set(4,240)
    #cap.set(3,160)
    #cap.set(4,120)
elif args['webcam'] != 0:
	cap = cv2.VideoCapture(args['webcam'])
	#cap.set(3,320)
	#cap.set(4,240)
else:
    cap = cv2.VideoCapture(args['path'])    #otherwise, use the given path or namec

#-------------------------------------------------------


_,img = cap.read()                          #gets the initial frame
img2 = img.copy()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##------------------------------------------------------
cv2.namedWindow('setup',1)

rect = []
k = None

while k != ord('i'):
    cv2.imshow('setup',img2)
    cv2.setMouseCallback('setup',clickEvent)

    k = cv2.waitKey(10) & 0xFF
#    if k == ord('i'):
#        break
cv2.destroyWindow('setup')


#mask = np.zeros(img.shape,dtype = np.uint8)
#roi = np.array([[rect[0],rect[1],rect[2],rect[3]]])
roi = np.array([rect])


#cv2.fillPoly(mask,roi,(255,255,255))

#average = np.float32(mask)

box = cv2.boundingRect(roi)
x,y,w,h = box

while True:
    
    _,img = cap.read()

    frame = img[x:x+w,y:y+h]
    

    cv2.imshow('cropped',frame)

    k = cv2.waitKey(30) & 0xFF
    if k == 27 or k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











