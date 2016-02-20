import cv2
import time
import argparse
import numpy as np

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
parser.add_argument('-s','--show',type=int,default=1,help="""0 or 1 indicating if the windows for the
images should be displayed. 0 is False and 1 is True. The default value is 1""")

args=vars(parser.parse_args())

##-------------------------Function Definitions----------------------------##
def clickEvent(event,x,y,flags,userdata):
    global rect
    if event==cv2.EVENT_LBUTTONDOWN:
        rect.append((y,x))                  #Numpy manages the coordinates as (y,x) instead of (x,y) like the rest of the world

##-----------------------Setting Initial Properties------------------------##
GUI = args['show']                          #A boolean indicating if the GUI should be used

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
##----------------------------------------------------------------------------##

_,img = cap.read()                          #gets the initial frame
img2 = img.copy()
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#average = np.float32(img)

##---------------Setting up ROI---------------------------------------

if GUI:
    cv2.namedWindow('setup',1)
    k = None
    rect = []
    cv2.imshow('setup',img2)
    while k != ord('q') and k!=27:
        cv2.setMouseCallback('setup',clickEvent)
        k = cv2.waitKey(0) & 0xFF

    cv2.destroyWindow('setup')
    roi = np.array([rect])
    box = cv2.boundingRect(roi)
    x,y,w,h = box
else:
    size = raw_input("""Enter the ROI information:
x,y,w,h; where (x,y)is the left uppermost corner,
w is width, h is height. Enter the numbers separated by commas:
""")
    size = [int(n) for n in size.split(',')] 
    y,x,h,w = size              #x,y are switched because of the way numpy uses them for arrays
    

roi_mask = img[x:x+w,y:y+h]
#cv2.imshow('selection',roi_mask)
#cv2.waitKey(10) & 0xFF


while True:
    grabbed,img = cap.read()
    if not         break
    roi_mask = img[x:x+w,y:y+h]


    #cv2.imshow('image',img)
    cv2.imshow('cropped',roi_mask)
    #time.sleep(0.075)

    k = cv2.waitKey(60) & 0xFF
    if k == 27 or k == ord('q'):
        break

print(str(y),str(x),str(h),str(w))
cap.release()
cv2.destroyAllWindows()
















