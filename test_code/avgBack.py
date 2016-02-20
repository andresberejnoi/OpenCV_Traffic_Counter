##--------------------------Import Statements-----------------------------##
import cv2
import argparse                 #Allows the program to take arguments from the command line
import numpy as np

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
args=vars(parser.parse_args())



##-------------------------Function Definitions----------------------------##
def boundObjects(frame,thresh,numCnts = 10):
    cnts,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:numCnts]
    global counter,width,halfH,halfW,prev_x,prev_y,minArea
    global p1_count_line,p2_count_line

    for c in cnts:
        if cv2.contourArea(c) < minArea:
            continue

        rect = cv2.minAreaRect(c)
        points = cv2.cv.BoxPoints(rect)
        points = np.int0(points)
        cv2.drawContours(frame,[points],0,(0,255,0),1)

        #Getting the center coordinates of the contour box
        cx = int(rect[0][0])
        cy = int(rect[0][1])

        cv2.circle(frame,(cx,cy),3,(0,0,255),2)

        if args['direction'].upper()=='H':
            if prev_y < p1_count_line[1] < cy or cy <p1_count_line[1]< prev_y:
                counter += 1
                cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),3)
        elif args['direction'].upper()=='V':
            if prev_x < p1_count_line[0] < cx or cx <p1_count_line[0]< prev_x:
                counter += 1
                cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),3)

            
        prev_x = cx
        prev_y = cy

    return frame

#def drawLine(frame,orientation='H',p1,p2,color=(0,0,255)):
#    if orientation.upper()=='H':
#        cv2.line(frame,p1...

##-----------------------Setting Initial Properties------------------------##

##------------Initializing the Video Source------------##
if args.get('path',None) is None:           #when a path is not provided, use camera
    cap = cv2.VideoCapture(0)
    cap.set(3,320)
    cap.set(4,240)
else:
    cap = cv2.VideoCapture(args['path'])    #otherwise, use the given path or name

#---------------setting global variables---------------##
width = int(cap.get(3))
height = int(cap.get(4))
    
halfH = height/2                   #half of the height 
halfW = width/2                   #half of the width

threeFourthH = halfH + int(halfH/2)
threeFourthW = halfW + int(halfW/2)

width = int(cap.get(3))
height = int(cap.get(4))
prev_x,prev_y = 0,0
counter = 0                                 #global object counter
minArea = args['minArea']
rate = 0.01

##----------------------------------------------------------------------------##

_,img = cap.read()                          #gets the initial frame
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
average = np.float32(img)

p1_count_line = None
p2_count_line = None

if args['direction'].upper()=='H' or args.get('direction',None) is None:
    p1_count_line = (0,threeFourthH)
    p2_count_line = (width,threeFourthH)
elif args['direction'].upper()=='V':
    p1_count_line = (halfW,0)
    p2_count_line = (halfW,height)
else: raise ValueError('Expected an "H" or a "V" only')
    
##---------------------------------------MAIN LOOP--------------------------------------------##
while True:
    grabbed,img = cap.read()
    if not grabbed:
        break

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    cv2.accumulateWeighted(img,average,rate)
    result = cv2.convertScaleAbs(average)       #the average background
    fmask = cv2.absdiff(result,img)         #difference between the running average background and the current frame

  ##------Extra blur------##
    fmask = cv2.GaussianBlur(fmask,(21,21),0)
    fmask = cv2.blur(fmask,(28,28))

    _,thresh = cv2.threshold(fmask,40,200,0)

 ##-----Noise reduction-----##
 #   dimg = thresh
    dimg = cv2.erode(thresh,None)
    #dimg = cv2.erode(dimg,None)
    #dimg = cv2.dilate(dimg,None)                  #Noise reduction function
    dimg = cv2.dilate(dimg,None)
    
    cv2.imshow('dilate',dimg)

#-------------------------------------------
    #Setting the boxes for the camshift function
    
    img2 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    boundObjects(img2,dimg,1)
    cv2.line(img2,p1_count_line,p2_count_line,(0,0,255),1)
    #print (counter)


##---------------Showing The Frames-----------------##
    #cv2.imshow('Original',img)
    cv2.imshow('average', result)
    #cv2.imshow('dilate',dimg)
    cv2.imshow('boxes',img2)


##-------------Termination Conditions-------------##
    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        print (counter)
        break

cap.release()
cv2.destroyAllWindows()

