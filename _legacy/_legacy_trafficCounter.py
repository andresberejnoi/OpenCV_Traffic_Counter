#!/usr/bin/env python
##--------------------------Import Statements-----------------------------##
import cv2			#Python bindings for OpenCV
import argparse                 #Allows the program to take arguments from the command line
import numpy as np		#API for matrix operations
import time
import os 
import imutils

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

  ##
  #The counting system is very fragile and simple. It is also very easily implemented
    # but it probably has less than 70 or 60% accuracy for counting vehicles. At least it is
    #able to run on the Raspberry Pi B if the right set up is present.
##----------------------Setting up the Argument Parser----------------------##

parser = argparse.ArgumentParser(description='Finds the contours on a video file')          #creates a parser object
parser.add_argument('-p','--path',type=str,help="""A video filename or path.
Works better with .avi files.
If no path or name is provided, the camera will be used instead.""")        #instead of using metavar='--path', just type '--path'. For some reason the metavar argument was causing problems
parser.add_argument('-a','--minArea',type=int,help='The minimum area (in pixels) to draw a bounding box',
                    default=120)
parser.add_argument('-d','--direction', type=str,default=['H','0.5'],nargs=2,help="""A character: H or V
representing the orientation of the count line. H is horizontal, V is vertical.
If not provided, the default is horizontal. The second parameter
is a float number from 0 to 1 indicating the place at which the
line should be drawn.""")
parser.add_argument('-n','--numCount',type=int,default=10,help="""The number of contours to be detected by the program.""")
parser.add_argument('-w','--webcam',type=int,nargs='+',help="""Allows the user to specify which to use as the video source""")
parser.add_argument('--rgb',action='store_true',help="Boolean flag to use rbg colors. Default is to use grayscale")
parser.add_argument('-v','--video_out',type=str,default="",help="Provide a video filename to output")
args=vars(parser.parse_args())



##-------------------------Function Definitions----------------------------##
def info(real_num,calc_num):

    real_num = float(real_num)
    calc_num = float(calc_num)
    d = abs(real_num-calc_num)

    result = (d/real_num)*100

    print("""Accuracy: {0}%""".format(round(100-result,2)))
    print("""Error: {0}%""".format(round(result,2)))
    #return str_result

# ---------Event handlers
def clickEvent(event,x,y,flags,userdata):
    global rect
    if event==cv2.EVENT_LBUTTONDOWN:
        rect.append((y,x))                  #Numpy manages the coordinates as (y,x) instead of (x,y)

def clickEventPoly(event,x,y,flags,userdata):
    global poly
    if event==cv2.EVENT_LBUTTONDOWN:
        poly.append((x,y))

def saveToDisk(event,x,y,flags,userdata):
    global img2, window_mask, roi_mask,result
    if event==cv2.EVENT_LBUTTONDOWN:
        cv2.imwrite("""img2_{0}.jpeg""".format((str(x),str(y))),img2)
        cv2.imwrite("""win_mask_{0}.jpeg""".format((str(x),str(y))),window_mask)
        cv2.imwrite("""roi_mask_{0}.jpeg""".format((str(x),str(y))),roi_mask)
        cv2.imwrite("""res_{0}.jpeg""".format((str(x),str(y))),result)



##=============================

def findClosestPoint(p1,p2):
    '''Compares two points (2D) and returns their euclidean distance.
    It might be more efficient to use numpy's linalg.norm() function.'''

    dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    return dist


def boundObjects(frame,thresh):
    global counter,width,halfH,halfW,prev_x,prev_y,minArea,numCnts
    global p1_count_line,p2_count_line,font,ctrs,GUI

    cnts,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)                #this line is for opencv 2.4, and also now for OpenCV 4.4, so this is the current one
    #_,cnts,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)               #here is how it is done in opencv 3.1; (findContours() returns an extra value)
    cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:numCnts]

    index = 1
    current_cent = []       #a list of the centroids in the current frame
    for c in cnts:
        if cv2.contourArea(c) < minArea:
            continue

        rect = cv2.minAreaRect(c)
        #points = cv2.cv.BoxPoints(rect)            #This line was good for opencv 2.4, but in version 3.1, the cv part is removed:
        points = cv2.boxPoints(rect)                # This is the way to do it in opencv 3.1
        points = np.int0(points)

        #Getting the center coordinates of the contour box
        cx = int(rect[0][0])
        cy = int(rect[0][1])

        w,h = rect[1]                   #Unpacks the width and height of the frame

        C = np.array((cx,cy))
        current_cent.append((cx,cy))

        #Finding the centroid of c in the previous frame
        if len(ctrs)==0: prev_x,prev_y = cx,cy
        elif len(cnts)==0: prev_x,prev_y = cx,cy
        else:
            minPoint = None
            minDist = None
            for i in range(len(ctrs)):
                dist = np.linalg.norm(C-ctrs[i])                #numpy's way to find the euclidean distance between two points
                if (minDist is None) or (dist < minDist):
                    minDist = dist
                    minPoint = ctrs[i]
            #This if is meant to reduce overcounting errors
            if not minDist>=w/2:
                prev_x,prev_y = minPoint
            else: prev_x,prev_y = cx,cy
        #ctrs = current_cent


        #Determines if the line has been crossed
        if args['direction'][0].upper()=='H':
            if (prev_y <= p1_count_line[1] <= cy) or (cy <= p1_count_line[1] <= prev_y):
                counter += 1
                cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),5)
                print (counter)
        elif args['direction'][0].upper()=='V':
            if (prev_x <= p1_count_line[0] <= cx) or (cx <= p1_count_line[0] <= prev_x):
                counter += 1
                cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),5)
                print (counter)

        #cv2.line(frame,(int(points[0][0]),int(points[0][1])),(int(points[0][0])+int(w/2),int(points[0][1])),(0,0,255))
        cv2.drawContours(frame,[points],0,(0,255,0),1)
        cv2.line(frame,(prev_x,prev_y),(cx,cy),(0,0,255),1)         #A line to show motion
        cv2.circle(frame,(cx,cy),3,(0,0,255),4)
        #cv2.putText(frame,'('+str(cx)+','+str(cy)+')',(cx,cy),font,0.3,(0,255,0),1)
        cv2.putText(frame,str(index),(cx,cy-15),font,0.4,(255,0,0),2)

        index += 1

    ctrs = current_cent


##-----------------------Setting Initial Properties------------------------##
#real_num = float(raw_input("""[For How many cars are in the video? :"""))
##------------Initializing the Video Source------------##
if args['path'] is None and args['webcam']is None:           #when a path is not provided, use camera
    cap = cv2.VideoCapture(0)
    #cap.set(3,320)
    #cap.set(4,240)
    #cap.set(3,160)
    #cap.set(4,120)
elif args['webcam']is not None:
    cap = cv2.VideoCapture(args['webcam'][0])
    if len(args['webcam']) > 1:             #if the desired video resolution is indicated, use it
       cap.set(3,args['webcam'][1])
       cap.set(4,args['webcam'][2])

else:
    cap = cv2.VideoCapture(args['path'])    #otherwise, use the given path or namec


##----------------------------------------------------------------------------##

_,img = cap.read()                          #gets the initial frame

grabbed,img = cap.read()
while not grabbed:
    grabbed,img = cap.read()    #just making sure that a valid frame is read

img = imutils.resize(img,width=640)
#vid_height,vid_width,_ = img.shape


img2 = img.copy()
#img2 = imutils.resize(img2,width=640)
if not args['rgb']:
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#average = np.float32(img)

##---------------Setting up ROI---------------------------------------
ALL_WINDOW = True
#cv2.namedWindow('setup',1)
k = None
rect = []
cv2.imshow('setup',img2)
while k != ord('q') and k != ord('Q') and k != 27 and k != ('\n'):
    cv2.setMouseCallback('setup',clickEvent)
    k = cv2.waitKey(0) & 0xFF

cv2.destroyWindow('setup')


if len(rect) > 0:
    ALL_WINDOW = False
    roi = np.array([rect])
    box = cv2.boundingRect(roi)
    x,y,w,h = box
    roi_mask = img[x:x+w,y:y+h]

else:
    roi_mask = img

#vid_height,vid_width,_ = roi_mask.shape
vid_height = roi_mask.shape[0]
vid_width  = roi_mask.shape[1]

#cap.set(cv2.CAP_PROP_FRAME_WIDTH,vid_width)   # float
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,vid_height)  # float

poly = []
print('Now select the exact ROI you want')
k=None
cv2.namedWindow('setup2',1)
cv2.imshow('setup2',roi_mask)

while k!=ord('q') and k != ord('Q') and k != 27:
    cv2.setMouseCallback('setup2',clickEventPoly)
    k = cv2.waitKey(0) & 0xFF
    if k ==ord('\n'):
        poly = []
        break

cv2.destroyWindow('setup2')
roi_poly = np.array([poly])

black_mask = None
print(poly)
if len(poly)!= 0:
    black_mask = np.zeros(roi_mask.shape,dtype=np.uint8)
    cv2.fillPoly(black_mask,roi_poly,(255,255,255))

    average = np.float32(black_mask)
else: average = np.float32(roi_mask)

average = cv2.resize(average, (vid_width,vid_height))

#---------------setting global variables---------------##
width = roi_mask.shape[1]
height = roi_mask.shape[0]

counter = 0                                 #global object counter
minArea = args['minArea']
numCnts = args['numCount']
rate = 0.01
ctrs = []                                   #a list of the centroids from the previous frames
font = cv2.FONT_HERSHEY_SIMPLEX
frame_num = 0                   #counts the current frame number
##----------------------------------------------------------------------------##

p1_count_line = None
p2_count_line = None

args['direction'][1] = float(args['direction'][1])
if args['direction'][0].upper()=='H' or args.get('direction',None) is None:
    fract = int(height*args['direction'][1])
    p1_count_line = (0,fract)
    p2_count_line = (width,fract)
elif args['direction'][0].upper()=='V':
    fract = int(width*args['direction'][1])
    p1_count_line = (fract,0)
    p2_count_line = (fract,height)
else: raise ValueError('Expected an "H" or a "V" only')

##-----------------------------------------------------------------------------------------##
#||||||||||||||||||||||||||||||||||||||| MAIN LOOP ||||||||||||||||||||||||||||||||||
##------------------------------------------------------------------------------------
screenshot_folder = "_screenshots"
video_out_folder  = "_videos"
if not os.path.exists(screenshot_folder):
    os.mkdir(screenshot_folder)

#-------- Creating video writer
video_out = False
if len(args['video_out']) > 0:
    video_out = True
    video_res = (int(vid_width),int(vid_height))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_format = '.avi'
    if not os.path.exists(video_out_folder):
        os.mkdir(video_out_folder)

    print(f'Output Resolution: {video_res}')
    base_output_video_name = args['video_out']
    #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    mjpg = 'MJPG'
    xdiv = 'XDIV'
    divx = 'DIVX'
    x264 = 'X264'
    mp4v = 'MP4V'
    fourcc = cv2.VideoWriter_fourcc(*mjpg)
    screenshot_out  = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_screenshot'+ video_format),
                                     fourcc,video_fps,video_res)
    win_mask_out    = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name +'_win_mask'+ video_format),
                                     fourcc,video_fps,video_res)
    roi_mask_out    = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_roi_mask'+ video_format),
                                     fourcc,video_fps,video_res)
    res_mask_out    = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_res_mask'+ video_format),
                                     fourcc,video_fps,video_res)
    dimg_mask_out   = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_dimg_mask'+ video_format),
                                     fourcc,video_fps,video_res)
    thresh_mask_out = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_thresh_mask'+ video_format),
                                     fourcc,video_fps,video_res)
    fmask_out       = cv2.VideoWriter(os.path.join(video_out_folder, base_output_video_name + '_fmask'+ video_format),
                                     fourcc,video_fps,video_res)

    

init_time = time.time()
while 1:
    grabbed,img = cap.read()
    if not grabbed:
        break
    #--------------
    img = cv2.resize(img, (vid_width,vid_height))
    if not ALL_WINDOW:
        roi_mask = img[x:x+w,y:y+h]
    else: roi_mask = img

    if not args['rgb']:
        roi_mask = cv2.cvtColor(roi_mask,cv2.COLOR_BGR2GRAY)

    if not black_mask is None:
        window_mask = cv2.bitwise_and(roi_mask,black_mask)
    else:
        window_mask = roi_mask
    #--------------

    #print(f'Window_mask shape: {window_mask.shape}')
    #print(f'average shape: {average.shape}')
    if frame_num < 1:            #Hardcoded value indicating how many frames to let pass once the video begins
        frame_num += 1
        cv2.accumulateWeighted(window_mask,average,rate)
        continue
    cv2.accumulateWeighted(window_mask,average,rate)
    result = cv2.convertScaleAbs(average)       #the average background
    fmask = cv2.absdiff(result,window_mask)         #difference between the running average background and the current frame

    ##------Extra blur------##
    
    fmask = cv2.GaussianBlur(fmask,(21,21),0)
    fmask = cv2.blur(fmask,(28,28))
    fmask = cv2.GaussianBlur(fmask,(21,21),0)
    fmask = cv2.GaussianBlur(fmask,(21,21),0)
    fmask = cv2.GaussianBlur(fmask,(21,21),0)

    _,thresh = cv2.threshold(fmask,30,255,0)

    

 ##-----Noise reduction-----##
    #dimg = cv2.erode(thresh,None)      #Dilate function expands the white contours
    dimg = cv2.dilate(thresh,None)
    dimg = cv2.dilate(dimg,None)
    dimg = cv2.dilate(dimg,None)
    dimg = cv2.dilate(dimg,None)                  #Noise reduction function
    dimg = cv2.dilate(dimg,None)
    #dimg = cv2.dilate(dimg,None)
    cv2.imshow('dilate',dimg)
    dimg2 = dimg.copy()                             #A copy just to keep the original intact
#-------------------------------------------
    #Setting the boxes for the bounding process
    if not args['rgb']:
        img2 = cv2.cvtColor(window_mask,cv2.COLOR_GRAY2BGR)
    else:
        img2 = window_mask.copy()
    boundObjects(img2,dimg)
##---------------Showing The Frames-----------------##
    cv2.imshow('roi',roi_mask)
    cv2.imshow('polygon',window_mask)
    cv2.imshow('average', result)
    cv2.line(img2,p1_count_line,p2_count_line,(0,0,255),1)
    cv2.imshow('boxes',img2)
    cv2.imshow('fmask',fmask)
##-------------Termination Conditions-------------##
    #cv2.setMouseCallback('boxes',saveToDisk)                #When the window is clicked, a screenshot is saved to the current directory
    k = cv2.waitKey(40) & 0xFF
    if k == 27 or k == ord('q') or k == ord('Q'):
        break

    if k == ord('s') or k == ord('S'):            #if the letter s/S is pressed, a screenshot of the current frame on each window will be saved to the current folder
        frame_id = cap.get(1)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_screenshot.jpeg"),img2)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_win_mask.jpeg"),window_mask)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_roi_mask.jpeg"),roi_mask)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_res.jpeg"),result)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_dimg.jpeg"),dimg2)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_thresh.jpeg"),thresh)
        cv2.imwrite(os.path.join(screenshot_folder,f"{frame_id}_fmask.jpeg"),fmask)
        
    if video_out:
        screenshot_out.write(img2)
        win_mask_out.write(window_mask)
        roi_mask_out.write(roi_mask)
        res_mask_out.write(result)
        dimg_mask_out.write(dimg2)
        thresh_mask_out.write(thresh)
        fmask_out.write(fmask)

#print("""Time on main loop: {0} secs""".format(str(round(time.time()-init_time,2))))
print('Vehicles Detected: '+str(counter))
#info(real_num,counter)

cap.release()
if video_out:
    screenshot_out.release()
    win_mask_out.release()
    roi_mask_out.release()
    res_mask_out.release()
    dimg_mask_out.release()
    thresh_mask_out.release()
    fmask_out.release()

cv2.destroyAllWindows()
