#Code taken from here: http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/

import argparse
import datetime
import imutils
import time
import cv2

#Construct the argument parser and parse
ap = argparse.ArgumentParser()
ap.add_argument("-v",'--video', help='path to the video file')
ap.add_argument('-a', '--min-area', type=int, default=500, help='minimum area size')
args = vars(ap.parse_args())

#If the video path is not given, use the webcam
if args.get('video', None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
    
#Otherwise, we read the provided file
else:
    camera = cv2.VideoCapture(args['video'])
    
#Start the first frame
firstFrame = None


#Loop for the duration of the video
while True:
    (grabbed, frame) = camera.read()
    text = 'Unoccupied'
    
    if not grabbed:
        break
    
    #Resize the frame, turn it into gray scale and blur
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    #if the first frame is None, start it
    if firstFrame is None:
        firstFrame = gray
        continue
        
    #Get the abolute difference between the intial frame and the current one
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    #find contours and other stuff
    thresh = cv2.dilate(thresh,  None,  iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE)
        
    #loop over contours
    for c in cnts:
        #if the contour is too small, ignore it
        if cv2.contourArea(c) < args['min_area']:
            continue
        
        #Compute the bounding box and draw it
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = 'Occupied'
    
    #draw the text and time stamp
    cv2.putText(frame, 'Room Status: {}'.format(text), (10, 20), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime('%A %d %B %Y %I:%M:%S%p'), 
        (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
    
    #Show the frame and check if the user presses a key
    cv2.imshow('Security Feed', frame)
    cv2.imshow('Thresh',  thresh)
    cv2.imshow('Frame Delta',  frameDelta)
    key = cv2.waitKey(1) & 0xFF
    
    #if q is pressed, break the loop
    if key == ord('q'):
        break
        

#Cleanup
camera.release()
cv2.destroyAllWindows()
        
    
    
    


