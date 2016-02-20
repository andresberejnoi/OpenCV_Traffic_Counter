import cv2
import numpy as np
import argparse

#Start the frame of video and the points of the tracked area
frame = None                        #Current frame being processed
roiPts = []                               # The list of points for our ROI (Region Of Interest)
inputMode = False                   #This is going to be used to speciy if we are currently selecting the object we want to track


def selectROI(event, x, y, flags, param):
    '''Selects the ROI (Region Of Interest) in a video'''
    global frame,  roiPts,  inputMode
    
    
    
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y))
        cv2.circle(frame,  (x, y),  4,  (0, 255, 0), 2)
        cv2.imshow('Frame', frame)
        

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--video', 
        help = "path to the (optional) video file")
    args = vars(ap.parse_args())
    
    global frame,  roiPts, inputMode
    
    #If the video path is not provided, use input from the camera:
    if not args.get("video", False):
        camera = cv2.VideoCapture(0)
    
    #If the path is provided, open the file
    else:
        camera = cv2.VideoCapture(args['video'])
        
    #Setup the mouse callback:
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame",  selectROI)
    
    #Set up the termination criteria for camshift
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,  10, 1)
    roiBox = None
    
    #keep looping over the frames
    while True:
        (grabbed, frame) = camera.read()
        
        #Check to see if we have reached the end of video
        if not grabbed:
            break
            
        #Check if the ROI has been computed
        if roiBox is not None:
            #Convert frame to hsv
            #then perform the mean shift
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            
            #apply camshift to the back projection and draw the box
            (r, roiBox) = cv2.CamShift(backProj, roiBox, termination)
            pts = np.int0(cv2.cv.BoxPoints(r))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
        
        
        #show the frame and record if the user presses a key
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xff
        
        #handle if the 'i' key is pressed, then go into ROI selection mode
        if key == ord('i') and len(roiPts) < 4:
            #indicate that we are in input mode and clone the frame
            inputMode = True
            orig = frame.copy()
            
            #Keep looping until the ROI points have been selected
            #press any key to exit ROI selection
            while len(roiPts)<4:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
                
            #determine the top-left
            roiPts = np.array(roiPts)
            s = roiPts.sum(axis = 1)
            tl = roiPts[np.argmin(s)]
            br = roiPts[np.argmax(s)]
            
            #get the ROI for the bounding box and convert it to hsv color space
            roi = orig[tl[1]:br[1], tl[0]:br[0]]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            #Compute the hsv histogram and store the box
            roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
            roiBox = (tl[0], tl[1], br[0], br[1])
            
        elif key == ord('q'):
            break
            
    camera.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
            
    
    
        
        
        
