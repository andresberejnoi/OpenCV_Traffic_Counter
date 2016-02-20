from time import time
import numpy as np
import cv2

def info(real_num,calc_num):
    
    real_num = float(real_num)
    calc_num = float(calc_num)
    d = abs(real_num-calc_num)

    result = (d/real_num)*100

    print("""Accuracy: {0}%""".format(round(100-result,2)))
    print("""Error: {0}%""".format(round(result,2)))

class DetectorError(Exception):
    def __init__(self,message='Something went wrong'):
        self.message = message

    def __str__(self):
        return self.message

    def __repr__(self):
        return str(self)

class Detector(object):

    def __init__(self,imgSource = 0,videoWidth = 320,videoHeight = 240,rate=0.01):
        
        self.cap = cv2.VideoCapture(imgSource)
        if type(imgSource)==int:
            self.cap.set(3,videoWidth)
            self.cap.set(4,videoHeight)

        self.counter = 0                              #object counter

        self._rect = []
        self._poly = []
        self.ALL_WINDOW = False
        self.P1 = None
        self.P2 = None
        self.rate = rate
        self.ctrs = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

        self.weight = None
        self.height = None
        self.minArea = None
        self.num_cnts = None
        self.frame_num = 0
        self.line_mode = None
        
    def clickEvent(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self._rect.append((y,x))                  #Numpy manages the coordinates as (y,x) instead of (x,y) like the rest of the world

    def clickEventPoly(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self._poly.append((x,y))

    def get_counter(self):
        return self.counter


    def boundObjects(frame,thresh):
        #global counter,width,halfH,halfW,prev_x,prev_y,minArea,numCnts
        #global p1_count_line,p2_count_line,font,ctrs,GUI
        cnts,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:numCnts]    

        index = 1
        current_cent = []       #a list of the centroids in the current frame
        for c in cnts:
            if cv2.contourArea(c) < self.minArea:
                continue

            rect = cv2.minAreaRect(c)
            points = cv2.cv.BoxPoints(rect)
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
                if not minDist>=float(w)/2:
                    prev_x,prev_y = minPoint
                else: prev_x,prev_y = cx,cy
            #ctrs = current_cent
                #Determines if the line has been crossed
            if self.line_mode.upper()=='H':
                if (prev_y <= self.P1[1] <= cy) or (cy <= self.P1[1] <= prev_y):
                    self.counter += 1
                    cv2.line(frame,self.P1,self.P2,(0,255,0),3)
                    #cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),3)
                    print (self.counter)
            elif self.line_mode.upper()=='V':
                if (prev_x <= self.P1[0] <= cx) or (cx <= self.P1[0] <= prev_x):
                    self.counter += 1
                    cv2.line(frame,self.P1,self.P2,(0,255,0),3)
                    #cv2.line(frame,p1_count_line,p2_count_line,(0,255,0),3)
                    print (self.counter)

            cv2.drawContours(frame,[points],0,(0,255,0),1)
            cv2.line(frame,(prev_x,prev_y),(cx,cy),(255,0,0),1)         #A line to show motion
            #cv2.circle(frame,(cx,cy),3,(0,0,255),2)
            cv2.putText(frame,'('+str(cx)+','+str(cy)+')',(cx,cy),font,0.3,(0,255,0),1)
            cv2.putText(frame,str(index),(cx,cy-15),font,0.4,(255,0,0),2)

            index += 1
            
        ctrs = current_cent


    def setup(self,line_m,fraction):
        
        grabbed,img = self.cap.read()
        if not grabbed:
            raise DetectorError('Could not get frame from video')
        x,y,w,h = None,None,None,None

        cv2.namedWindow('setup',1)
        k = None
        cv2.imshow('setup',img)
        while k != ord('q') and k != 27:
            cv2.setMouseCallback('setup',self.clickEvent)
            k = cv2.waitKey(0) & 0xFF
            if k == ord('\n'):
                self.ALL_WINDOW = True
                break
        cv2.destroyWindow('setup')

        if not self.ALL_WINDOW:
            roi = np.array([self._rect])
            box = cv2.boundingRect(roi)
            x,y,w,h = box
            roi_mask = img[x:x+w,y:y+h]
        else:
            roi_mask = img


        print('Now select the exact ROI you want')
        k=None
        cv2.namedWindow('setup2',1)
        cv2.imshow('setup2',roi_mask)

        while k!=ord('q') and k != 27:
            cv2.setMouseCallback('setup2',self.clickEventPoly)
            k = cv2.waitKey(0) & 0xFF
            if k ==ord('\n'):
                self._poly = []
                break
        cv2.destroyWindow('setup2')
        roi_poly = np.array([self._poly])

        black_mask = None
        print(self._poly)
        if len(self._poly)!= 0:
            black_mask = np.zeros(roi_mask.shape,dtype=np.uint8)
            cv2.fillPoly(black_mask,roi_poly,(255,255,255))

            average = np.float32(black_mask)
        else: average = np.float32(roi_mask)

        #-------------------------------------
        self.width = roi_mask.shape[1]
        self.height = roi_mask.shape[0]

        #Sets up the counting line
        if line_m.upper()=='H' or line_m is None:
            fract = int(self.height*fraction)
            self.P1 = (0,fract)
            self.P2 = (self.width,fract)
        elif line_m.upper()=='V':
            fract = int(self.width*fraction)
            self.P1 = (fract,0)
            self.P2 = (fract,self.height)
        else: raise ValueError('Expected an "h" or a "v" only')


        return (roi_mask,black_mask,average,(x,y,w,h))
        

    def trackGUI(self,line_mode='h',fract=0.5):
        self.line_mode = line_mode
        roi_mask,black_mask,avg,box = self.setup(line_mode,fract)
        x,y,w,h = box

        init_time = time()
        while True:
            grabbed,img = self.cap.read()
            if not grabbed:
                break
            #-----------------------
            if not self.ALL_WINDOW:
                roi_mask = img[x:x+w,y:y+h]
            else: roi_mask = img
            roi_mask = cv2.cvtColor(roi_mask,cv2.COLOR_BGR2GRAY)
            
            if not black_mask is None:
                window_mask = cv2.bitwise_and(roi_mask,black_mask)
            else:
                window_mask = roi_mask
            #======================
            if self.frame_num < 50:            #Hardcoded value indicating how many frames to let pass once the video begins
                self.frame_num += 1
                cv2.accumulateWeighted(window_mask,avg,self.rate)
                continue
            cv2.accumulateWeighted(window_mask,avgself.rate)
            result = cv2.convertScaleAbs(avg)       #the average background
            fmask = cv2.absdiff(result,window_mask)         #difference between the running average background and the current frame

            ##------Extra blur------##
            fmask = cv2.GaussianBlur(fmask,(21,21),0)
            fmask = cv2.blur(fmask,(28,28))
            #fmask = cv2.GaussianBlur(fmask,(21,21),0)
            #fmask = cv2.GaussianBlur(fmask,(21,21),0)
            #fmask = cv2.GaussianBlur(fmask,(21,21),0)
            
            
            _,thresh = cv2.threshold(fmask,30,255,0)

         ##-----Noise reduction-----##
            dimg = cv2.erode(thresh,None)
            dimg = cv2.erode(dimg,None)
            dimg = cv2.dilate(dimg,None)                  #Noise reduction function
            dimg = cv2.dilate(dimg,None)
            #dimg = cv2.dilate(dimg,None)
            cv2.imshow('dilate',dimg)
        #-------------------------------------------

            #Setting the boxes for the bounding process
            img2 = cv2.cvtColor(window_mask,cv2.COLOR_GRAY2BGR)
            self.boundObjects(img2,dimg)

        ##---------------Showing The Frames-----------------##
            cv2.imshow('roi',roi_mask)
            cv2.imshow('polygon',window_mask)
            cv2.imshow('average', result)
            cv2.line(img2,self.P1,self.P2,(0,0,255),1)    
            cv2.imshow('boxes',img2)
        ##-------------Termination Conditions-------------##
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or k == ord('q'):
                break
        print("""Time on main loop: {0} secs""".format(str(round(time.time()-init_time,2))))
        print('Vehicles Detected: '+str(counter))
        info(real_num,counter)
        #try:
        #    print('('+str(y)+' '+str(x)+' '+str(h)+' '+str(w)+')')
        #except:
        #    print('Whole frame was used')

        cap.release()
        cv2.destroyAllWindows()
