import numpy as np
import cv2
import imutils
import os
import time 


class TrafficCounter(object):
    def __init__(self,video_source=0,
                 line_direction='H',
                 line_position=0.5,
                 video_width = 640,
                 min_area = 1000,
                 video_out=False,
                 numCnts=10):
        self.crop_rect         = []         #stores the click coordinates where to crop the frame
        self.mask_points       = []         #stores the click coordinates of the mask to apply to cropped frame
        self.font              = cv2.FONT_HERSHEY_SIMPLEX
        self.p1_count_line     = None 
        self.p2_count_line     = None
        self.counter           = 0
        self.line_direction    = line_direction       
        self.line_position     = line_position       
        self.minArea           = min_area        
        self.numCnts           = numCnts
        self.video_source      = cv2.VideoCapture(video_source)    
        self.screenshot_folder = '_screenshots'
        self.video_out_folder  = '_videos'

        self._vid_width        = video_width       
        self._vid_height       = None        #PLACEHOLDER
        self.black_mask        = None        #PLACEHOLDER, user creates it by clicking on several points
        self.video_out         = video_out       
        if video_out:
            self._set_video_writers()

        self._compute_frame_dimensions()
        self._set_up_line(line_direction,line_position)

    def _set_video_writers(self):
        pass
        
    def _set_up_line(self,line_direction,line_position):
        if line_direction.upper()=='H' or line_direction is None:
            fract = int(self._vid_height*float(line_position))
            self.p1_count_line = (0,fract)
            self.p2_count_line = (self._vid_width,fract)
        elif line_direction.upper() == 'V':
            fract = int(self._vid_width*float(line_position))
            self.p1_count_line = (fract,0)
            self.p2_count_line = (fract,self._vid_height)
        else:
            raise ValueError('Expected an "H" or a "V" only for line direction')
    
    def _compute_frame_dimensions(self):
        grabbed,img = self.video_source.read()
        while not grabbed:
            grabbed,img = self.video_source.read()
        img = imutils.resize(img,width=self._vid_width)
        self._vid_height = img.shape[0]
        self._vid_width  = img.shape[1]

    def _click_crop_event(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.crop_rect.append((y,x))                  #Numpy manages the coordinates as (y,x) instead of (x,y)

    def _click_mask_event(self,event,x,y,flags,userdata):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.mask_points.append((x,y))
    
    def _draw_bounding_boxes(self,frame,contour_id,bounding_points,cx,cy,prev_cx,prev_cy):
        cv2.drawContours(frame,[bounding_points],0,(0,255,0),1)
        cv2.line(frame,(prev_cx,prev_cy),(cx,cy),(0,0,255),1)         #count line when it is inactive
        cv2.circle(frame,(cx,cy),3,(0,0,255),4)
        cv2.putText(frame,str(contour_id),(cx,cy-15),self.font,0.4,(255,0,0),2)

    def _is_line_crossed(self,frame,cx,cy,prev_cx,prev_cy):
        is_crossed = False
        if self.line_direction.upper() == 'H':
            if (prev_cy <= self.p1_count_line[1] <= cy) or (cy <= self.p1_count_line[1] <= prev_cy):
                self.counter += 1
                cv2.line(frame,self.p1_count_line,self.p2_count_line,(0,255,0),5)
                is_crossed = True

        elif self.line_direction.upper() == 'V':
            if (prev_cx <= self.p1_count_line[0] <= cx) or (cx <= self.p1_count_line[0] <= prev_cx):
                self.counter += 1
                cv2.line(frame,self.p1_count_line,self.p2_count_line,(0,255,0),5)
                is_crossed = True
        return is_crossed

    def bind_objects(self,frame,thresh_img):
        '''Draws bounding boxes and detects when cars are crossing the line
        frame: numpy image where boxes will be drawn onto
        thresh_img: numpy image after subtracting the background and all thresholds and noise reduction operations are applied
        '''
        cnts,_ = cv2.findContours(thresh_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)                #this line is for opencv 2.4, and also now for OpenCV 4.4, so this is the current one
        cnts = sorted(cnts,key = cv2.contourArea,reverse=True)[:self.numCnts]

        cnt_id          = 1
        cur_centroids  = []
        prev_centroids = []
        for c in cnts:
            if cv2.contourArea(c) < self.minArea:
                continue
            rect = cv2.minAreaRect(c)
            points = cv2.boxPoints(rect)                # This is the way to do it in opencv 3.1
            points = np.int0(points)

            #Getting the center coordinates of the contour box
            cx = int(rect[0][0])
            cy = int(rect[0][1])

            w,h = rect[1]                   #Unpacks the width and height of the frame

            C = np.array((cx,cy))
            cur_centroids.append((cx,cy))

            #Finding the centroid of c in the previous frame
            if len(prev_centroids)==0: prev_cx,prev_cy = cx,cy
            elif len(cnts)==0: prev_cx,prev_cy = cx,cy
            else:
                minPoint = None
                minDist = None
                for i in range(len(prev_centroids)):
                    dist = np.linalg.norm(C - prev_centroids[i])                #numpy's way to find the euclidean distance between two points
                    if (minDist is None) or (dist < minDist):
                        minDist = dist
                        minPoint = prev_centroids[i]
                #This if is meant to reduce overcounting errors
                if not minDist >= w/2:
                    prev_cx,prev_cy = minPoint
                else: prev_cx,prev_cy = cx,cy
            
            _is_crossed = self._is_line_crossed(frame,cx,cy,prev_cx,prev_cy)
            if _is_crossed:
                print(f"Total Count: {self.counter}")
            self._draw_bounding_boxes(frame,cnt_id,points,cx,cy,prev_cx,prev_cy)

            cnt_id += 1
        prev_centroids = cur_centroids       #updating centroids for next frame

    def _set_up_masks(self):
        grabbed,img = self.video_source.read()
        while not grabbed:
            grabbed,img = self.video_source.read()

        img = imutils.resize(img,width=self._vid_width)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self._vid_height = img.shape[0]

        ##-------Show setup window
        k = None
        cv2.namedWindow('setup2',1)
        cv2.imshow('setup2',img)
        while k != ord('q') and k != ord('Q') and k != 27 and k != ('\n'):
            cv2.setMouseCallback('setup2',self._click_mask_event)
            k = cv2.waitKey(0) & 0xFF

        cv2.destroyWindow('setup2')

        roi_points = np.array([self.mask_points])
        self.black_mask = None
        if len(self.mask_points)!= 0:
            self.black_mask = np.zeros(img.shape,dtype=np.uint8)
            cv2.fillPoly(self.black_mask,roi_points,(255,255,255))

            self.raw_avg = np.float32(self.black_mask)
        else: self.raw_avg = np.float32(img)

        self.raw_avg = cv2.resize(self.raw_avg, (self._vid_width,self._vid_height))


    def main_loop(self):
        self._set_up_masks()
        rate_of_influence = 0.01
        frame_num = 0
        FRAME_CROPPED = False
        while True:
            grabbed,img = self.video_source.read()
            if not grabbed:
                break
            #--------------
            img = cv2.resize(img,(self._vid_width,self._vid_height))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if FRAME_CROPPED:
                working_img = img[x:x+w,y:y+h]
            else:
                working_img = img.copy()
            if self.black_mask is not None:
                working_img = cv2.bitwise_and(working_img,self.black_mask)

            if frame_num < 1:           #Hardcoded value indicating how many frames to let pass once the video begins
                frame_num += 1
                cv2.accumulateWeighted(working_img,self.raw_avg,rate_of_influence)
                continue
            
            cv2.accumulateWeighted(working_img,self.raw_avg,rate_of_influence)
            background_avg = cv2.convertScaleAbs(self.raw_avg)           #reference background average image
            subtracted_img = cv2.absdiff(background_avg,working_img)

            ##-------Adding extra blur------
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)
            subtracted_img = cv2.GaussianBlur(subtracted_img,(21,21),0)

            ##-------Applying threshold
            _,threshold_img  = cv2.threshold(subtracted_img,30,255,0)

            ##-------Noise Reduction
            dilated_img      = cv2.dilate(threshold_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)
            dilated_img      = cv2.dilate(dilated_img,None)

            ##-------Drawing bounding boxes and counting
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)          #Giving frame 3 channels for color (for drawing colored boxes)
            self.bind_objects(img,dilated_img)

            ##-------Showing Frames
            cv2.imshow('Masked Frame',working_img)              #working_img is the frame after being cropped and masked
            cv2.imshow('Background-Subtracted',subtracted_img)  #subtracted_img is the frame after the background has been subtracted from it
            cv2.imshow('Threshold Applied',dilated_img)         #dilated_img is threshold_img plus the noise reduction functions
            cv2.imshow('Running Avg of Background',background_avg)
            cv2.imshow('Motion Detection',img)

            ##-------Termination Conditions
            k = cv2.waitKey(25) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                break
            elif k == ord('s') or k == ord('S'):                #if the letter s/S is pressed, a screenshot of the current frame on each window will be saved to the current folder
                frame_id = int(self.video_source.get(1))        #get current frame index
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_masked_frame.jpeg"),working_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_background_subtracted.jpeg"),subtracted_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_threshold_applied.jpeg"),dilated_img)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_background_average.jpeg"),background_avg)
                cv2.imwrite(os.path.join(self.screenshot_folder,f"{frame_id}_car_counting.jpeg"),img)

        self.video_source.release()
        cv2.destroyAllWindows()

    def start_counting(self):
        
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
