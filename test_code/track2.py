import cv2
import numpy as np

def drawMidLine(dir = 'H'):
    '''drawMidLine(dir = 'H') -> None (inefficient; Don't use; Not even finished)
        dir: a character (H or V) indicating whether
        the line is vertical or horizontal.
        It draws a line in the middle of the frame.'''
    if dir.upper() == 'H':
        half = int(cap.get(4)/2)
    elif dir.upper() == 'V':
        half = int(cap.get(3)/2)
    else:
        raise ValueError('dir can only be "H" or "V"')

    #The code is not completed

def getCentroid(a,b,c,d):
    '''Calculates the centroid relative to b (doesn't work)'''
    hz = cv2.magnitude(b,c)         #the horizontal distance
    vt = cv2.magnitude(b,a)         #the vertical distance
    C = (int(b[0]+hz),int(b[1]+vt))

    return C

    

    

cap = cv2.VideoCapture(0)
#Reducing video size to improve framerate. Final video size: 320x240
cap.set(3,320)                  #Setting the width of the video. The 3 represents the width
cap.set(4,240)                  #Setting the height of the video. The 4 represents the height

#setting global variables
halfH = int(cap.get(4)/2)           #half of the height 
halfW = int(cap.get(3)/2)           #half of the width
width = int(cap.get(3))

bmask = cv2.BackgroundSubtractorMOG()
img = cap.read()[1]




i = 0
counter = 0
while cap.isOpened():           #Checks if the camera is still connected
    img = cap.read()[1]
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    fmask = bmask.apply(imgray)
    

    _,thresh = cv2.threshold(fmask,127,255,0)
    cv2.imshow('thresh',thresh)

    cnts,hier = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]
    
    img2 = img.copy()
    img3 = img.copy()
    #cv2.line(img, (0,halfH),(width,halfH),(0,0,255),1)

    #Initializing the previous coordinates
    prev_x = 0
    prev_y = 0
    for c in cnts:
        if cv2.contourArea(c) < 50:
            continue
        #Creating minimum area rectangles over contours
        rect = cv2.minAreaRect(c)
        points = cv2.cv.BoxPoints(rect)
        points = np.int0(points)
        cv2.drawContours(img2,[points],0,(0,255,0),1)

        cx = int(rect[0][0])
        cy = int(rect[0][1])

        cv2.circle(img2,(cx,cy),4,(0,0,255),3)          #Draws the centroid for each rectangle

        

        if prev_y < halfH < cy:
            counter += 1
            cv2.line(img2, (0,halfH),(width,halfH),(0,255,0),3)
            #cv2.putText(img2,str(counter),
        
        prev_x = cx
        prev_y = cy

 #       print('area: ', str(cv2.contourArea(points)))
        
#        M = cv2.moments(points)
#        a,b,c,d = points        #unpacks the box points
#        print(a,b,c,d)
#        d = cv2.magnitude(b,c)
        #centroid = getCentroid(a,b,c,d)
        #print centroid

        #cv2.circle(img2,centroid, 4,(0,0,255),1)
        

        
#        for i in range(len(points)):
#            if i == 1:
#                cv2.circle(img2,tuple(points[i]),3,(0,0,255),1)
        #print(points)
        #print('')

#        #Creating circles around the objects:
#        (x,y),radius = cv2.minEnclosingCircle(c)
#        C = (int(x),int(y))         #Center of the circle. It needs integers
#        radius = int(radius)        #radius. It also needs to be an integer
#        cv2.circle(img,C,radius,(255,0,0),1)

#        #creating ellipses
#        ellipse = cv2.fitEllipse(c)
#        cv2.ellipse(img3,ellipse,(0,255,0),1)
#        #print(ellipse[0],ellipse[1])
        

#        M = cv2.moments(c)

#        if i %5000 == 0:
#            print('len(M) = ' + str(len(M)))
#            print('area = ' + str(M['m00']))
#            print('')
        #x,y,w,h = cv2.boundingRect(c)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
#    cv2.imshow('Circles',img)
    cv2.line(img2, (0,halfH),(width,halfH),(0,0,255),1)
#    cv2.line(img3, (0,halfH),(width,halfH),(0,0,255),1)
    cv2.imshow('Min',img2)
#    cv2.imshow('ellipse',img3)

    k = cv2.waitKey(10) & 0xFF
    if k == 27 or k == ord('q'):
        cap.release()

    i += 1

cv2.destroyAllWindows()
