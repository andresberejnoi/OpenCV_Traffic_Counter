import cv2


cap = cv2.VideoCapture(0)
cap.set(3,320)
cap.set(4,240)

#To find contours, use: cv2.findContours(source, [cont1,cont2,...],
#To draw the contours, use: cv2.drawContours(...)
i = 0

while cap.isOpened():
    
    img = cap.read()[1]
    #cv2.imshow('Normal',img)
    
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #imgray = cv2.bilateralFilter(imgray,11,17,17)
    #cv2.imshow('Gray Scale',imgray)

    ret,thresh = cv2.threshold(imgray,127,255,0)
    cv2.imshow('Threshold Mask',thresh)

    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Thresh After',thresh)

    contours = sorted(contours, key=cv2.contourArea,reverse=True)[:10]

    #Drawing contours:
    img2 = img.copy()           #a deep copy of the original image frame
    cv2.drawContours(img2, contours,-1,(0,255,0),1)
    cv2.imshow("Detect Window",img2)


  #Creating bounding boxes:
    #Text to write with the boxes:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = 'moving'
    
    for c in contours:
        x,y,x2,y2 = cv2.boundingRect(c)         #Returns a tuple of four coordinates corresponding to two points

        cv2.rectangle(img,(x,y),(x+x2,y+y2), (0,255,0),1)
        cv2.putText(img,text,(x2,y2),font,1,(0,255,0))

    cv2.imshow('Rectangles',img)        

#    if i%2000 == 0:
#        print ('len(contours): ', str(len(contours)))
#        print type(detect)
        

    k = cv2.waitKey(10) & 0xFF

    if k == 27 or k == ord('q'):
        cap.release()

cv2.destroyAllWindows()
