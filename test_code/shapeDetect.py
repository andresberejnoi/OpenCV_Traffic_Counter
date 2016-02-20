import cv2


def detect(image = 'shapes.jpg'):
    '''image: an image name or path in the system'''
    img = cv2.imread(image)

    imgray = cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY)
    
    while True:
        
        thresh = cv2.Canny(imgray, 100, 200)
        cv2.imshow('Thresh', thresh)
        
        contours,  hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        #Drawing the contours:
        cv2.drawContours(img, contours, -1, (0, 0, 255), 10 )
        
        cv2.imshow('Shapes', img)
        
        k = cv2.waitKey(0) & 0xFF
        if k == 27 or k == ord('q'):
            break
    cv2.destroyAllWindows()

detect()



