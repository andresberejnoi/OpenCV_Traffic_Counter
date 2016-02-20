import cv2

#cap = cv2.VideoCapture(0)

def imageDiff(a, b, c):
    '''Substracts one image from the next and returns the difference.
    The difference can be used as a mask image that only shows the motion.'''
    t0 = cv2.absdiff(a, b)
    t1 = cv2.absdiff(b, c)
        
    t2 = cv2.bitwise_and(t0, t1)
    return t2

cap = cv2.VideoCapture(0)                                   #0 is the default port for the  built-in camera in laptops
t = cap.read()[1]
tp = cap.read()[1]
tpp = cap.read()[1]

t = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
tp = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
tpp = cv2.cvtColor(tpp, cv2.COLOR_BGR2GRAY)

while True:
    img = imageDiff(t, tp, tpp)
    cv2.imshow('Motion Detection', img)
    
    res, img = cap.read()
    t = tp
    tp = tpp
    tpp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    k = cv2.waitKey(10) & 0xFF
    if k==27 or k == ord('q'):
        cv2.destroyAllWindows()
        break
        
cap.release()
cv2.destroyAllWindows()
