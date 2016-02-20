import cv2
def diff(a, b, c):
    t = cv2.absdiff(a, b)
    tp = cv2.absdiff(b, c)
    tpp = cv2.bitwise_and(t, tp)
    return tpp

path = 'dogs.avi'
cap = cv2.VideoCapture(path)
#cap.set(3, 720)
#cap.set(4, 480)

t = cap.read()[1]
tp = cap.read()[1]
tpp = cap.read()[1]

if cap.isOpened():

    while True:
        img = diff(t, tp, tpp)
        cv2.imshow('Motion', img)
        
        img = cap.read()[1]
        
        t = tp
        tp = tpp
        tpp = img
        
        
        k = cv2.waitKey(10) & 0xff
        
        if k == 27 or k == ord('q'):
            break
else:
    print('Video not loaded correctly')
    exit()
cap.release()
cv2.destroyAllWindows()
