import cv2

cap = cv2.VideoCapture(0)

i = 0
while cap.isOpened():
    res, img = cap.read()
    cv2.imshow('Test Again', img)
    
    k = cv2.waitKey(1)                                              #if the argument for waitKey() is zero, then the video is frozen until a key is pressed, and on every key press, only one frame is moved (this is on Ubuntu Gnome 15.04)
    if k == 1048603:                        #Experimental value obtained by printing k. Other tutorials put 27 which is the ASCII value for the escape key, but at least in this system, it does not work.
#        cap.release()
#        cv2.destroyAllWindows()
        break
    elif k == -1:
        continue
    else: print(k)
    
    i += 1
    
    if i > 100000:
        break
cap.release()
cv2.destroyAllWindows()
