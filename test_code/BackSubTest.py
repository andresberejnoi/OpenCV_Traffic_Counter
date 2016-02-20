

def main():
    import cv2

    #cap = cv2.VideoCapture('trafficDemoHD.avi')                      #The zero indicates that the video source will be taken from the built-in webcam. You can also use a file name or file path to open an already recorded video
    cap = cv2.VideoCapture('Video11.avi')                     
 #   cap = cv2.VideoCapture(0)                      #The zero indicates that the video source will be taken from the built-in webcam. You can also use a file name or file path to open an already recorded video

    fbg = cv2.BackgroundSubtractorMOG()                         # There are three algorithms available: BackgroundSubtractorMOG, BackgroundSubtractorMOG2, BackgroundSubtractorGMG

    while True:                                   #Checks if the camera is in use
        ret,img = cap.read()
        if not ret:
            break
        fgmask = fbg.apply(img)                                             #This applies the background subtraction to img and returns a new image data
        
        cv2.imshow('Image', fgmask)
        k = cv2.waitKey(30) & 0xff
        if k == 27 or k == ord('q'):
            print(k)
            break
        elif k == -1:                                           #According to documentation, when no key is pressed, waitKey returns -1. However, on this system (Ubuntu Gnome 15.04) it is returning 255 as default
            continue
        else: print(k)

    cap.release()
    cv2.destroyAllWindows()                         #Should destroy or close all windows created by the program. However, that does not work
    

def main2():
    import cv2
     
    cam = cv2.VideoCapture(0)
     
    ret, bg = cam.read()
     
    while True:
        frame = cam.read()[1]
        frame = cv2.blur(frame, (5,5))
        mask = cv2.absdiff(frame, bg)
     
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
     
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
     
        gray = cv2.blur(gray, (9,9))
        #cv2.imshow("mask", mask)
        cv2.imshow("gray", gray)
        cv2.imshow("webcam", frame)
        cv2.imshow('mask', mask)
        
        k = cv2.waitKey(2) & 0xFF
        if k == 27 or k == ord('q'):
            break
     
    cam.release()
    cv2.destroyAllWindows()
main()
