import cv2

cap = cv2.VideoCapture(0)

writer = cv2.VideoWriter('VideoTest.avi',cap.get(6),20,(int(cap.get(3)),int(cap.get(4))))


while 1:
    _,img = cap.read()

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    

    cv2.imshow('camera',img)
    writer.write(img)

    k = cv2.waitKey(10) & 0xff
    if k == ord('q') or k == 27:
        break
cap.release()
exit()
