import cv2

cap = cv2.VideoCapture(0)

while True:
    oldFrame = cap.read()[1]
    oldGray = cv2.cvtColor(oldFrame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(oldGray,  mask = None)
