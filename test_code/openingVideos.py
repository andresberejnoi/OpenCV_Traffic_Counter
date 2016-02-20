import cv2


fbg = cv2.BackgroundSubtractorMOG()                                 #A backgound subtraction object

cap = cv2.VideoCapture(0)
while cap.isOpened():
    img = cap.read()[1]                                                           # img is a numpy array that contains information of the image
    
    
