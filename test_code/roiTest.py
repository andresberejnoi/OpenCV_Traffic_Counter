import cv2
import numpy as np

# original image
cap = cv2.VideoCapture(0)
#image = cv2.imread('image.png')
_, image = cap.read()

print(image.shape)

# mask (of course replace corners with yours)
mask = np.zeros(image.shape, dtype=np.uint8)
#roi_corners = np.array([[(10,10), (300, 10) , (300,300), (10,300)]], dtype=np.int32)

roi_corners = np.array([[(int(cap.get(3)/2) - 20,  int(cap.get(4)/2)), (int(cap.get(3)/2) + 20,  int(cap.get(4)/2)), (0, int(cap.get(4))), 
(int(cap.get(3)), int(cap.get(4)))]])

white = (255, 255, 255)
cv2.fillPoly(mask, roi_corners, white)

while True:
	
	_, image = cap.read()

	# apply the mask
	masked_image = cv2.bitwise_and(image, mask)
	
	#_, thresh = cv2.threshold(masked_image, 127, 255, 0)
	thresh = cv2.Canny(masked_image, 127, 255)
	
	cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# display your handywork
	cv2.imshow('masked image', masked_image)
	cv2.imshow('thresh after', thresh)
	
	k = cv2.waitKey(10) & 0xFF
	if k == 27 or k ==ord('q'):
		break
	
cv2.waitKey()
cv2.destroyAllWindows()
