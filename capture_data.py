import numpy as np
import cv2

#Projector dimensions
projW = 1920
projH = 1080

cap = cv2.VideoCapture(0)

while True:
	ret_val, img = cap.read()
	cv2.imshow('my webcam', img)
	if cv2.waitKey(1) == 27:
		break

cv2.destroyAllWindows()
		
display((projH, projW), cap)