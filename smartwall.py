import numpy as np
import cv2

from calibrate import calibrateCamera
	
#Projector dimensions
projH = 900
projW = 640

cap = cv2.VideoCapture(0)

transform = calibrateCamera((projH, projW), cap)

display((projH, projW), cap, transform)