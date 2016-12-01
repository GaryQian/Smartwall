import numpy as np
import cv2

from calibrate import calibrateCamera
	
#Projector dimensions
projW = 1920
projH = 1080


cap = cv2.VideoCapture(1)

transform, detectedPoints = calibrateCamera((projH, projW), cap)

display((projH, projW), cap, transform)