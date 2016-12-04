import numpy as np
import cv2
import win32api

from calibrate import calibrateCamera
from display import display

#Projector dimensions
projW = 1920
projH = 1080


cap = cv2.VideoCapture(0)

transform, detectedPoints = calibrateCamera((projH, projW), cap)

display((projH, projW), cap, transform, detectedPoints) 