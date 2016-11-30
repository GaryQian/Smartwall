import numpy as np
import cv2
import time
from nonmaxsuppts import nonmaxsuppts

def calibrateCamera(projDim, cap):
	points = [(projDim[0] * .5, projDim[1] * .5),
	(projDim[0] * .1, projDim[1] * .1),
	(projDim[0] * .1, projDim[1] * .9),
	(projDim[0] * .9, projDim[1] * .1),
	(projDim[0] * .9, projDim[1] * .9),
	(projDim[0] * .5, projDim[1] * .75),
	(projDim[0] * .5, projDim[1] * .25),
	(projDim[0] * .75, projDim[1] * .5),
	(projDim[0] * .25, projDim[1] * .5),
	(projDim[0] * .3, projDim[1] * .3),
	(projDim[0] * .7, projDim[1] * .3),
	(projDim[0] * .3, projDim[1] * .7),
	(projDim[0] * .7, projDim[1] * .7)]
	out = np.ndarray((projDim[1], projDim[1], 3), np.float32)
	out[:,:,:] = 0
	for point in points:
		cv2.circle(out, (int(point[1]), int(point[0])), projDim[1] / 50, (0, 255, 0), thickness=-1)
	# Display the resulting frame
	cv2.imshow('Calibrating...',out)
	print('Set up projector.')
	#Allow image to display. Useless while loop otherwise.
	startTime = time.time()
	while(True):
		ret, frame = cap.read()
		if cv2.waitKey(1) < 0:
			break
	wait = raw_input('Press enter/return to continue...')
	print('Calibrating camera...')
	time.sleep(2)
	
	transform = None
	kernel = np.ones((4,4),np.uint8)
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		
		#convert to HSV color space to detect hue
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		#lowerColor = np.array([30,60,60])
		#upperColor = np.array([110,255,255])
		lowerColor = np.array([30,150,50])
		upperColor = np.array([80,255,255])

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lowerColor, upperColor)
		
		## Bitwise-AND mask and original image
		#res = cv2.bitwise_and(frame,frame, mask= mask)

		#erode to remove noise
		mask = cv2.erode(mask, kernel,iterations = 1)
		
		detectedPoints = nonmaxsuppts(mask, 15, 100)
		out = np.ndarray(frame.shape, np.float32)
		for pt in detectedPoints:
			out[pt[0]][pt[1]] = [255, 255, 255]
		#cv2.FindHomography(srcPoints, dstPoints, H, method=0, ransacReprojThreshold=3.0, status=None) 
		
		# Display the resulting frame
		cv2.imshow('Calibrating...',out)
		#cv2.imshow('Calibrating...',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	
	return transform