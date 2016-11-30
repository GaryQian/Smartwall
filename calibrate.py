import numpy as np
import cv2
import time

def calibrateCamera(projDim, cap):
	out = np.ndarray((projDim[0], projDim[1], 3), np.float32)
	out[:,:,:] = 1
	
	cv2.circle(out, (projDim[1] / 2, projDim[0] / 2), projDim[1] / 100, (0, 255, 0), thickness=-1)
	# Display the resulting frame
	cv2.imshow('Calibrating...',out)
	print('Set up projector.')
	#Allow image to display. Useless while loop otherwise.
	while(True):
		ret, frame = cap.read()
		if cv2.waitKey(1) < 0:
			break
	wait = raw_input('Press any key to continue...')
	print('Calibrating camera...')
	time.sleep(1)
	
	transform = None
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		
		#convert to HSV color space to detect hue
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		lowerGreen = np.array([45,60,60])
		upperGreen = np.array([90,255,255])

		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lowerGreen, upperGreen)
		
		## Bitwise-AND mask and original image
		#res = cv2.bitwise_and(frame,frame, mask= mask)

		
		#cv.FindHomography(srcPoints, dstPoints, H, method=0, ransacReprojThreshold=3.0, status=None) 
		
		# Display the resulting frame
		cv2.imshow('Calibrating...',mask)
		#cv2.imshow('Calibrating...',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	
	return transform