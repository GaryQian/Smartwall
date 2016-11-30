import numpy as np
import cv2
import time
import imutils
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
	#print('Set up projector. Press any key to continue.')
	cv2.imshow('Calibrating...',out)
	cv2.waitKey(1)
	wait = raw_input('Press enter/return to continue...')
	print('Calibrating camera...')
	time.sleep(0.5)
	
	transform = None
	kernel = np.ones((4,4),np.uint8)
	blurRad = 11
	boxblur = np.ones((blurRad,blurRad),np.float32) / (blurRad * blurRad)
	#boxblur2 = np.ones((blurRad * 2,blurRad * 2),np.float32) / (blurRad * blurRad)
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		frame = cv2.filter2D(frame, -1, boxblur)
		
		#convert to HSV color space to detect hue
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		#lowerColor = np.array([30,60,60])
		#upperColor = np.array([110,255,255])
		lowerColor = np.array([40,150,50])
		upperColor = np.array([95,255,255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lowerColor, upperColor)
		
		## Bitwise-AND mask and original image
		#res = cv2.bitwise_and(frame,frame, mask= mask)

		#erode to remove noise
		#mask = cv2.erode(mask, kernel,iterations = 1)
		
		detectedPoints = []
		
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		for c in cnts:
			# compute the center of the contour
			M = cv2.moments(c)
			if (M["m00"] == 0):
				cX = 0
				cY = 0
			else:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
		 
			# draw the contour and center of the shape on the image
			cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
			cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
			#cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
		
		
		print len(detectedPoints)
		#out = np.ndarray(frame.shape, np.float32)
		#for pt in detectedPoints:
		#	out[pt[0]][pt[1]] = [255, 255, 255]
		#cv2.FindHomography(srcPoints, dstPoints, H, method=0, ransacReprojThreshold=3.0, status=None) 
		
		# Display the resulting frame
		cv2.imshow('Calibrating...',frame)
		#cv2.imshow('Calibrating...',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	
	return transform
	
'''def collapse(img):
	h, w = img.shape
	points = []
	for i in range(h):
		for j in range(w):
			if (img[i][j] > 100):
	'''			
				