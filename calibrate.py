import numpy as np
import cv2
import time
import imutils
import sys

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
		cv2.circle(out, (int(point[1]), int(point[0])), projDim[1] / 30, (0, 255, 0), thickness=-1)
	# Display the resulting frame
	#print('Set up projector. Press any key to continue.')
	cv2.imshow('Calibrating...',out)
	cv2.waitKey(100)
	wait = raw_input('Press enter/return to continue...')
	print('Calibrating camera. Move calibration image to projector.')
	time.sleep(1)
	
	transform = None
	kernel = np.ones((4,4),np.uint8)
	blurRad = 11
	boxblur = np.ones((blurRad,blurRad),np.float32) / (blurRad * blurRad)
	attempts = 0
	while(True):
		attempts += 1
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		frame = cv2.filter2D(frame, -1, boxblur)
		
		#convert to HSV color space to detect hue
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		#lowerColor = np.array([30,60,60])
		#upperColor = np.array([110,255,255])
		lowerColor = np.array([40,50,50])
		upperColor = np.array([90,255,255])
		
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
			if (M["m00"] != 0):
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			else:
				continue
			cont = True
			#check if the point is not right next to another
			for pt in detectedPoints:
				if abs(pt[0] - cY) + abs(pt[1] - cX) < 40:
					cont = False
					break
			if cont:
				detectedPoints.append((cY,cX))
				cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
				cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
				cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			# draw the contour and center of the shape on the image
			
			#
		
		print len(detectedPoints)
		
		# Display the resulting frame
		cv2.imshow('Calibrating...',frame)
		#cv2.imshow('Calibrating...',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		if (len(detectedPoints) > len(points) * 0.5 and len(detectedPoints) > 4):
			transform = cv2.findHomography(np.array(detectedPoints), np.array(points), method=cv2.RANSAC, ransacReprojThreshold=3.0) 
			break
		elif (attempts > 150):
			print('Failed to calibrate. Please relaunch and try again.')
			cap.release()
			cv2.destroyAllWindows()
			sys.exit()
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	print('Found transform successfully!')
	return transform
	
'''def collapse(img):
	h, w = img.shape
	points = []
	for i in range(h):
		for j in range(w):
			if (img[i][j] > 100):
	'''			
				