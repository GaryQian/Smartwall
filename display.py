import numpy as np
import cv2
import time
import imutils
import sys
import win32api

def display(projDim, cap, transform, dp):
	
	
	erode = np.ones((3,3),np.uint8)
	blurRad = 11
	boxblur = np.ones((blurRad,blurRad),np.float32) / (blurRad * blurRad)
	out = erase(projDim)
	attempt = 0
	while(True):
		attempt += 1
	
		# Capture frame-by-frame
		ret, frame = cap.read()
		fh, fw, _ = frame.shape
		
		# Our operations on the frame come here
		
		
		
		# Our operations on the frame come here
		frameBlur = cv2.filter2D(frame, -1, boxblur)
		
		#convert to HSV color space to detect hue
		hsv = cv2.cvtColor(frameBlur, cv2.COLOR_BGR2HSV)
		# define range of blue color in HSV
		#lowerColor = np.array([30,60,60])
		#upperColor = np.array([110,255,255])
		lowerColor = np.array([40,50,70])
		upperColor = np.array([90,255,255])
		
		# Threshold the HSV image to get only blue colors
		mask = cv2.inRange(hsv, lowerColor, upperColor)
		
		## Bitwise-AND mask and original image
		#res = cv2.bitwise_and(frame,frame, mask= mask)

		#erode to remove noise
		mask = cv2.erode(mask, erode,iterations = 1)
		
		detectedPoints = []
		
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = cnts[0] if imutils.is_cv2() else cnts[1]
		for c in cnts:
			# compute the center of the contour
			M = cv2.moments(c)
			if (M["m00"] != 0 and attempt > 10):
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			else:
				continue
			cont = True
			#check if the point is not right next to another
			for pt in detectedPoints:
				if abs(pt[0] - cY) + abs(pt[1] - cX) < 30:
					cont = False
					break
			if cont:
				detectedPoints.append((cY,cX))
				cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
				cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
				cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
			# draw the contour and center of the shape on the image
			
			#
		
		print len(detectedPoints)
		
		
		
		for p in detectedPoints:
			
			p = np.dot(transform[0], [int(p[0]), int(p[1]), 1])
			p = p / p[2]
			cv2.circle(out, (int(p[1]), int(p[0])), projDim[1] / 120, (0, 0, 255), thickness=-1)
			if (len(detectedPoints) == 1):
				win32api.SetCursorPos((int(p[1]), int(p[0])))
		out[0:fh/2,0:fw/2,] = frame[::2,::2]
		# Display the resulting frame
		cv2.imshow('Smartwall',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def erase(projDim):
	out = np.ndarray((projDim[0], projDim[1], 3), np.uint8)
	out[:,:,:] = 0
	return out
	

	
