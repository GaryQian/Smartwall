import numpy as np
import cv2
import time
import imutils
import sys
import win32api
import win32con
import cPickle as pickle
from collections import deque
from keras.models import load_model

from keras.models import Sequential

def display(projDim, cap, transform, dp):
	training = False
	manualOffset = (-6, 0)
	counter = pickle.load(open("counter.dat", "rb"))
	
	erode = np.ones((3,3),np.uint8)
	blurRad = 9
	boxblur = np.ones((blurRad,blurRad),np.float32) / (blurRad * blurRad)
	out = erase(projDim)
	attempt = 0
	x = 0
	
	###################
	#Deep learing vars#
	###################
	model = load_model('model3deep.dat')
	pastGestures = deque([0, 0, 0, 0, 0])
	currentGesture = 0
	###################
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
		# define range of green color in HSV
		lowerColor = np.array([40,60,70])
		upperColor = np.array([85,255,255])
		
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
				cX = int(M["m10"] / M["m00"]) - manualOffset[1]
				cY = int(M["m01"] / M["m00"]) - manualOffset[0]
			else:
				continue
			cont = True
			#check if the point is not right next to another
			for pt in detectedPoints:
				if abs(pt[0] - cY) + abs(pt[1] - cX) < 30:
					cont = False
					break
			if cont:
				p = np.dot(transform[0], [cY, cX, 1])
				p = (p / p[2])
				if (p[0] >= 0 and p[1] >= 0 and p[0] <= projDim[0] and p[1] <= projDim[1]):
					detectedPoints.append((cY,cX))
					#cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)
					cv2.circle(frame, (cX, cY), 7, (0, 0, 255), -1)
					#cv2.putText(frame, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		
		for p in detectedPoints:
			
			trans = np.dot(transform[0], [int(p[0]), int(p[1]), 1])
			trans = (trans / trans[2])
			cv2.circle(out, (int(trans[1]), int(trans[0])), projDim[1] / 120, (0, 0, 255), thickness=-1)
			if (len(detectedPoints) == 1):
				win32api.SetCursorPos((int(trans[1]), int(trans[0])))
				window = obtainWindow(frame, p, trans, projDim)
				if window != None:
					cv2.imshow('Smartwall',window)
				if (not training and window != None):
					#Use deep learning predict
					imgs = np.ndarray((1, 32, 32, 3))
					imgs[0] = window
					prob = model.predict_proba(imgs, batch_size=10, verbose=0)
					prediction = 0
					if (prob[0][1] > prob[0][0]):
						prediction = 1
					print prediction
					pastGestures.append(prediction)
					pastGestures.popleft()
					temp = currentGesture
					if (pastGestures.count(0) >= 3):
						if (currentGesture == 1):
							currentGesture = 0
							win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)
						else:
							#Nothing
							x = x
					else:
						if (currentGesture == 1):
							#Nothing
							x = x
						else:
							currentGesture = 1
							win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
				elif (window != None):
					#Write training data
					if (attempt % 5 == 0):
						cv2.imwrite('trainingData2/0/img' + str(counter) + '.png', window)
						print counter
						counter += 1
						pickle.dump(counter, open( "counter.dat", "wb" ))
			else:
				cv2.imshow('Smartwall',frame)
		out[0:fh/2,0:fw/2,] = frame[::2,::2]
		
		# Display the resulting frame
		#print frame.shape
		#cv2.imshow('Smartwall',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()

def erase(projDim):
	out = np.ndarray((projDim[0], projDim[1], 3), np.uint8)
	out[:,:,:] = 0
	return out
	
def obtainWindow(frame, p, trans, projDim):
	windowScale = 2
	windowFrame = 32 / 2 * windowScale
	if (trans[0] >= 32 and trans[1] >= 32 and trans[0] <= projDim[0] - 32 and trans[1] <= projDim[1] - 32 and p[0] >= 32 and p[1] >= 32 and p[0] <= frame.shape[0] - 32 and p[1] <= frame.shape[1] - 32):
		return frame[p[0] - windowFrame:p[0] + windowFrame:windowScale,p[1] - windowFrame:p[1] + windowFrame:windowScale]
	return None
	