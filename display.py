import numpy as np
import cv2

def calibrateCamera(projDim, cap):
	out = np.ndarray((projH, projW, 3), np.float32)
	out[:,:,:] = 1
	
	cv2.circle(out, (projDim[1] / 2, projDim[0] / 2), projDim[1] / 100, (0, 255, 0), thickness=-1)
	# Display the resulting frame
	cv2.imshow('Calibrating...',out)
	
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		
		
		# Display the resulting frame
		#cv2.imshow('Projector',out)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
	
#Projector dimensions
projH = 900
projW = 640

cap = cv2.VideoCapture(0)

transform = calibrateCamera((projH, projW), cap)


while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	
	# Our operations on the frame come here
	out = np.ndarray((projH, projW, 3), np.float32)
	out[:,:,:] = 0.9
	# Display the resulting frame
	cv2.imshow('Smartwall',out)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


	
