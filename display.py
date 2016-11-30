import numpy as np
import cv2

def display(projDim, cap, transform):

	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		out = np.ndarray((projDim[0], projDim[1], 3), np.float32)
		out[:,:,:] = 0.9
		# Display the resulting frame
		cv2.imshow('Smartwall',out)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()


	
