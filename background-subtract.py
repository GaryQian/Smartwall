import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

##how to change parameters to increase threshold???
##MOG1 seems like it would work really well if we could actually tweak the thresholding.
subtractor = cv2.BackgroundSubtractorMOG(history=100000000, nmixtures=5, backgroundRatio=0.1, noiseSigma=0)

#subtractor = cv2.BackgroundSubtractorMOG2(history=100, varThreshold=20.0, bShadowDetection=True)

while(True):
    ret, frame = cap.read()

    fgmask = subtractor.apply(frame, learningRate=0)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR);

    fgmask = (fgmask / 255) * frame

    # Convert image to YCrCb
    imageYCrCb = cv2.cvtColor(fgmask,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

    # Do contour detection on skin region
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contour on the source image

    count = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 200 and area < 1000:
            count += 1
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(fgmask, contours, i, (0, 255, 0), 3)
            cv2.circle(fgmask, (cX, cY), 7, (0, 0, 255), -1)
    print count

    cv2.imshow('frame',fgmask)
    cv2.imshow('orig', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()