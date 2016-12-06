import numpy as np
import cv2

cap = cv2.VideoCapture(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

min_RGB = np.array([130, 100, 0], np.uint8)
max_RGB = np.array([255, 223, 196], np.uint8)

subtractor = cv2.BackgroundSubtractorMOG2(history=100, varThreshold=5.0)

while(True):
    ret, frame = cap.read()

    fgmask = subtractor.apply(frame, learningRate=0.005)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    fgmask = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR);

    fgmask = (fgmask / 255) * frame

    # Convert image to YCrCb
    image = cv2.cvtColor(fgmask,cv2.COLOR_BGR2YCR_CB)

    # Find region with skin tone in YCrCb image
    skinRegion = cv2.inRange(image,min_YCrCb,max_YCrCb)
    #skinRegion = cv2.inRange(image, min_RGB, max_RGB)
    # Do contour detection on skin region
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contour on the source image

    count = 0

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        w_over_h = w / h
        h_over_w = h / w
        extent = area / (w * h)
        if area > 100 and area < 4500 and w_over_h < 4 and h_over_w < 4 and extent > 0.25:
            count += 1
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.drawContours(fgmask, contours, i, (0, 255, 0), 3)
            cv2.circle(fgmask, (cX, cY), 7, (0, 0, 255), -1)
        elif area > 50:
            count += 1
            M = cv2.moments(c)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(fgmask, (cX, cY), 7, (0, 0, 255), -1)
            cv2.drawContours(fgmask, contours, i, (0, 0, 255), 3)
            

    print count

    cv2.imshow('frame',fgmask)
    cv2.imshow('orig', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()