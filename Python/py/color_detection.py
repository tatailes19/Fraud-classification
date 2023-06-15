import cv2
import numpy as np

video=cv2.VideoCapture(0)

while True :
    true,frame=video.read()
    image=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(image,np.array([0, 10, 60]),np.array([20, 150, 255]))

    cv2.imshow("mask",mask)
    cv2.imshow("img",frame)

    cv2.waitKey(1)