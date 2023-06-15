import cv2 as cv
import numpy as np

img=cv.imread("D:/pictures/just pics/photo_5841380858475428192_y.jfif")
img=cv.resize(img,(500,700))

blank=np.zeros(img.shape,np.uint8)

blur=cv.GaussianBlur(img,(3,3),cv.BORDER_DEFAULT)

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
canny=cv.Canny(blur,125,175)

#ret,tresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
contours,hir=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)
print(len(contours))

cv.drawContours(blank,contours,-1,(255,255,255),1)

cv.imshow("photo",gray)
cv.imshow("photo",canny)
#cv.imshow("photo",tresh)
cv.imshow("photo1",img)
cv.imshow("blank",blank)

cv.waitKey(0)