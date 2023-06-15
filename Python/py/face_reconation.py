import numpy as np 
import cv2
import os

haar=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

people=[]
for i in os.listdir("D:/train"):
    people.append(i)


# features=np.load("features.npy")
# labels=np.load("labels.npy")

face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_recognizer.yml")


image=cv2.imread(r"D:\val\mindy_kaling\1.jpg")
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)



cv2.imshow("unidentified",gray)

face=haar.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)
for x,y,w,h in face :
    roi_=gray[y:y+h,x:x+w]
    label,conf=face_recognizer.predict(roi_)
    print(people[label])
    print(conf)
    cv2.putText(image,str(people[label]),(50,270),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


cv2.imshow("image",image)
cv2.waitKey(0)