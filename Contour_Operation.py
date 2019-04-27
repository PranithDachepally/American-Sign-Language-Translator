import cv2 as cv
import numpy as np
import Training as st

#Get the biggest Controur
def getMaxContour(contours,minArea=200):
    maxC=None
    maxArea=minArea
    for cnt in contours:
        area=cv.contourArea(cnt)
        if(area>maxArea):
            maxArea=area
            maxC=cnt
    return maxC

#Get Gesture Image by prediction
def getGestureImg(cnt,img,th1,model):
    x,y,w,h = cv.boundingRect(cnt)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    imgT=img[y:y+h,x:x+w]
    imgT=cv.bitwise_and(imgT,imgT,mask=th1[y:y+h,x:x+w])
    imgT=cv.resize(imgT,(200,200))
    imgTG=cv.cvtColor(imgT,cv.COLOR_BGR2GRAY)
    resp=st.predict(model,imgTG)
    print(chr(int(resp[1])+64))
    img=cv.imread('TrainData/'+chr(int(resp[1])+64)+'_2.jpg')
    return img,chr(int(resp[1])+64)