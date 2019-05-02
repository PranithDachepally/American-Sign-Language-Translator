import cv2 as cv
import numpy as np
import Contour_Operation as co
import Training as st
import re

svm = st.trainSVM()
cap = cv.VideoCapture(0)
font = cv.FONT_HERSHEY_SIMPLEX
temp = 0
predicted_text = ''
previouslabel = None
previousText = ''
label = None
text =' '

while (cap.isOpened()):
    if temp ==1 :
        _, img = cap.read()
        print('if cam opened')
        cv.rectangle(img, (400, 200), (600, 450), (255, 0, 0), 4)
        hand = img[200:450, 400:600]
        hand_ycrcb = cv.cvtColor(hand, cv.COLOR_BGR2YCR_CB)
        # cv.imshow('y_crcb',hand_ycrcb)
        cv.waitKey(0)
        blur = cv.GaussianBlur(hand_ycrcb, (11, 11), 0)
        # cv.imshow('blur',blur)
        cv.waitKey(0)
        skin_min = np.array((0, 138, 67))
        skin_max = np.array((255, 173, 133))
        mask = cv.inRange(blur, skin_min, skin_max)
        # cv.imshow('mask',mask)
        cv.waitKey(0)
        contours, hierarchy = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, 2)
        cnt = co.getMaxContour(contours, 4000)
        #    if cnt!= None:
        gesture, label = co.getGestureImg(cnt, hand, mask, svm)
        if (label != None):
            if (temp == 0):
                previouslabel = label
            if previouslabel == label:
                previouslabel = label
            else:
                temp = 0
        if (temp == 40):
            if (label == 'P'):
                label = " "
            text = text + label
            if (label == 'O'):
                words = re.split(" +", text)
                words.pop()
                text = " ".join(words)
            print(text)
        # cv.imshow('frame', gesture)
        cv.putText(img, label, (50, 150), font, 8, (0, 125, 155), 2)
        cv.putText(img, text, (50, 450), font, 3, (0, 0, 255), 2)
        cv.imshow('Frame', img)
        # cv.imshow('Mask', mask)
        k = 0xFF & cv.waitKey(10)
        if k == 27:
            break
    else:
        _,img = cap.read()
        cv.rectangle(img,(400,200),(600,450),(255,0,0),4)
        cv.imshow('window',img)
        print('else cam opened')
    if cv.waitKey(1) & 0xFF == ord('q'):
        temp = 1

cap.release()
cv.destroyAllWindows()
