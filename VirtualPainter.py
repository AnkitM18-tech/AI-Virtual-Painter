import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

# BG Path
folderPath = "./bgs"
bgList = os.listdir(folderPath)
# print(bgList)
overLayList = []
brushThickness = 15
eThickness = 50
xp,yp = 0,0
imgCanvas = np.zeros((480,640,3),np.uint8)

# Adding images to the Overlay image list
for img in bgList:
    image = cv2.imread(f"{folderPath}/{img}")
    overLayList.append(image)
# print(len(overLayList))

# Initial setup
header = overLayList[0]
drawColor = (222,46,235)
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

detector = htm.handDetector(detectionCon=0.85)

while True:
    # Video Feed
    success,img = cap.read()
    img = cv2.flip(img,1)
    # OverLay Image
    img[0:90,0:640] = header
    # will give a trasparent effect
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    # Finding Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw=False)
    if len(lmList) != 0:
        # print(lmList)
        # tip of the index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        # If selection mode - Two fingers up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            # print("Selection Mode")
            # Checking for the click on the Pallete
            if y1 < 90:
                if 142<x1<184:
                    header = overLayList[0]
                    drawColor = (222, 46, 235)
                elif 226<x1<267:
                    header = overLayList[1]
                    drawColor = (46, 46, 235)
                elif 303<x1<354:
                    header = overLayList[2]
                    drawColor = (235, 58, 46)
                elif 393<x1<438:
                    header = overLayList[3]
                    drawColor = (46, 146, 235)
                elif 476<x1<562:
                    header = overLayList[4]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),drawColor,cv2.FILLED)
        # If drawing mode - Index finger up
        if fingers[1] and not fingers[2]:
            cv2.circle(img,(x1,y1),15,(255,255,0),cv2.FILLED)
            # print("Drawing Mode")
            if xp == 0 and yp == 0:
                xp,yp = x1,y1
            if drawColor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawColor,eThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eThickness)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp = x1,y1
    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _,imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    # Displaying the Images
    cv2.imshow("Image",img)
    # cv2.imshow("Image Canvas",imgCanvas)
    # cv2.imshow("Image Inverse",imgInv)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break