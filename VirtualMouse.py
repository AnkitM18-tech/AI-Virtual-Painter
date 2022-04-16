import cv2
import numpy as np
import HandTrackModule as htm
import time
import mouse
import pyautogui as pa

wCam, hCam = 640,480
pTime=  0
plocX,plocY = 0,0
clocX,clocY = 0,0
frameR = 100 #Frame reduction
smoothening = 7

cap = cv2.VideoCapture(0)
cap.set(3,wCam)
cap.set(4,hCam)
detector = htm.handDetector(maxHands=1,detectionCon=0.85)
wScr, hScr = pa.size()[0],pa.size()[1]

while True:
    success,img = cap.read()
    # Find hand Landmarks
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)
    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        # print(x1,y1,x2,y1)
        # Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
        # Only index finger - Moving mouse
        if fingers[1] and not fingers[2]:
            # Convert Coordinates
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            # Smoothen Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            # Move mouse
            mouse.move(wScr-clocX,clocY)
            cv2.circle(img,(x1,y1),15,(255,0,255),cv2.FILLED)
            plocX,plocY = clocX,clocY
        # Both Index and middle fingers up : Clicking Mouse
        if fingers[1] and fingers[2]:
            # Find distance between Fingers
            length,img,lineInfo = detector.findDistance(8,12,img)
            print(length)
            # Click mouse if distance is short
            if length < 40:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                mouse.click()
    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(20,50),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),3)
    # Display
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break