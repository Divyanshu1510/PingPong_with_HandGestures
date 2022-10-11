import cv2
from image import overlayPNG
from HandTrackingModule import HandDetector
import numpy as np


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


# Background
imgBackground = cv2.imread("Resources/Background.png") 
# GameOver
imggameOver = cv2.imread("Resources/gameOver.png") 
#Ball
imgBall = cv2.imread("Resources/Ball.png", cv2.IMREAD_UNCHANGED) 
# Pad1
imgPad1 = cv2.imread("Resources/pad1.png", cv2.IMREAD_UNCHANGED) 
# Pad2
imgPad2 = cv2.imread("Resources/pad2.png", cv2.IMREAD_UNCHANGED) 


# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)


# Variable
gameOver = False
ballsPos = [100, 100]
speedX = 25
speedY = 25
score = [0 ,0]


while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # Find the hand and its landmarks
    hands = detector.findHands(img, flipType=False,draw=False)
    # hands, img= detector.findHands(img, flipType=False)
    
    # Background
    img = cv2.addWeighted(img, 0.05, imgBackground, 0.95, 0)

    # Checking Hands
    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            h1, w1, _ = imgPad1.shape
            y1 = y - h1//2
            y1 = np.clip(y1, 20, 415)

            if hand['type'] == 'Left':
                img = overlayPNG(img, imgPad1, (59, y1))
                if 62 < ballsPos[0] < 62 + w1 and y1 < ballsPos[1]< y1+h1:
                    speedX = -speedX
                    ballsPos[0] += 30
                    score[0] += 1

            if hand['type'] == 'Right':
                img = overlayPNG(img, imgPad2, (1195, y1))
                if 1195-50 < ballsPos[0] < 1195 and y1 < ballsPos[1]< y1+h1:
                    speedX = -speedX
                    ballsPos[0] -= 30
                    score[1] += 1
    # Game Over
    if ballsPos[0] < 40 or ballsPos[0] > 1200:
        gameOver = True

   
    if gameOver:
        img = imggameOver
        cv2.putText(img, str(score[0] + score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_COMPLEX, 2.5, (20, 0, 20), 5)


    # Game is not over so move the ball
    else:
        # Moving Ball
        if ballsPos[1]>= 500 or ballsPos[1] <= 10:
            speedY = -speedY


        ballsPos[0] += speedX
        ballsPos[1] += speedY


        # Draw ball
        img = overlayPNG(img, imgBall, ballsPos)

        cv2.putText(img, str(score[0]), (300, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)
        cv2.putText(img, str(score[1]), (900, 650), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 5)


    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        gameOver = False
        ballsPos = [100, 100]
        speedX = 15
        speedY = 15
        score = [0 ,0]
        imggameOver = cv2.imread("Resources/gameOver.png") 
    if key == ord('q'):
        quit()
