import cv2  # Can be installed using "pip install opencv-python"
import mediapipe as mp  # Can be installed using "pip install mediapipe"
import time
import math
import numpy as np


class handDetector():     #این کلاس را خودمان نوشته ایم و برای این است که توابع تشخیص دست را قرار دهیم
    def __init__(self, mode=False, maxHands=2, detectionCon=False, trackCon=0.5):  #به صورت بیش فرض مقادیری را قرار داده ایم در موقع ساخت کلاس تغییر میدهیم
        #هر کدام از مقادیری که در موقع ساخت کلاس میگیریم درون متغیر سلف  میریزیم که این مقادیر در شی کلاس بماند و باک نشود
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):    #تابع بیدا کردن دست ها در تصویر
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #به این دلیل که تصاویر دریافتی در وب کم با فرمتی که نیاز داریم متفاوت است باید عوض کنیم
        self.results = self.hands.process(imgRGB) 

        if self.results.multi_hand_landmarks:  #اگر که دست بیدا شد بر روی تصاویر شکل انرا میکشیم
            for handLms in self.results.multi_hand_landmarks:   #در این حلفه  تمام دست ها در تصویر را میکشیم
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img   #در انتها تصویر را برمیگردانیم

    def findPosition(self, img, handNo=0, draw=True):   # موقعیت دست را به دست میاوریم
        xList = []
        yList = []
        bbox = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):    #  چک میکنیم که کدام انگشت صاف میباشد و تمام انگشت ها را ابند میکنیم
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 3):  #میتوانیم مختصات تعداد انگشت های که ابند میشود را کم و زیاد کنیم

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):   # فاصله بین دو انگشت را مخاسبه میکنیم
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:  #اگر که کشیدن فالس شود این دایره ها کشیده نمیشود
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)   #رنگ هر کدام را میتوانیم تغییر دهیم
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)   #فاصبه بین دو انگشت مختصاتش کم میشود

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()