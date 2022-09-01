import cv2
import numpy as np
import time
import HandTracking as ht   #این کلاس را ما نوشتیم و ایمبورت میکنیم
import autopy   

### Variables Declaration
pTime = 0               # برای محاسبه سرعت
width = 640             # عرض تصویر
height = 480            # طول تصویر
frameR = 100            # سرعت بردازش
smoothening = 8         # بردازش تصویر
prev_x, prev_y = 0, 0   # نقاط قبلی
curr_x, curr_y = 0, 0   # نقاط موجود

cap = cv2.VideoCapture(0)   # دریافت تصویر از وب کم  با استفاده ازز cv2
cap.set(3, width)           
cap.set(4, height)

detector = ht.handDetector(maxHands=1)                  # در اینجا یک آبجکت از کلاسی که نوشتیم تعریف میکنیم   حداکثر یک دست در تصویر بیدا کند
screen_width, screen_height = autopy.screen.size()      # به دست اوردن سایز صفحه نمایش بنجره
while True:  #علت استفاده از حلقه while این است که ما باید تصویر بگیریم و چک کنم 
    success, img = cap.read()                           # در بالا ابتدا مشخص کردیم که وصل بشیم به وب کم حالا تصویرشا بشت هم میکیریم
    img = detector.findHands(img)                       # دست را در تصویر بیدا میکنیم
    lmlist, bbox = detector.findPosition(img)           # توی تصویر دست موقعیتشا در میاریم

    if len(lmlist)!=0:
        x1, y1 = lmlist[8][1:]
        x2, y2 = lmlist[12][1:]

        fingers = detector.fingersUp()      # چک میکنیم انشگتا تکتک بالا هستند
        # cv2.rectangle(img, (frameR, frameR), (width - frameR, height - frameR), (255, 0, 255), 2)   # Creating boundary box
        if fingers[1] == 1 and fingers[2] == 0:     # چک میکنیم که فقط انشگت اشاره بالا باشد
            x3 = np.interp(x1, (frameR,width-frameR), (0,screen_width))
            y3 = np.interp(y1, (frameR, height-frameR), (0, screen_height))

            curr_x = prev_x + (x3 - prev_x)/smoothening
            curr_y = prev_y + (y3 - prev_y) / smoothening

            autopy.mouse.move(screen_width - curr_x, curr_y)    # حال موس را تکان میدهیم
            cv2.circle(img, (x1, y1), 7, (255, 0, 255), cv2.FILLED)  # رنگ نوک انگشت
            prev_x, prev_y = curr_x, curr_y  # نوقعیت موس را ابدیت میکنیم

        if fingers[1] == 1 and fingers[2] == 1:     # اگر که هر دو انگشت بالا باشند وارد این شرط میشم
            length, img, lineInfo = detector.findDistance(8, 12, img)

            if length < 40:     # اگر دو انگشت به اندازه کافی به هم نزدیک باشند وارد این شرط میشم
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()    # عملیات کلیک موس رخ میدهد

    cTime = time.time()                     #زمان الان را میگیریم
    fps = 1/(cTime-pTime)                 #محاسبه فریم ریت نرم افزار
    pTime = cTime#
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)# نوشتن سرعت بر روی بنجره
    cv2.imshow("Image", img)             #نمایش تصویر
    cv2.waitKey(1)                  #واجب است ختما باشد بر حسب میلی ثانیه