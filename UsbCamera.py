# -*- coding: utf-8 -*-
import cv2
import  numpy as np
import serial
import binascii
import time
# 打开串口
ser = serial.Serial("/dev/ttyAMA0", 9600)
red_low = np.array([156,128,46])#lower
red_hig = np.array([179,255,255]) #higher

green_low = np.array([35,128,46])
green_high=np.array([77,255,255])

isAuto = True

def writeser(datt):
    hexstr = binascii.a2b_hex(datt)
    print datt
    ser.write(hexstr)


def Camera_Init():
    global  isAuto
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret, frame = cap.read()
        cv2.imshow('capture',frame)
        # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        red_marker = find_Marker(frame, red_low, red_hig)
        green_marker = find_Marker(frame, green_low, green_high)
        if red_marker <> 0:
            box = cv2.boxPoints(red_marker)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

            if isAuto:
                print '红灯'
                writeser("AA2BBB")
                time.sleep(0.1)
                isAuto = False
        else:
            if green_marker <> 0:
                box = cv2.boxPoints(green_marker)
                box = np.int0(box)
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

                if isAuto == False:
                    print '绿灯'
                    writeser("AA1100")
                    time.sleep(0.1)
                    isAuto = True
        cv2.imshow('capture', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
def find_Marker(img,low,hig):
    global red_hig,red_low
    kernel_2 = np.ones((2, 2), np.uint8)  # juan ji ceng
    kernel_3 = np.ones((3, 3), np.uint8)
    kernel_4 = np.ones((4, 4), np.uint8)
    if img is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, low, hig)
        # lv bo
        erosion = cv2.erode(mask, kernel_4, iterations=1)
        erosion = cv2.erode(erosion, kernel_4, iterations=1)
        dilation = cv2.dilate(erosion, kernel_4, iterations=1)
        dilation = cv2.dilate(dilation, kernel_4, iterations=1)
        # target
        target = cv2.bitwise_and(img, img, mask=dilation)
        # lvbo
        ret, binary = cv2.threshold(dilation, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow('binary',binary)
        (_, cnts, _) = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts == []:
            return 0
    c = max(cnts,key = cv2.contourArea)
    return cv2.minAreaRect(c)

if __name__=='__main__':
    Camera_Init()