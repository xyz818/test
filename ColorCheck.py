# -*- coding: utf-8 -*-
import os
import  sys
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np
import  thread

# from collections import  deque
reload(sys)
sys.setdefaultencoding('utf-8')



red_low = np.array([156,128,46])#lower
red_hig = np.array([179,255,255]) #higher

green_low = np.array([35,128,46])
green_high=np.array([77,255,255])

isBroad = False

def braod_Color(color):
    cmd = 'ilang  '+color
    os.system(cmd)



def Camera_Init():
    global isBroad
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        sleep(1)
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                cv2.imshow('1',output.array)
                frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                cv2.imshow('2',frame)
                red_marker = find_Marker(frame,red_low,red_hig)
                green_marker = find_Marker(frame,green_low,green_high)
                if red_marker <> 0:
                    box = cv2.boxPoints(red_marker)
                    box = np.int0(box)
                    cv2.drawContours(frame,[box],-1,(0,255,0),2)
                    print '红灯'
                    if isBroad:
                        thread.start_new_thread(braod_Color, ('红灯',))
                        isBroad = False
                else:
                    if green_marker<>0:
                        box = cv2.boxPoints(green_marker)
                        box = np.int0(box)
                        cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
                        print '绿灯'
                        if isBroad == False:
                            thread.start_new_thread(braod_Color, ('绿灯',))
                            isBroad = True
                cv2.imshow('capture',frame)
                cv2.waitKey(1)
                output.truncate(0)
    cv2.destroyAllWindows()
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