# -*- coding: utf-8 -*-
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np


def Camera_Init():
    with picamera.PiCamera() as camera:
        camera.resolution = (320,240)
        sleep(1)
        kernel_4 = np.ones((5, 5), np.uint8)
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output,'rgb',use_video_port=True):
                dst = cv2.cvtColor(output.array,cv2.COLOR_RGB2BGR)
                # dst = cv2.flip(dst,1) #fang zhuan
                gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

                gray = cv2.GaussianBlur(gray, (7, 5), 0)

                ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # binary
                cv2.bitwise_not(thresh, thresh)
                image = cv2.Canny(thresh, 50, 150)
                # print 'shape'+str(dst.shape)
                # cv2.imshow('none', image)
                circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20,
                                           maxRadius=120)  # huofu
                if circles is not None:
                    # on_publish("/test/server", "Hello Python!", 1)
                    for circle in circles[0]:
                        x = int(circle[0])
                        y = int(circle[1])
                        r = int(circle[2])
                        dst = cv2.circle(dst, (x, y), r, (0, 0, 255), -1)
                cv2.imshow('dst',dst)

                cv2.waitKey(1)

                output.truncate(0)
if __name__=='__main__':
    Camera_Init()
