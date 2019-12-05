# -*- coding: utf-8 -*-
import os
import  sys
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np
import  thread




def Camera_Init():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        sleep(1)
        path = r'/root/opencv/opencv-3.2.0/data/haarcascades/'
        detector = cv2.CascadeClassifier(path+'haarcascade_upperbody.xml')

        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray,1.3,5)
                for (x,y,w,h) in faces:
                    img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)


                cv2.imshow('gra',frame)
                cv2.waitKey(1)
                output.truncate(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
    Camera_Init()