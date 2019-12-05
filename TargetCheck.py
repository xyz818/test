# -*- coding: utf-8 -*-
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np

import TargetApi
def Camera_Init():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        sleep(1)
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)


                filepath = 'tim.jpg'

                cv2.imwrite(filepath,frame)
                # print file.size

                # TargetApi.detection('imgage.png')
                # cv2.imshow('capture', frame)
                cv2.waitKey(1)
                output.truncate(0)
if __name__=='__main__':
    # Camera_Init()
    filepath = 'tim.jpg'

    TargetApi.detection(filepath)