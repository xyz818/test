# -*- coding: utf-8 -*-
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  thread
import  numpy as np




def Camera_Init():
    global black_hig,black_low

    with picamera.PiCamera() as camera:
        camera.resolution = (320,240)
        sleep(1)
        kernel_4 = np.ones((5, 5), np.uint8)
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output,'rgb',use_video_port=True):
                dst = cv2.cvtColor(output.array,cv2.COLOR_RGB2BGR)

                dst = cv2.flip(dst,1) #fang zhuan
                gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

                # img = cv2.GaussianBlur(gray,(7,5),0)
                # cv2.imshow('gussian after',img)
                ret, bianary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                bianary = cv2.erode(bianary, kernel_4, iterations=2)
                bianary = cv2.dilate(bianary,kernel_4)
                # cv2.bitwise_not(bianary, bianary)
                bianary =  imclearborder(bianary)

                # cv2.imshow('dilate',bianary)
                # out_bianary, contours, hierarchy = cv2.findContours(bianary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # for cnt in range(len(contours)):
                #     area = cv2.contourArea(contours[cnt])
                #     if 5000 > area or area > 9900:
                #         cv2.drawContours(bianary, [contours[cnt]], 0, 0, -1)
                out_bianary, contours, hierarchy = cv2.findContours(bianary, cv2.RETR_EXTERNAL,
                                                                    cv2.CHAIN_APPROX_SIMPLE)
                # if (ENDTIME - STARTTIME).seconds < 10:
                for cnt in range(len(contours)):
                    epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
                    approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
                    corners = len(approx)
                    print  corners
                    if corners == 4:
                        cv2.drawContours(dst, contours, cnt, (0, 255, 0), 5)



                    #


                    #     cv2.drawContours(bianary, [contours[cnt]], 0, 0, -1)
                # print 'over'
                cv2.imshow('result',dst)

                cv2.waitKey(1)

                output.truncate(0)


def imclearborder(frame):
    extended = cv2.copyMakeBorder(frame,10,10,10,10,cv2.BORDER_CONSTANT,value=[255,255,255])
    mh,mw = extended.shape[:2]
    mask = np.zeros([mh+2,mw+2],np.uint8)
    cv2.floodFill(extended,mask,(5,5),(0,0,0),flags=cv2.FLOODFILL_FIXED_RANGE)
    # cropImg = extended[10:10+height]
    cv2.imshow('extends',extended)
    return extended



if __name__=='__main__':
    Camera_Init()
