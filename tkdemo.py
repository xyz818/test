# -*- coding: utf-8 -*-
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  thread
import paho.mqtt.client as mqtt
import json
from datetime import  datetime
MQTTHOST = "192.168.0.177"
MQTTPORT = 1883
mqttClient = mqtt.Client()
STARTTIME = datetime.now()
ENDTIME = datetime.now()
isPloy = False
MAC = '746bda48f3633cdf'
shape = 0
# 连接MQTT服务器
def on_mqtt_connect(client,userdata,flags,rc):
    print(str(rc))
    mqttClient.subscribe("SF/MsgUpStream", 1)


# publish 消息
def on_publish(topic, payload, qos):
    print 'send mqtt'
    mqttClient.publish(topic, payload, qos)


# 消息处理函数
def on_message(client, userdata, msg):
    print(msg.topic + " " + ":" + str(msg.payload))
    # Camera_Init()
    global isPloy, STARTTIME,MAC,shape
    JSON = json.loads(msg.payload)
    print JSON['mac']
    if MAC == JSON['mac'] and JSON['msgType']=='0' and JSON['data']=='0': #json cmp
        STARTTIME = datetime.now()
        shape = int(JSON['name'])
        isPloy = True
        print isPloy,shape
        print STARTTIME


# subscribe 消息
def on_subscribe():
    mqttClient.on_message = on_message  #


def mqtt_main():
    mqttClient.on_connect = on_mqtt_connect
    mqttClient.on_message =  on_message
    mqttClient.connect(MQTTHOST, MQTTPORT, 60)
    mqttClient.loop_forever()
    # on_publish("/test/server", "Hello Python!", 1)
    # on_subscribe()
    # while True:
    #     pass



def Camera_Init():
    global isPloy,STARTTIME,ENDTIME,MAC,shape
    jsonData = {}
    jsonData['sendType'] = '0'
    jsonData['mac'] = MAC
    jsonData['addr'] = ''
    jsonData['msgType'] = '0'

    with picamera.PiCamera() as camera:
        camera.resolution = (320,240)
        sleep(1)
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output,'rgb',use_video_port=True):
                dst = cv2.cvtColor(output.array,cv2.COLOR_RGB2BGR)
                # dst = cv2.flip(dst,1) #fang zhuan
                gray = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY) #gray

                if shape == 1:
                    if isPloy:
                        ENDTIME = datetime.now()
                        img = cv2.GaussianBlur(gray, (7, 5), 0)
                        # cv2.imshow('gussian after', img)
                        ret, bianary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        out_bianary, contours, hierarchy = cv2.findContours(bianary, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in range(len(contours)):
                            area = cv2.contourArea(contours[cnt])
                            if 5000 > area or area > 9900:
                                 cv2.drawContours(bianary, [contours[cnt]], 0, 0, -1)
                        out_bianary, contours, hierarchy = cv2.findContours(bianary, cv2.RETR_EXTERNAL,
                                                                            cv2.CHAIN_APPROX_SIMPLE)
                        if (ENDTIME - STARTTIME).seconds < 10:
                            for cnt in range(len(contours)):
                                epsilon = 0.01 * cv2.arcLength(contours[cnt], True)
                                approx = cv2.approxPolyDP(contours[cnt], epsilon, True)
                                corners = len(approx)
                                if corners==4:
                                    jsonData['name'] = '1'
                                    cv2.drawContours(dst, contours,cnt,(0,255,0), 5)
                                    # print 'len:' + str(len(circles[0]))
                                    jsonData['data'] = '1'
                                    thread.start_new_thread(on_publish, ("SF/MsgDownStream", json.dumps(jsonData), 1,))
                                        # on_publish("/test/server", "Hello Python!", 1)
                                    shape=0
                                    isPloy = False
                        else:
                            jsonData['name'] = '1'
                            jsonData['data'] = '2'
                            thread.start_new_thread(on_publish, ("SF/MsgDownStream", json.dumps(jsonData), 1,))
                            isPloy = False
                            shape = 0
                            # area = cv2.contourArea(contours[cnt])
                            # p = cv2.arcLength(contours[cnt], True)
                if shape == 2:
                    # cv2.imshow('img', gray)
                    gray = cv2.GaussianBlur(gray, (7, 5), 0)
                    ret,thresh = cv2.threshold(gray,128,255,cv2.THRESH_BINARY) #binary
                    cv2.bitwise_not(thresh,thresh)
                    image = cv2.Canny(thresh,50,150)
                    # print 'shape'+str(dst.shape)
                    # cv2.imshow('none', image)
                    circles = cv2.HoughCircles(image,cv2.HOUGH_GRADIENT,1,100,param1=100,param2=30,minRadius=20,maxRadius=120)#huofu

                    if isPloy:
                        ENDTIME = datetime.now()
                        if (ENDTIME - STARTTIME).seconds < 10:
                            if circles is not None:
                                jsonData['name'] = '2'
                                print 'len:'+str(len(circles[0]))
                                jsonData['data']='1'
                                thread.start_new_thread(on_publish,("SF/MsgDownStream",json.dumps( jsonData ), 1,))
                                    # on_publish("/test/server", "Hello Python!", 1)
                                for circle in circles[0]:
                                    x = int(circle[0])
                                    y = int(circle[1])
                                    r = int(circle[2])
                                    dst = cv2.circle(dst,(x,y),r,(0,0,255),-1)
                                isPloy = False
                                shape = 0
                        else:
                            jsonData['name'] = '2'
                            jsonData['data'] = '2'
                            thread.start_new_thread(on_publish, ("SF/MsgDownStream",json.dumps( jsonData ), 1,))
                            isPloy = False
                            shape = 0
                    # cv2.imshow('res',imgCir)
                # if circles is not None:
                #     print 'len:' + str(len(circles[0]))
                #     # thread.start_new_thread(on_publish, ("/test/server", "Hello Python success!", 1,))
                #     # on_publish("/test/server", "Hello Python!", 1)
                #     for circle in circles[0]:
                #         x = int(circle[0])
                #         y = int(circle[1])
                #         r = int(circle[2])
                #         dst = cv2.circle(dst, (x, y), r, (0, 0, 255), -1)
                cv2.imshow('target',dst)
                cv2.waitKey(1)

                output.truncate(0)
    cv2.destroyAllWindows()


if __name__=='__main__':
    thread.start_new_thread( Camera_Init,())
    mqtt_main()
    # starttime = datetime.now()
    # print starttime
    # sleep(2)
    # endtime =datetime.now()
    # print  endtime
    # print (endtime-starttime).seconds

