# -*- coding: utf-8 -*-
from widgets import *
import os
import  sys
reload(sys)
sys.setdefaultencoding('utf-8')
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np
import  thread
import time
sys.path.append('/root/caffe/python')
import json
import paho.mqtt.client as mqtt
from imutils.object_detection import non_max_suppression

import predict
import imutils
import requests

MAC = 'LXG001'
BASE_URL = 'http://192.168.0.177:18080/RabbitMqServer/deviceinfo/'+MAC




from datetime import  datetime
MQTTHOST = "192.168.0.177"
MQTTPORT = 1883


#color check
red_low = np.array([156,128,46])#lower
red_hig = np.array([179,255,255]) #higher

green_low = np.array([35,128,46])
green_high=np.array([77,255,255])
#end color check

mqttClient = mqtt.Client()
STARTTIME = datetime.now()
ENDTIME = datetime.now()
_type = -1
isCheckColor = False
isCheckHuman = False
isCheckPlate = False

isHasHuman = False
isHasRed = False

isPloy = False

count = 0


def get_pid():
    rsp = requests.get(BASE_URL)
    str = rsp.content
    data = json.loads(str)
    return data['pi_seq']

pid = 0

# 连接MQTT服务器
def on_mqtt_connect(client,userdata,flags,rc):
    global  pid
    print(str(rc))
    pid = int(get_pid())
    mqttClient.subscribe("raspi_down/"+str(pid), 1)
    print pid

# publish 消息
def on_publish(topic, payload, qos):
    print 'send mqtt'
    mqttClient.publish(topic, payload, qos)




# 消息处理函数
def on_message(client, userdata, msg):
    print("123")
    print(msg.topic + " " + ":" + str(msg.payload))
    # Camera_Init()
    global MAC,STARTTIME,_type,isCheckColor,isCheckHuman,isCheckPlate,isHasHuman,isHasRed,count
    print '1'
    JSON = json.loads(msg.payload)
    print '2'
    print JSON['deviceId']
    if MAC == JSON['deviceId']: #json cmp
        STARTTIME = datetime.now()
        _type = int(JSON['checkType'])
        count = 0
        if _type == 1:
            isCheckColor = False
            isHasRed = False
        if _type == 2:
            isCheckHuman = False
            isHasHuman = False
        if _type == 3:
            isCheckPlate = False
        # isPloy = True
        print isPloy,_type
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








class Extractor_GUI():
    def __init__(self):
        self.__init_gui()
        # self.__init_model()

        self.isCheckScraw = True
    def __init_gui(self):

        # self.currentBroadValue = '123'
        self.window = tk.Tk()
        self.window.wm_title('AIOT智能交通综合案例')
        # books = ('颜色检测', '物体识别 ', 'd', \
        #          '涂鸦检测','结束检测')
        self.window.config(background='#FFFFFF')
        self.canvas = ICanvas(self.window, width=320, height=240)
        self.canvas.grid(row=0, column=0)
        self.fm_control = tk.Frame(self.window, width=200, height=240, background='#FFFFFF')
        self.fm_control.grid(row=0, column=1, padx=10, pady=2)


        # self.fm_status = tk.Frame(self.window, width=320, height=120, background='#FFFFFF')
        # self.fm_status.grid(row=1, column=0, padx=0, pady=2)
        self.lbStatus = tk.Label(self.fm_control,text='提示：没有需要识别的...',bg='Yellow',justify='left',anchor='w')
        self.lbStatus.place(x = 0,y = 10,width=320,height=30)
        self.lbValue = tk.Label(self.fm_control, text='等待识别中... ', bg='Green', justify='left', anchor='w')
        self.lbValue.place(x=0, y=70, width=320, height=30)
        self.predictor = predict.CardPredictor()
        self.predictor.train_svm()

        # self.fm_title = tk.Frame(self.window, width=200, height=120, background='#FFFFFF')
        # self.fm_title.grid(row=1, column=1, padx=0, pady=2)
        # self.lbCue = tk.Label(self.fm_title, text='提示：点击上方按钮，选择需要识别的... ', bg='Blue', fg='White',justify='left', anchor='w')
        # self.lbCue.place(x=0, y=30, width=200, height=50)
        self.__action_read_frame()
        # self.btn_prev_frame1 = tk.Button(self.fm_status, text='Start1', command=self.__action_read_frame)
        # self.btn_prev_frame1.grid(row=0, column=0, padx=10, pady=2)
        #
        # self.btn_next_frame3 = tk.Button(self.fm_status, text='Start2', command=None)
        # self.btn_next_frame3.grid(row=1, column=0, padx=10, pady=20)


    def checkHuman(self,hog,frame):
        global  isCheckHuman,isHasHuman
        image = imutils.resize(frame, width=min(400, frame.shape[1]))
        # orig = image.copy()
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8), padding=(16, 16), scale=1.05)

        # draw the original bounding boxes
        # for (x, y, w, h) in rects:
        #     cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.75)

        if len(pick) <> 0 :
            # draw the final bounding boxes
            for (x, y, w, h) in pick:
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            isHasHuman = True
            # isCheckHuman = True




        # color check




    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)

        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def show_roi(self, r, roi, color):
        if r:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi = Image.fromarray(roi)
            self.imgtk_roi = ImageTk.PhotoImage(image=roi)
            # self.roi_ctl.configure(image=self.imgtk_roi, state='enable')
            # self.r_ctl.configure(text=str(r))
            # self.update_time = time.time()
            # try:
            #     c = self.color_transform[color]
            #     self.color_ctl.configure(text=c[0], background=c[1], state='enable')
            # except:
            #     self.color_ctl.configure(state='disabled')
        # elif self.update_time + 8 < time.time():
            # self.roi_ctl.configure(state='disabled')
            # self.r_ctl.configure(text="")
            # self.color_ctl.configure(state='disabled')

    def check_chepai(self,img_bgr):
        global  pid,MAC

        # self.imgtk = self.get_imgtk(img_bgr)
        #
        # self.image_ctl.configure(image=self.imgtk)


        global  isCheckPlate
        # self.show_roi(r, roi, color)
        print 'r:'
        # try:
        r, roi, color = self.predictor.predict(img_bgr)
        if len(r) > 0 and color is not None:
            strV = ''
            # print r
            for v in r:
                strV = strV + (v.decode('utf-8'))
            self.lbValue['text'] = strV
            jsonData = {}
            jsonData['deviceId'] = MAC
            jsonData['checkType'] = 3
            jsonData['sensorId'] = 'www'
            jsonData['checkValue'] = strV
            print 'over'
            thread.start_new_thread(on_publish, ("/uptohtml/" + str(pid), json.dumps(jsonData, ensure_ascii=False), 1,))
            print '12'
            isCheckPlate = True
        # except:
        #     print 'pass'
        #     pass



        # predict_time = time.time()

        # cv2.waitKey(1)
        # output.truncate(0)


    def find_Color(self,img, low, hig):
        global red_hig, red_low
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
        c = max(cnts, key=cv2.contourArea)
        return cv2.minAreaRect(c)



    #hong lv deng jiance
    def check_color(self, frame):
        # global isBroad
        global  MAC,isCheckColor,isHasRed
        red_marker = self.find_Color(frame, red_low, red_hig)
        green_marker = self.find_Color(frame, green_low, green_high)
        if red_marker <> 0:
            box = cv2.boxPoints(red_marker)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)

            if cmp(self.lbValue['text'],'红')<>0:
                self.lbValue['text'] = '红'
                isHasRed = True
                # jsonData= {}
                # jsonData['deviceId']=MAC
                # jsonData['checkType']= 1
                # jsonData['checkValue']='red'
                # # isCheckColor = True
                # thread.start_new_thread(on_publish, ("raspi_up/"+MAC, json.dumps(jsonData), 1,))

            # if not isBroad:
            #     isBroad = True
            #     thread.start_new_thread(self._braod_, ('红',))

        elif green_marker <> 0:
                box = cv2.boxPoints(green_marker)
                box = np.int0(box)
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
                if cmp(self.lbValue['text'], '绿') <> 0:
                    self.lbValue['text'] = '绿'
                    isHasRed = False
                    # jsonData = {}
                    # jsonData['deviceId'] = MAC
                    # jsonData['checkType'] = 1
                    # jsonData['checkValue'] = 'green'
                    # # isCheckColor = True
                    # thread.start_new_thread(on_publish, ("raspi_up/" + MAC, json.dumps(jsonData), 1,))
                # if not isBroad:
                #     isBroad = True
                #     thread.start_new_thread(self._braod_, ('绿',))

        else:
            self.lbValue['text'] = '未检测到红绿灯'
            isHasRed = False



    def __action_read_frame(self):
        self.from_video()





    def from_video(self):
        global _type,STARTTIME,ENDTIME,isCheckColor,isCheckHuman,isCheckPlate,isHasHuman,count,pid
        # cap = cv2.VideoCapture(0)
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 320)
            sleep(1)
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            with picamera.array.PiRGBArray(camera) as output:
                for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                    frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                    M = cv2.getRotationMatrix2D((160,160),90,1)
                    frame = cv2.warpAffine(frame,M,(320,240))
                    # frame = frame.trans(frame)
                    # frame = cv2.flip(frame,1)
                    # cv2.
                    # print 'rec'
                    if _type == 1:  # honglv deng
                        # print 'redgreen'
                        self.lbStatus['text'] = '提示：当前正在检测红绿灯...'
                        if isCheckColor == False:
                            ENDTIME = datetime.now()
                            self.check_color(frame)
                            if ((ENDTIME - STARTTIME).seconds+1) % 6 == 0:
                                if count == 0:
                                    if isHasRed == True:
                                        jsonData= {}
                                        jsonData['deviceId']=MAC
                                        jsonData['checkType']= 1
                                        jsonData['checkValue']='red'
                                        jsonData['sensorId'] = 'www'
                                        # isCheckColor = True
                                        thread.start_new_thread(on_publish, ("/uptohtml/"+str(pid), json.dumps(jsonData), 1,))
                                        count = 1
                                    else:
                                        jsonData = {}
                                        jsonData['deviceId'] = MAC
                                        jsonData['checkType'] = 1
                                        jsonData['checkValue'] = 'green'
                                        jsonData['sensorId'] = 'www'
                                        isCheckColor = True
                                        thread.start_new_thread(on_publish, ("/uptohtml/"+str(pid), json.dumps(jsonData), 1,))
                                        count = 1
                            else:
                                count = 0




                    elif _type==2: #renti jiance
                        # print 'wuti1'
                        self.lbStatus['text'] = '提示：当前正在检测行人...'
                        if isCheckHuman == False:
                            ENDTIME = datetime.now()
                            self.checkHuman(hog, frame)
                            if ((ENDTIME - STARTTIME).seconds + 1) % 6 == 0:
                                if count == 0:
                                    if isHasHuman == True:
                                        jsonData = {}
                                        self.lbValue['text'] = '检测到行人'
                                        jsonData['deviceId'] = MAC
                                        jsonData['checkType'] = 2
                                        jsonData['checkValue'] = '1'
                                        jsonData['sensorId'] = 'www'
                                        thread.start_new_thread(on_publish, ("/uptohtml/"+str(pid), json.dumps(jsonData), 1,))
                                        count = 1
                                        isHasHuman = False
                                    else:
                                        jsonData = {}
                                        self.lbValue['text'] = '无人通过'
                                        jsonData['deviceId'] = MAC
                                        jsonData['checkType'] = 2
                                        jsonData['checkValue'] = '0'
                                        jsonData['sensorId'] = 'www'
                                        thread.start_new_thread(on_publish, ("/uptohtml/"+str(pid), json.dumps(jsonData), 1,))
                                        count = 1
                                        isCheckHuman = True
                            else:
                                count = 0
                    elif _type == 3:#chepaishibie
                        self.lbStatus['text']='提示：当前正在检测车牌...'
                        ENDTIME = datetime.now()
                        # if (ENDTIME - STARTTIME).seconds > 2:
                        if isCheckPlate==False:
                            self.check_chepai(frame)
                            if (ENDTIME - STARTTIME).seconds > 5:
                                jsonData = {}
                                jsonData['deviceId'] = MAC
                                jsonData['checkType'] = 3
                                jsonData['checkValue'] = 'none'
                                jsonData['sensorId'] = 'www'
                                self.lbValue['text'] = '检测超时'
                                thread.start_new_thread(on_publish, ("/uptohtml/"+str(pid), json.dumps(jsonData,ensure_ascii=False), 1,))
                                isCheckPlate = True




                    self.canvas.add(frame)
                            # print frame.shape
                    self.window.update_idletasks()
                    self.window.update()
                            # cv2.imshow('gra', frame)
                    cv2.waitKey(1)
                    output.truncate(0)
            cv2.destroyAllWindows()

    def launch(self):
        self.window.mainloop()

if __name__ == '__main__':
    thread.start_new_thread(mqtt_main, ())
    ext = Extractor_GUI()
    ext.launch()


    # ext.from_video()
