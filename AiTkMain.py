# -*- coding: utf-8 -*-
# Python 2.x使用这行
#from Tkinter import *
# Python 3.x使用这行
from Tkinter import *
import random
import predict
import numpy as np
import cv2
from PIL import Image, ImageTk
import threading
import time
import picamera
import  picamera.array

from widgets import *


class App:
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None



    def __init__(self, master):
        self.master = master
        self.initWidgets()
    def initWidgets(self):
        # 定义字符串元组
        books = ('疯狂Python讲义', '疯狂Swift讲义', '疯狂Kotlin讲义',\
            '疯狂Java讲义', '疯狂Ruby讲义')
        # self.image_ctl = Label(root,
        #         text='123')
        # self.image_ctl.place(x = 220, y = 36 , width=320, height=240)
        self.image_ctl =ICanvas(self.master, width=800, height=800)
        self.image_ctl.place(x=220,y=36,width=324,height=240)
        for i in range(len(books)):
            # 生成3个随机数
            ct = [random.randrange(256) for x in range(3)]
            grayness = int(round(0.299*ct[0] + 0.587*ct[1] + 0.114*ct[2]))
            # 将元组中3个随机数格式化成16进制数,转成颜色格式
            bg_color = "#%02x%02x%02x" % tuple(ct)
            # 创建Label，设置背景色和前景色
            lb = Button(root,
                text=books[i],
                fg = 'White' if grayness < 120 else 'Black',
                bg = bg_color,command=self.from_vedio)
            # 使用place()设置该Label的大小和位置
            lb.place(x = 20, y = 36 + i*36, width=180, height=30)

    def btnCommand(self):
        print '123'

    def from_vedio(self):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            # with picamera.PiCamera() as camera:
            # 	camera.resolution = (320, 240)
            # 	self.camera = camera
            if not self.camera.isOpened():
                print 'failure1'
                # mBox.showwarning('警告', '摄像头打开失败！')
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True

    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        print time.time()
        cv2.imshow('123',img)
        print wide
        print high
        # if wide > self.viewwide or high > self.viewhigh:
        #     wide_factor = self.viewwide / wide
        #     high_factor = self.viewhigh / high
        #     factor = min(wide_factor, high_factor)
        #     wide = int(wide * factor)
        #     if wide <= 0: wide = 1
        #     high = int(high * factor)
        #     if high <= 0: high = 1
        #     im = im.resize((wide, high), Image.ANTIALIAS)
        #     imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            # print '123'
            # with picamera.array.PiRGBArray(self.camera) as output:
            # 	for foo in self.camera.capture_continuous(output, 'rgb', use_video_port=True):
            # 		print '1234'
            # 		img_bgr = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
            # 		print '12345'
            # cv2.imshow('12', img_bgr)
            # self.image_ctl.delete('all')
            # img = self.img.resize((int(self.img_w * self.scale), int(self.img_h * self.scale)))

            img = cv2.transpose(img_bgr)
            img = cv2.flip(img, 1)
            print(img.shape)

            self.image_ctl.add(img)
            self.master.update_idletasks()
            self.master.update()
            # self.imgtk = self.get_imgtk(img_bgr)
            # self.image_ctl.configure(text='11')

            # self.image_ctl.configure(image=self.imgtk)

            # if time.time() - predict_time > 2:
            #     r, roi, color = self.predictor.predict(img_bgr)
            #     self.show_roi(r, roi, color)
            #     predict_time = time.time()

            # cv2.waitKey(1)
            # output.truncate(0)
        print("run end")



root = Tk()
root.title("AIOT DEMO")
# 设置窗口的大小和位置
# width x height + x_offset + y_offset
root.geometry("600x300+30+30")
App(root)
root.mainloop()

