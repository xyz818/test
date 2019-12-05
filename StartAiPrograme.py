# -*- coding: utf-8 -*-
from widgets import *
import os
import  sys
import picamera
import  picamera.array
import  cv2
from time import  sleep
import  numpy as np
import  thread
import tensorflow as tf
import time
sys.path.append('/root/caffe/python')
import caffe
from collections import deque
#color check
red_low = np.array([156,128,46])#lower
red_hig = np.array([179,255,255]) #higher

green_low = np.array([35,128,46])
green_high=np.array([77,255,255])
#end color check

isBroad = False  # shi fou  zhengzai bobao
# isCheckScraw = True


#scraw check
size = 200
rz = 28.0
ratio = rz / size

draw_on = False
last_pos = (0, 0)
color = (255, 255, 255)
radius = 8

# caffe.set_device(0)
caffe.set_mode_cpu()
net = caffe.Net('/root/caffe/examples/makefiles/deploy.prototxt', '/root/caffe/examples/makefiles/deploy.caffemodel', caffe.TEST)
fcl2 = open('/root/caffe/examples/makefiles/class_list.txt', 'r')
fcl = open('/root/caffe/examples/makefiles/class_list_chn.txt', 'r')
class_list = fcl.readlines()
class_list_eng = fcl2.readlines()
cls = []
for line in class_list:
    cls.append(line.split(' ')[0])

# cap = cv2.VideoCapture(0)
isCheckScraw = False
#tensorflow target check
sys.path.append("/root/models/research/")
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


MODEL_NAME = '/root/ssd_mobilenet_v1_coco_2018_01_28'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('/root/models/research/object_detection/data', 'mscoco_label_map.pbtxt')
model_path = "/root/ssd_mobilenet_v1_coco_2018_01_28/model.ckpt"
NUM_CLASSES = 90
end= time.clock()
print 'tensor 1'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
isCheckTarget = False


# 设定红色阈值，HSV空间
redLower = np.array([170, 100, 100])
redUpper = np.array([179, 255, 255])
# 初始化追踪点的列表
mybuffer = 16
pts = deque(maxlen=mybuffer)
counter = 0

class Extractor_GUI():
    def __init__(self):
        self.__init_gui()
        # self.__init_model()

        self.isCheckScraw = True
    def __init_gui(self):
        self.btnPos = -1
        # self.currentBroadValue = '123'
        self.window = tk.Tk()
        self.window.wm_title('人工智能图像识别')
        books = ('灰度转换','二值转换','边缘检测','颜色检测', '圆形检测', \
                  '物体追踪','人脸检测','涂鸦检测','物体识别','结束检测')
        self.window.config(background='#FFFFFF')

        self.fm_title1 = tk.Frame(self.window,width=200,height=30,background='#FFFFFF')
        self.fm_title1.grid(row=0, column=0, padx=0, pady=0)
        self.lb1 = tk.Label(self.fm_title1, text='实验简介', bg='Yellow', fg='Black', justify='left', anchor='w')
        self.lb1.place(x=0, y=0, width=200, height=30)

        self.fm_title2 = tk.Frame(self.window,width=640,height=30,background='#FFFFFF')
        self.fm_title2.grid(row=0, column=1, padx=0, pady=0)

        self.lb2 = tk.Label(self.fm_title2, text='图像采集区', bg='Green', fg='White', justify='left', anchor='w')
        self.lb2.place(x=0, y=0, width=640, height=30)

        self.fm_title3 = tk.Frame(self.window,width=200,height=30,background='#FFFFFF')
        self.fm_title3.grid(row=0, column=2, padx=0, pady=0)

        self.lb3 = tk.Label(self.fm_title3, text='实验案例列表', bg='Blue', fg='White', justify='left', anchor='w')
        self.lb3.place(x=0, y=0, width=200, height=30)


        self.fm_brief = tk.Frame(self.window,width=200,height=480,background='#FFFFFF')
        self.fm_brief.grid(row=1, column=0, padx=0, pady=0)

        self.lbBrief = tk.Label(self.fm_brief,text='',bg='Blue',fg='White',justify='left',anchor='w',wraplength=180)
        self.lbBrief.place(x=0, y=0, width=200, height=480)
        self.fm_promt = tk.Frame(self.window, width=200, height=50,background='#FFFFFF')
        self.fm_promt.grid(row=2, column=0, padx=0, pady=0)
        self.canvas = ICanvas(self.window, width=640, height=480)
        self.canvas.grid(row=1, column=1)
        self.fm_control = tk.Frame(self.window, width=200, height=480, background='#FFFFFF')
        self.fm_control.grid(row=1, column=2, padx=0, pady=0)
        # self.btn_prev_frame = tk.Button(self.fm_control, text='Start', command=self.__action_read_frame)
        # self.btn_prev_frame.grid(row=0, column=0, padx=10, pady=2)
        # self.lb_current_frame = tk.Label(self.fm_control, background='#FFFFFF')
        # self.lb_current_frame.grid(row=0, column=1, padx=10, pady=2)
        # self.lb_current_frame['text'] = '----'
        # self.btn_next_frame = tk.Button(self.fm_contrlbBriefol, text='Next Frame', command=None)
        # self.btn_next_frame.grid(row=0, column=2, padx=10, pady=2)
        #
        for i in range(len(books)):
            # 生成3个随机数
            ct = [random.randrange(256) for x in range(3)]
            grayness = int(round(0.299 * ct[0] + 0.587 * ct[1] + 0.114 * ct[2]))
            # 将元组中3个随机数格式化成16进制数,转成颜色格式
            bg_color = "#%02x%02x%02x" % tuple(ct)
            # 创建Label，设置背景色和前景色
            lb = tk.Button(self.fm_control,
                           text=books[i],
                           fg='White' if grayness < 120 else 'Black',
                           bg=bg_color,command=lambda i=i:self.btn_click(i))


            # 使用place()设置该Label的大小和位置
            lb.place(x=0, y=5 + i * 32, width=180, height=30)

        # 使用place()设置该Label的大小和位置

        self.fm_status = tk.Frame(self.window, width=640, height=50, background='#FFFFFF')
        self.fm_status.grid(row=2, column=1, padx=0, pady=0)
        self.lbStatus = tk.Label(self.fm_promt,text='提示：没有需要识别的...',bg='Yellow',justify='left',anchor='w')
        self.lbStatus.place(x = 0,y = 0,width=200,height=50)
        self.lbValue = tk.Label(self.fm_status, text='等待识别中... ', bg='Green', justify='left', anchor='w')
        self.lbValue.place(x=0, y=0, width=640, height=50)
        self.fm_title = tk.Frame(self.window, width=200, height=50, background='#FFFFFF')
        self.fm_title.grid(row=2, column=2, padx=0, pady=0)
        self.lbCue = tk.Label(self.fm_title, text='提示：点击上方按钮，选择需要识别的... ', bg='Blue', fg='White',justify='left', anchor='w')
        self.lbCue.place(x=0, y=0, width=200, height=50)

        self.__action_read_frame()
        # self.btn_prev_frame1 = tk.Button(self.fm_status, text='Start1', command=self.__action_read_frame)
        # self.btn_prev_frame1.grid(row=0, column=0, padx=10, pady=2)
        #
        # self.btn_next_frame3 = tk.Button(self.fm_status, text='Start2', command=None)
        # self.btn_next_frame3.grid(row=1, column=0, padx=10, pady=20)
    def btn_click(self,pos):
        self.btnPos =pos
        if pos ==3:
            self.lbBrief['text'] = '颜色检测：对颜色空间的图像进行有效处理，在HSV空间进行，然后对于基本色中对应的HSV分量需要给定一个严格的范围，该实验是通过计算的模糊范围'
        elif pos==2:
            self.lbBrief['text'] = '图像边缘信息主要集中在高频段，通常说图像锐化或检测边缘，实质就是高频滤波。我们知道微分运算是求信号的变化率，具有加强高频分量的作用。在空域运算中来说，对图像的锐化就是计算微分。由于数字图像的离散信号，微分运算就变成计算差分或梯度。图像处理中有多种边缘检测（梯度）算子，常用的包括普通一阶差分，Robert算子（交叉差分），Sobel算子等等，是基于寻找梯度强度。拉普拉斯算子（二阶差分）是基于过零点检测。通过计算梯度，设置阀值，得到边缘图像。 '
        elif pos == 9:
            self.lbBrief['text'] = ''

        elif pos == 8:
            self.lbBrief['text']='物体识别：做物体检测的网络有很多种，如faster rcnn，ssd，yolo等等，通过不同维度的对比，各个网络都有各自的优势。毕竟树莓派计算能力有限，我们这里先选择专门为速度优化过最快的网络SSD。'
        elif pos == 4:
            self.lbBrief['text'] = '圆形检测：基于python使用opencv实现在一张图片中检测出圆形，通过霍夫变换函数实现该效果 '
        elif pos == 7:
            self.lbBrief['text'] = '涂鸦检测：神经网络能学会辨识随手画的灵魂涂鸦吗？只要数据够多就可以！该实验通过caffe框架，对数据进行训练，通过手写涂鸦进行识别内容。 '
        elif pos == 5:
            self.lbBrief['text'] = '物体跟踪：追踪红颜色物体，并画出轮廓和运动轨迹。 '
        elif pos == 1:
            self.lbBrief['text']='二值转换：图像二值化就是将图像上的像素点的灰度值设置为0或255，也就是将整个图像呈现出明显的黑白效果的过程'
        elif pos == 0:
            self.lbBrief['text']='灰度转换：灰度化处理就是将一幅色彩图像转化为灰度图像的过程。 '
        elif pos == 6:
            self.lbBrief['text']='人脸识别：opencv2中人脸检测使用的是 detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用.'
    #medio broad
    def _braod_(self,text):
        global isBroad
        # if self.currentBroadValue <> text:
        print 'isBorading'
        cmd = 'ilang  ' + text
        os.system(cmd)
        isBroad = False
    #end medio broad

    def __action_read_frame(self):
        self.from_video()

    def from_video(self):
        global isBroad,isCheckTarget,isCheckScraw
        # cap = cv2.VideoCapture(0)
        path = r'/root/opencv/opencv-3.2.0/data/haarcascades/'
        detector = cv2.CascadeClassifier(path + 'haarcascade_frontalface_default.xml')
        eye_detector = cv2.CascadeClassifier(path + 'haarcascade_eye.xml')
        with picamera.PiCamera() as camera:
            camera.resolution = (320, 240)
            sleep(1)
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
                    writer = tf.summary.FileWriter("logs/", sess.graph)
                    sess.run(tf.global_variables_initializer())
                    loader = tf.train.import_meta_graph(model_path + '.meta')
                    loader.restore(sess, model_path)
                    print 'loader ok'
                    with picamera.array.PiRGBArray(camera) as output:
                        for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                            # frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                            frame = output.array
                            #cv2.imshow('1',frame)
                            # frame = output.array
                            # frame = frame.trans(frame)
                            frame = cv2.flip(frame,1)
                            #cv2.imshow('2',frame)
                            if self.btnPos == 1:
                                # frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                                self.lbStatus['text']='提示：当前正在二值转换... '

                            elif self.btnPos == 0: #huidu
                                # frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
                                self.lbStatus['text'] = '提示：当前正在灰度转换...'

                            elif self.btnPos == 3:
                                self.lbStatus['text'] = '提示：当前正在检测颜色...'
                                self.check_color(frame)

                            elif self.btnPos == 2:#bianyuan
                                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                                ret, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)  # binary
                                cv2.bitwise_not(frame, frame)
                                frame = cv2.Canny(frame, 50, 150)

                            elif self.btnPos == 5:
                                frame = self.color_sports(frame)

                            elif self.btnPos==8:
                                # print 'wuti1'
                                if isCheckTarget == False:
                                    self.lbStatus['text'] = '提示：当前正在物体识别...'
                                    self.check_target(frame,sess)

                            elif self.btnPos == 4:#yuanxin
                                self.lbStatus['text']='提示：当前正在检测圆形...'
                                self.check_shape(frame)

                            elif self.btnPos == 7:#tuya
                                ROI_ratio = 0.2
                                sz = frame.shape
                                cx = sz[0] / 2
                                cy = sz[1] / 2
                                ROI = int(sz[0] * ROI_ratio)
                                if isCheckScraw == False and isBroad == False:
                                    self.lbStatus['text']='提示：当前正在识别涂鸦...'
                                    self.check_scraw(frame)
                                # print '3'
                                cv2.rectangle(frame, (cy - ROI, cx - ROI), (cy + ROI, cx + ROI),
                                              (255, 255, 0), 2)
                            # print 'add'
                            elif self.btnPos == 9:
                                # self.btnPos = -1
                                self.lbStatus['text']='提示：没有需要识别的...'

                            elif self.btnPos == 6:#renlian
                                self.lbStatus['text']='提示：当前正在人脸检测...'
                                # frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                                faces = detector.detectMultiScale(gray, 1.3, 5)
                                for (x, y, w, h) in faces:
                                    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                    roi_gray = gray[y:y + h, x:x + w]
                                    roi_color = img[y:y + h, x:x + w]
                                    eyes = eye_detector.detectMultiScale(roi_gray)
                                    for (ex, ey, ew, eh) in eyes:
                                        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                            self.canvas.add(frame)
                            # print frame.shape
                            self.window.update_idletasks()
                            self.window.update()
                            # cv2.imshow('gra', frame)
                            cv2.waitKey(1)
                            output.truncate(0)
                    cv2.destroyAllWindows()
        # idx = 0
        # while True:
        #     ret, frame = cap.read()
        #     # print(frame)
        #     # img = cv2.transpose(frame)
        #     # img = cv2.flip(img, 1)
        #     # print(img.shape)

    def color_sports(self,frame):
        global counter, redLower, redUpper, pts
        # print '1'
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 根据阈值构建掩膜
        mask = cv2.inRange(hsv, redLower, redUpper)
        # 腐蚀操作
        mask = cv2.erode(mask, None, iterations=2)
        # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
        mask = cv2.dilate(mask, None, iterations=2)
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # 初始化瓶盖圆形轮廓质心

        # print '2'
        # 如果存在轮廓
        if len(cnts) > 0:
            # print '3'
            # 找到面积最大的轮廓
            c = max(cnts, key=cv2.contourArea)
            # 确定面积最大的轮廓的外接圆
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            # 计算轮廓的矩
            M = cv2.moments(c)
            # 计算质心
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # 只有当半径大于10时，才执行画图
            # print '4'
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (255, 0, 0), -1)
                # 把质心添加到pts中，并且是添加到列表左侧
                pts.appendleft(center)
        else:  # 如果图像中没有检测到瓶盖，则清空pts，图像上不显示轨迹。
            pts.clear()

        for i in xrange(1, len(pts)):
            if pts[i - 1] is None or pts[i] is None:
                continue
            # 计算所画小线段的粗细
            thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
            # 画出小线段
            cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), thickness)
            # 判断移动方向
            if counter >= 10 and i == 1 and len(pts) >= 10:
                # print('123')
                dX = pts[-10][0] - pts[i][0]
                dY = pts[-10][1] - pts[i][1]
                (dirX, dirY) = ("", "")
                #
                #     if np.abs(dX) > 20:
                #         dirX = "East" if np.sign(dX) == 1 else "West"
                #
                #     if np.abs(dY) > 20:
                #         dirY = "North" if np.sign(dY) == 1 else "South"
                #
                #     if dirX != "" and dirY != "":
                #         direction = "{}-{}".format(dirY, dirX)
                #     else:
                #         direction = dirX if dirX != "" else dirY
                #
                #     cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                #                 (0, 255, 0), 3)
                cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY), (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                # print('1234')
        counter += 1
        return frame

    # color check
    def check_color(self,frame):
        global isBroad
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        red_marker = self.find_Color(img, red_low, red_hig)
        green_marker = self.find_Color(img, green_low, green_high)
        if red_marker <> 0:
            box = cv2.boxPoints(red_marker)
            box = np.int0(box)
            cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
            self.lbValue['text']='红'
            if not isBroad:
                isBroad = True
                thread.start_new_thread(self._braod_, ('红',))

        else:
            if green_marker <> 0:
                box = cv2.boxPoints(green_marker)
                box = np.int0(box)
                cv2.drawContours(frame, [box], -1, (0, 255, 0), 2)
                self.lbValue['text'] = '绿'
                if not isBroad :
                    isBroad = True
                    thread.start_new_thread(self._braod_, ('绿',))


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
    #end check color

    #check shape
    def check_shape(self,frame):
        global isBroad
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # gray
        gray = cv2.GaussianBlur(gray, (7, 5), 0)
        ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # binary
        cv2.bitwise_not(thresh, thresh)
        image = cv2.Canny(thresh, 50, 150)
        # print 'shape'+str(dst.shape)
        # cv2.imshow('none', image)
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=20,
                                   maxRadius=120)  # huofu
        if circles is not None:
            for circle in circles[0]:
                x = int(circle[0])
                y = int(circle[1])
                r = int(circle[2])
                frame = cv2.circle(frame, (x, y), r, (255, 0, 0), -1)
            if  isBroad==False:
                isBroad = True
                # print '圆形'
                # self.currentBroadValue = '圆形'
                self.lbValue['text']='圆形'
                print '1'
                thread.start_new_thread(self._braod_, ('圆形',))
    #end check_shape


    #scraw check
    def check_scraw(self,input_image):
        global  isBroad,isCheckScraw
        isCheckScraw = True
        p1 = 120
        p2 = 45
        ROI_ratio = 0.2
        # stage = 0
        # ret_val, input_image = cap.read()
        sz = input_image.shape
        cx = sz[0] / 2
        cy = sz[1] / 2
        ROI = int(sz[0] * ROI_ratio)
        edges = cv2.Canny(input_image, p1, p2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        # print(edges.shape)
        cropped = edges[cx - ROI:cx + ROI, cy - ROI:cy + ROI, :]

        kernel = np.ones((4, 4), np.uint8)
        cropped = cv2.dilate(cropped, kernel, iterations=1)
        cropped = cv2.resize(cropped, (28, 28)) / 127.5 - 1

        img_caffe = np.array([cropped]).transpose(0, 3, 1, 2)
        in_ = net.inputs[0]
        net.forward_all(**{in_: img_caffe})
        res = net.blobs['softmax'].data[0].copy()
        res_label = np.argsort(res)[::-1][:1]
        print('*******************')
        # chn = ''.join([i for i in cls[stage][:-1] if not i.isdigit()])
        # print('Draw %010s %s Stage:[%d]' % (chn, class_list_eng[stage].split(' ')[0], stage + 1))
        # print('*******************')
        for label in res_label:
            chn = ''.join([i for i in cls[label][:-1] if not i.isdigit()])
            self.lbValue['text'] = ('%s %s - %2.2f' % (chn, class_list_eng[label].split(' ')[0], res[label]))
            str = ('我猜你画的是%s' % chn)
            print str
            # if label == stage:
            # print('Congratulations! Stage pass [%d]' % label)
                # stage += 1
            if  isBroad==False:
                isBroad = True
                thread.start_new_thread(self._braod_, (str,))

        print('*******************')
        # cv2.rectangle(input_image, (cy - ROI, cx - ROI), (cy + ROI, cx + ROI), (255, 255, 0), 5)
        isCheckScraw = False
        # cv2.imshow('ret', input_image)
        # cv2.imshow('ret2', cropped)
        # key = cv2.waitKey(1)
        # if key == ord('w'):
        #     p1 += 5
        # elif key == ord('s'):
        #     p1 -= 5
        # elif key == ord('e'):
        #     p2 += 5
        # elif key == ord('d'):
        #     p2 -= 5
        # elif key == ord('r'):
        #     ROI_ratio += 0.1
        # elif key == ord('f'):
        #     ROI_ratio -= 0.1
        # print([p1, p2])
        # isCheckScraw = True



    #end scraw



    #check target
    def check_target(self,image_np, sess):
        global isCheckTarget,isBroad
        isCheckTarget = True
        start = time.clock()
        print '*****************'
        print start
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=6)
        # print np.squeeze(classes).astype(np.int32)
        # print np.squeeze(scores)
        # print num_detections
        value=''
        for i in range(num_detections[0]):
            data =  category_index[np.squeeze(classes).astype(np.int32)[i]]

            print np.squeeze(scores)[i]
            value += str(data['name']) +','+str(np.squeeze(scores)[i])+';'
        self.lbValue['text'] = value
        # if isBroad == False and cmp(value,'') <> 0:
        #     isBroad = True
        #     thread.start_new_thread(self._braod_, (value,))

        end = time.clock()
        print end
        # cv2.imshow("checkOver", image_np)
        print 'One frame detect take time:', end - start
        print '----------------------------------'
        isCheckTarget = False



    def launch(self):
        self.window.mainloop()

if __name__ == '__main__':
    ext = Extractor_GUI()
    ext.launch()
    # ext.from_video()
