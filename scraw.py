# coding: utf-8
import pygame, random
import cv2, numpy as np, sys, pdb

sys.path.append('/root/caffe/python')
import caffe

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
cap = cv2.VideoCapture(0)
p1 = 120
p2 = 45
ROI_ratio = 0.2
stage = 0
while 1:
    ret_val, input_image = cap.read()
    sz = input_image.shape
    cx = sz[0] / 2
    cy = sz[1] / 2
    ROI = int(sz[0] * ROI_ratio)
    edges = cv2.Canny(input_image, p1, p2)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    print(edges.shape)
    cropped = edges[cx - ROI:cx + ROI, cy - ROI:cy + ROI, :]

    kernel = np.ones((4, 4), np.uint8)
    cropped = cv2.dilate(cropped, kernel, iterations=1)
    cropped = cv2.resize(cropped, (28, 28)) / 127.5 - 1

    img_caffe = np.array([cropped]).transpose(0, 3, 1, 2)
    in_ = net.inputs[0]
    net.forward_all(**{in_: img_caffe})
    res = net.blobs['softmax'].data[0].copy()
    res_label = np.argsort(res)[::-1][:5]
    print('*******************')
    chn = ''.join([i for i in cls[stage][:-1] if not i.isdigit()])
    print('Draw %010s %s Stage:[%d]' % (chn, class_list_eng[stage].split(' ')[0], stage + 1))
    print('*******************')
    for label in res_label:
        chn = ''.join([i for i in cls[label][:-1] if not i.isdigit()])
        print('%s %s - %2.2f' % (chn, class_list_eng[label].split(' ')[0], res[label]))
        if label == stage:
            print('Congratulations! Stage pass [%d]' % stage)
            stage += 1

    cv2.rectangle(input_image, (cy - ROI, cx - ROI), (cy + ROI, cx + ROI), (255, 255, 0), 5)
    cv2.imshow('ret', input_image)
    cv2.imshow('ret2', cropped)
    key = cv2.waitKey(1)
    if key == ord('w'):
        p1 += 5
    elif key == ord('s'):
        p1 -= 5
    elif key == ord('e'):
        p2 += 5
    elif key == ord('d'):
        p2 -= 5
    elif key == ord('r'):
        ROI_ratio += 0.1
    elif key == ord('f'):
        ROI_ratio -= 0.1
    print([p1, p2])