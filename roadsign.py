#-*- coding:utf-8 -*-
import sys

import cv2

sys.path.append('/root/caffe/python')
import caffe
import numpy as np
import json

root = '/root/TSR/models/gtsrb/'  # 0
deploy = root + 'deploy.prototxt'  # deploy文件
caffe_model = root + 'train.caffemodel'  # 训练好的 caffemodel
# img=root+'325.jpg'    #随机找的一张待测图片
labels_filename = root + 'list.txt'  # 类别名称文件，将数字标签转换回类别名称
caffe.set_mode_cpu();
net = caffe.Net(deploy, caffe_model, caffe.TEST)  # 加载model和network
mean_file = root + 'mean.binaryproto'
blob = caffe.proto.caffe_pb2.BlobProto()
meanData = open(mean_file, 'rb').read()
blob.ParseFromString(meanData)
array = np.array(caffe.io.blobproto_to_array(blob))
mean_npy = array[0]
# 图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  # 设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2, 0, 1))  # 改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
transformer.set_mean('data', mean_npy.mean(1).mean(1))  # 减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 48)  # 缩放到【0，255】之间
# transformer.set_channel_swap('data', (2, 1, 0))  # 交换通道，将图片由RGB变为BGR

def  getPre(im,programname):
    global  net,transformer
    # im=caffe.io.load_image(img)                   #加载图片
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中
    #执行测试

    out = net.forward()
    labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
    prob= net.blobs['prob'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
    print prob
    order=prob.argsort()[-1] #将概率值排序，取出最大值所在的序号
    print order
    print 'the class is:',labels[order]   #4该序号转换成对应的类别名称，并打印
    # returnData = {'name':labels[order],'pre':prob[order]}
    data = {'pre':float(prob[order]),'name':str(labels[order])}

    # print(data)
    # if float(prob[order]) > 0.8:
    print (data)
    return data
    # else :
    #     return {'maybepre':float(prob[order]),'name':str(labels[order])}
    # return json.dumps(data,ensure_ascii=False)
    # return data

if __name__ == '__main__':


    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        try:
            ret, frame = cap.read()

            gray = cv2.GaussianBlur(frame, (7, 5), 0)
            ret, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)  # binary
            cv2.bitwise_not(thresh, thresh)

            image = cv2.Canny(thresh, 50, 150)
            # print 'shape'+str(dst.shape)
            # cv2.imshow('none', image)
            circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=40,
                                       maxRadius=60)  # huofu
            if circles is not None:
                for circle in circles[0]:
                    x = int(circle[0])
                    y = int(circle[1])
                    r = int(circle[2])
                    # print x,y,r
                    dst = cv2.rectangle(frame, (x - r , y - r ), (x + r , y + r ), (0, 0, 255), 1, 4)
                    # cv2.imshow('before', frame)
                    framexy = frame[y-r:y + r ,x - r :x + r ] #FRAME[Y0:Y1,X0:X1]
                    # cv2.imshow('after', framexy)
                    data = getPre(framexy, '')


            cv2.imshow('cv', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except:
            print 'error'
            break


