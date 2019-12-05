# -*- coding: utf-8 -*-
# import the necessary packages
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import picamera
import  picamera.array
import  cv2
from time import  sleep


def Camera_Init():
    with picamera.PiCamera() as camera:
        camera.resolution = (320, 240)
        sleep(1)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        with picamera.array.PiRGBArray(camera) as output:
            for foo in camera.capture_continuous(output, 'rgb', use_video_port=True):
                frame = cv2.cvtColor(output.array, cv2.COLOR_RGB2BGR)
                # construct the argument parse and parse the arguments
                # initialize the HOG descriptor/person detector
                # loop over the image paths
                # load the image and resize it to (1) reduce detection
                # and (2) improve detection accuracy

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

                # draw the final bounding boxes
                for (x, y, w, h) in pick:
                    cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

                # show some information on the number of bounding boxes
                # filename = imagePath[imagePath.rfind("/") + 1:]
                # print("[INFO] {}: {} original boxes, {} after suppression".format(
                #     filename, len(rects), len(pick)))

                # show the output images
                # cv2.imshow("Before NMS", orig)
                cv2.imshow("After NMS", image)

                # cv2.waitKey(0)
                cv2.waitKey(1)
                output.truncate(0)
    cv2.destroyAllWindows()






if __name__=='__main__':
    Camera_Init()