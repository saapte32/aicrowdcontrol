import cv2
import os
import numpy as np
# from numpy import expand_dims
from keras.models import load_model
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from matplotlib import pyplot
# from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import time
import matplotlib.pyplot as plt
import pandas as pd
from Func1 import dist
from Func1 import load_image_pixels
from Func1 import decode_netout
from Func1 import correct_yolo_boxes
from Func1 import do_nms
from Func1 import get_boxes
from Func1 import draw_box1
from Func1 import draw_line
from Func1 import predictimg

# load yolov3 model
model = load_model(r'model.h5')
# define the expected input shape for the model
input_w, input_h = 256, 256
# define our new photo
# photo_filename = '/content/4people.jpg'
# load and prepare image



cap = cv2.VideoCapture(r'socialchina.mp4')
cap.set(cv2.CAP_PROP_FPS, 2)
centers = list()
f = 0
while cap.isOpened():

    full = time.time()
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame = frame.resize((input_w, input_h))
    s = time.time()
    image, image_w, image_h = load_image_pixels(frame, (input_w, input_h))
    e = time.time()
    print("\n-------------------")
    print('Load_image_pixels: ', e - s)
    print("-------------------\n")
    # make prediction
    start = time.time()
    yhat = model.predict(image)  # 1.2sec
    end = time.time()
    print("\n-------------------")
    print('Person predict: ', end - start)
    print("-------------------\n")

    # summarize the shape of the list of arrays
    #     print([a.shape for a in yhat])
    print("New Frame --------------------")
    # define the anchors
    anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
    # define the probability threshold for detected objects
    class_threshold = 0.4
    boxes = list()
    s1 = time.time()
    for i in range(len(yhat)):
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    e1 = time.time()
    print("\n-------------------")
    print('deode and correctyolo: ', e1 - s1)
    print("-------------------\n")
    # suppress non-maximal boxes
    s2 = time.time()
    do_nms(boxes, 0.5)
    e2 = time.time()
    print("\n-------------------")
    print('do_nms: ', e2 - s2)
    print("-------------------\n")

    labels = ["person"]

    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    frame1, centers, centers1 = draw_box1(frame, v_boxes, v_labels, v_scores)  # 0.8 sec

    s3 = time.time()
    act_dist = dist(centers)
    e3 = time.time()
    print("\n-------------------")
    print('dist: ', e3 - s3)
    print("-------------------\n")
    image = draw_line(predictimg(frame1), centers1, act_dist)
    image = draw_line(frame1, centers1, act_dist)  # 0.7 sec
    # end = time.time()
    # print("\n-------------------")
    # print(end - start)
    # print("-------------------\n")
    # start1 = time.time()
    image = predictimg(image)  # 0.7 sec

    fulle = time.time()
    print("\n-------------------")
    print('Each Frame: ', fulle - full)
    print("-------------------\n")
    s4 = time.time()
    cv2.imshow("out", frame1)
    e4 = time.time()
    print("\n-------------------")
    print('imshow: ', e4 - s4)
    print("-------------------\n")
    fulle = time.time()
    print("\n-------------------")
    print('Each Frame: ', fulle - full)
    print("-------------------\n")

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# co3 = Detect()
# co3.__next__()