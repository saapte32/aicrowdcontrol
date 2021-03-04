

from Func import dist
from Func import load_image_pixels
from Func import decode_netout
from Func import correct_yolo_boxes
from Func import do_nms
from Func import get_boxes
from Func import draw_box1
from Func import draw_line
from Func import predictimg


import cv2
import os
import numpy as np
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont
import time
import matplotlib.pyplot as plt
import pandas as pd


import asyncio


# async def personbox(frame, v_boxes, v_labels, v_scores):
#     frame1, centers, centers1 = draw_box1(frame, v_boxes, v_labels, v_scores)
#
#     return frame1,centers,centers1

#async def personline():


async def detect():
    # import pdb;pdb.set_trace()
    # load yolov3 model
    model = load_model(r'model.h5',compile=False)
    # define the expected input shape for the model
    input_w, input_h = 288, 288
    # define our new photo
    # photo_filename = '/content/4people.jpg'
    # load and prepare image
    #cap = cv2.VideoCapture(r'indmarket.mp4')
    cap = cv2.VideoCapture(r'hong.mp4')
    centers = list()
    while cap.isOpened():
        s=time.time()
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize((input_w, input_h))
        image, image_w, image_h = load_image_pixels(frame, (input_w, input_h))
        # make prediction
        yhat = model.predict(image)
        # summarize the shape of the list of arrays
        #     print([a.shape for a in yhat])
        print("New Frame --------------------")
        # define the anchors
        anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
        # define the probability threshold for detected objects
        class_threshold = 0.4
        boxes = list()
        for i in range(len(yhat)):
            boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

        correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        # suppress non-maximal boxes
        await do_nms(boxes, 0.5)
        await asyncio.sleep(0.0001)
        labels = ["person"]

        v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

        for i in range(len(v_boxes)):
            print(v_labels[i], v_scores[i])

        #personbox
       # frame1,centers,centers1=asyncio.get_running_loop().create_task(draw_box1(frame,v_boxes,v_labels,v_scores))
        frame1, centers, centers1 = await draw_box1(frame, v_boxes, v_labels, v_scores)
        #await asyncio.wait([frame1,centers,centers1])


        act_dist = dist(centers)

        frame1=await predictimg(frame1)
        #await asyncio.wait([frame1])

        image = await draw_line(frame1, centers1, act_dist)
        #await asyncio.wait([image])
        cv2.imshow("out", image)
        t=time.time()-s
        print("my t is ",t)
        #cv2.waitKey(1)

        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    try:
        event_loop=asyncio.get_event_loop()
        #asyncio.run(detect())
        event_loop.run_until_complete(detect())
       #  future = asyncio.Future()
       #  asyncio.ensure_future(detect())
       #  event_loop.run_until_complete(future)
    finally:
        event_loop.close()