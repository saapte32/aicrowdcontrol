# checking availability of GPU
import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# importing required libraries
import numpy as np
import time
import cv2
import pandas as pd
import time
import sys
import os
import matplotlib.pyplot as plt
import urllib.request
import pyttsx3
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
from multiprocessing import Process
import requests
from io import BytesIO
# from moviepy.editor import *
# from moviepy.Clip import *
# from moviepy.video.VideoClip import *
from datetime import datetime
from PIL import Image

# files for mask_detection
labelsPath = "C:/Users/jenas/Desktop/darknet/data/yolo.names"
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = "C:/Users/jenas/Desktop/yolov3_custom_train_3000.weights"
configPath = "C:/Users/jenas/Desktop/darknet/cfg/yolov3_custom_train.cfg"

net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# print('eeee')
# detecting face masks
def predict(image):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]

    ln = net1.getLayerNames()
    ln = [ln[i[0] - 1] for i in net1.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net1.setInput(blob)
    layerOutputs = net1.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.15

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.3)
    mc = 0
    nmc = 0

    if len(idxs) > 0:

        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if (LABELS[classIDs[i]] == 'masked'):
                mc += 1
                color = (0, 255, 0)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)
            if (LABELS[classIDs[i]] == 'not_masked'):
                nmc += 1
                color = (0, 0, 255)
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
                text = "{}".format(LABELS[classIDs[i]])
                cv2.putText(image, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 1)

    text1 = "No. of people wearing masks: " + str(mc)
    text2 = "No. of people not wearing masks: " + str(nmc)
    streamer = datetime.now().strftime("%d/%m/%Y::%H:%M:%S")
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)
    color3 = (0, 255, 255)

    cv2.putText(image, text1, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
    cv2.putText(image, text2, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
    cv2.putText(image, streamer, (2, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color3, 2)
    return image, nmc


# files for distance detection
scale = 0.00392
classes_file = "C:/Users/jenas/Desktop/darknet/data/coco.names"
weights = "C:/Users/jenas/Desktop/yolov3.weights"
config_file = "C:/Users/jenas/Desktop/darknet/cfg/yolov3.cfg"

classes = None
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net2 = cv2.dnn.readNet(weights, config_file)


# setting up criteria for violation
def violation(act_dist, p):
    close = []
    grp = []
    grp1 = []
    for i in range(len(act_dist)):
        if act_dist[i] < 1.82:
            close.append('Unsafe')
            grp.append(p[i])
            for sublist in grp:
                for val in sublist:
                    grp1.append(val)
        else:
            close.append('Safe')

    return close, set(grp1)


# to get distance between 2 people
def dist(centers):
    d = []
    p = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dis = ((abs(centers[i][0] - centers[j][0])) ** 2 + (abs(centers[i][1] - centers[j][1])) ** 2) ** 0.5
            d.append(int(dis))
            p.append([i + 1, j + 1])

    print("Total number of people : ", len(centers))
    print("\n")
    F = 3.6
    R = 20
    act_dist = []
    ifov = (3 / F) * (10) ** -3
    for i in d:
        act = R * i * ifov
        act_dist.append(round(act, 2))

    close, grp1 = violation(act_dist, p)

    df1 = pd.DataFrame(list(zip(p, act_dist, close)), columns=['Person', 'Actual Distance(meter)', 'Status'])
    df1 = df1.loc[df1['Status'] == 'Unsafe']
    print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
    print("People responsible for group violations : {}".format(grp1))

    return act_dist, len(df1.index)


# categorizing people as safe or unsafe wrt the distance they maintain from others
def run(image):
    np.random.seed(32)
    (H, W) = image.shape[:2]

    ln = net2.getLayerNames()
    ln = [ln[i[0] - 1] for i in net2.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net2.setInput(blob)
    layerOutputs = net2.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    threshold = 0.5

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.4)
    centers = []
    c = []
    for i in idxs:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        centers.append([int(x + w / 2), int(y + h), int(w), int(h)])
        c.append(classIDs[i])
    act_dist, d_vio = dist(centers)

    # print(act_dist)

    if (len(centers) == 1):
        if (c[0] == 0):
            color = (0, 255, 0)
            cv2.rectangle(image, (centers[0][0] - int(centers[0][2] / 2), centers[0][1] - centers[0][3]),
                          (centers[0][0] + int(centers[0][2] / 2), centers[0][1]), color, 1)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if act_dist[i] > 2.00:
                color = (0, 255, 0)
                if c[i] == 0:
                    cv2.rectangle(image, (centers[i][0] - int(centers[i][2] / 2), centers[i][1] - centers[i][3]),
                                  (centers[i][0] + int(centers[i][2] / 2), centers[i][1]), color, 1)
                    # cv2.putText(image, 'safe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                    # 	0.5, color, 1)
            else:
                color = (0, 0, 255)
                if c[i] == 0:
                    cv2.rectangle(image, (centers[i][0] - int(centers[i][2] / 2), centers[i][1] - centers[i][3]),
                                  (centers[i][0] + int(centers[i][2] / 2), centers[i][1]), color, 1)
                    # cv2.putText(image, 'unsafe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                    # 	0.5, color, 1)
    color3 = (0, 255, 255)
    streamer = datetime.now().strftime("%d/%m/%Y::%H:%M:%S")
    cv2.putText(image, streamer, (2, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color3, 2)
    return image, d_vio


# mask detection for streaming
def mask_detector_img(url):
    # gauth = GoogleAuth()

    # gauth.LocalWebserverAuth()
    # drive = GoogleDrive(gauth)
    # vid1 = []
    count = 5
    alert = "Face Mask Violations detected, please wear face mask"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    a = time.time()
    while (True):
        t = time.time()
        m = 0
        f = 1
        video_name_time_FM = datetime.now().strftime('%Y%m%d%H%M')
        video_name_FM = "FM" + video_name_time_FM + ".mp4"
        out1 = cv2.VideoWriter(video_name_FM, fourcc, 3.0, (624, 416))
        with tf.device('/CPU'):
            while True:
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img = np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img1 = cv2.resize(img, (624, 416))
                op1, m_vio = predict(img1)
                if f % count == 0:
                    if (m / f) > 2:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(alert)
                        engine.runAndWait()

                    elif m_vio > 0:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(alert)
                        engine.runAndWait()
                m += m_vio
                # vid1.append(op1)

                f += 1
                # cv2.imshow(' ', op1)
                out1.write(op1)
                cv2.imwrite('test_frame_mask.jpg', op1)
                if ((time.time() - t) // 60 == 1):
                    break
                    # if cv2.waitKey(1) & 0xff == ord("q"):
                    # 	break
            # path1 = os.path.join(gpath, video_name_FM)

            # with open(path1,"r") as file:
            # 	f1 = drive.CreateFile({'title':os.path.basename(file.name)})
            # 	f1.SetContentFile(path1)
            # 	f1.Upload()
            # os.remove(path)
            avg_mask = m / f
            print("Average mask violations: ", avg_mask)
            try:
                if ((time.time() - a) // 180 == 1):
                    dir_name = "C:/Users/jenas/Desktop/Crowd_AI"
                    test = os.listdir(dir_name)

                    for item in test:
                        if item.startswith("FM"):
                            os.remove(os.path.join(dir_name, item))

                    a = time.time()
                else:
                    continue
            except:
                print('Access error')
                a = time.time()


# distance finder for streaming
def person_dist_img(url):
    # gauth = GoogleAuth()

    # gauth.LocalWebserverAuth()
    # drive = GoogleDrive(gauth)
    # vid = []
    count = 5
    alert = "Social Distancing Violations Detected, Please maintain safe distance"
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    a = time.time()
    while (True):
        s = time.time()
        d = 0
        f = 1
        video_name_time_SD = datetime.now().strftime('%Y%m%d%H%M')
        video_name_SD = "SD" + video_name_time_SD + ".mp4"
        out2 = cv2.VideoWriter(video_name_SD, fourcc, 3.0, (624, 416))
        with tf.device('/CPU'):
            while True:
                # s = time.time()
                response = requests.get(url)
                img = Image.open(BytesIO(response.content))
                img = np.asarray(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img2 = cv2.resize(img, (624, 416))
                op2, d_vio = run(img2)
                if f % count == 0:
                    if (d / f) > 10:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(alert)
                        engine.runAndWait()

                    elif d_vio > 4:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 150)
                        engine.say(alert)
                        engine.runAndWait()
                d += d_vio
                # vid.append(op2)
                f += 1
                # cv2.imshow(' ', op2)
                out2.write(op2)
                cv2.imwrite('test_frame_distance.jpg', op2)
                print(time.time() - s)
                if ((time.time() - s) // 60 == 1):
                    break

            try:
                if ((time.time() - a) // 180 == 1):
                    dir_name = "C:/Users/jenas/Desktop/Crowd_AI"
                    test = os.listdir(dir_name)

                    for item in test:
                        if item.startswith("SD"):
                            os.remove(os.path.join(dir_name, item))

                    a = time.time()
                else:
                    continue
            except:
                print('Access error')
                a = time.time()

                # path2 = os.path.join(gpath, video_name_SD)

                # with open(path2,"r") as file:
                # 	f2 = drive.CreateFile({'title':os.path.basename(file.name)})
                # 	f2.SetContentFile(path2)
                # 	f2.Upload()
                # os.remove(path)
                # if cv2.waitKey(1) & 0xff == ord("q"):
                # 	break
                # print('------------------------------------------', len(vid))
                # videoclip = ImageSequenceClip(vid, fps=10)
                # videoclip.write_videofile(video_name_SD, fps=10)


# importing some more libraries
# from PIL import Image

# url to be given for live streaming
url = 'http://192.168.43.242:8080/shot.jpg'
# gpath = 'C:/Users/jenas/Desktop'

# name of saved video file
video = 'test1.mp4'

if __name__ == '__main__':
    # gauth = GoogleAuth()
    # gauth.LocalWebserverAuth()

    # drive = GoogleDrive(gauth)

    # # block for streaming (activate as required)
    p1 = Process(target=mask_detector_img, args=(url,))
    p2 = Process(target=person_dist_img, args=(url,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()