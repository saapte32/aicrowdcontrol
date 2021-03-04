# checking availability of GPU (optional)
import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import pyttsx3
# importing required libraries
import numpy as np
import time
import cv2
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
import urllib.request
from moviepy.editor import *
from moviepy.Clip import *
from moviepy.video.VideoClip import *
from PIL import Image

# files for mask_detection
base_file_path = os.path.dirname(os.path.abspath(__file__))
labelsPath = os.path.join(base_file_path,"yolo.names")
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath = os.path.join(base_file_path,"yolov3_custom_train_3000.weights")
configPath = os.path.join(base_file_path,"yolov3_custom_train.cfg")

net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# detecting face masks
def predict(image):
    np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
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

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
    # mc = 0
    nmc = 0

    if len(idxs) > 0:

        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # if(LABELS[classIDs[i]]=='masked'):
            # 	mc+=1
            # 	color = (0,255,0)
            # 	cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            # 	text = "{}".format(LABELS[classIDs[i]])
            # 	cv2.putText(image, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
            # 		0.5, color, 1)
            if (LABELS[classIDs[i]] == 'not_masked'):
                nmc += 1
    # color = (0,0,255)
    # 			cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
    # 			text = "{}".format(LABELS[classIDs[i]])
    # 			cv2.putText(image, text, (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
    # 				0.5, color, 1)

    # text1 = "No. of people wearing masks: " + str(mc)
    # text2 = "No. of people not wearing masks: " + str(nmc)
    # color1 = (0,255,0)
    # color2 = (0,0,255)

    # cv2.putText(image, text1, (2,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
    # cv2.putText(image, text2, (2,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
    return nmc


# files for distance detection
scale = 0.00392
classes_file = "D:\code\crowdfacemask\crowdai\detect\coco.names"
weights = "D:\code\crowdfacemask\crowdai\detect\yolov3.weights"
config_file = "D:\code\crowdfacemask\crowdai\detect\yolov3.cfg"

classes = None
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

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

    # print("Total number of people : ", len(centers))
    # print("\n")
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
    # print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
    # print("People responsible for group violations : {}".format(grp1))

    return len(df1.index)


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
    d_vio = dist(centers)

    # print(act_dist)

    # if (len(centers)==1):
    # 	if(c[0]==0):
    # 		color=(255,255,255)
    # 		cv2.rectangle(image, (centers[i][0] - int(centers[i][2]/2), centers[i][1] - centers[i][3]), (centers[i][0] + int(centers[i][2]/2), centers[i][1]), color, 1)

    # for i in range(len(centers)):
    # 	for j in range(i+1, len(centers)):
    # 		if act_dist[i] > 2.00:
    # 			color = (255,255,255)
    # 			if c[i]==0:
    # 				cv2.rectangle(image, (centers[i][0] - int(centers[i][2]/2), centers[i][1] - centers[i][3]), (centers[i][0] + int(centers[i][2]/2), centers[i][1]), color, 1)
    # 			# cv2.putText(image, 'safe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
    # 			# 	0.5, color, 1)
    # 		else:
    # 			color = (0,0,255)
    # 			if c[i]==0:
    # 				cv2.rectangle(image, (centers[i][0] - int(centers[i][2]/2), centers[i][1] - centers[i][3]), (centers[i][0] + int(centers[i][2]/2), centers[i][1]), color, 1)
    # 			# cv2.putText(image, 'unsafe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
    # 			# 	0.5, color, 1)

    return d_vio


# mask detection for saved videos
def mask_detector(video1):
    # with tf.device('/GPU:0'):
    # video1 = "C:/Users/jenas/Desktop/test1.mp4"
    #cap1 = cv2.VideoCapture(video1)
    cap1 = cv2.VideoCapture(0)
    c1 = -1
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('MD10.mp4', fourcc, 10.0, (624, 416))
    cap1.open(video1)
    while True:
        ret, frame = cap1.read()
        c1 += 1
        # frame = cv2.resize(frame, (624, 416))
        if c1 % 5 == 0:
            m_vio = predict(frame)
            print(m_vio, " m")
            # display(predict(frame))
            # cv2.imshow("Mask Detector", image01)
            # out.write(image01)
            # if cv2.waitKey(1) & 0xff == ord("q"):
            # 	break
    cap1.release()
    cv2.destroyAllWindows()


# distance finder for saved videos
def person_dist(video2):
    # with tf.device('/GPU:0'):
    #cap2 = cv2.VideoCapture(video2)
    cap2 = cv2.VideoCapture(0)
    c2 = -1
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out = cv2.VideoWriter('SD10.mp4', fourcc, 10.0, (624, 416))
    cap2.open(video2)
    while True:
        # s = time.time()
        ret, frame = cap2.read()
        c2 += 1
        # frame = cv2.resize(frame, (624, 416))
        if c2 % 5 == 0:
            d_vio = run(frame)
            print(d_vio, " d")
            # print(time.time() - s)
            # cv2.imshow("Distance Observer", image02)
            # out.write(image02)
            # if cv2.waitKey(1) & 0xff == ord("q"):
            # 	break
    cap2.release()
    cv2.destroyAllWindows()


# combine processes
# def combine(video3):
# 	with tf.device('/GPU:0'):
# 		cap3=cv2.VideoCapture(video3)
# 		fourcc = fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# 		out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (624, 416))
# 		cap3.open(video3)
# 		while True:
# 			s = time.time()
# 			ret, frame=cap3.read()
# 			frame = cv2.resize(frame, (624, 416))
# 			frame1 = run(frame)
# 			frame2 = predict(frame1)

# 			print(time.time() - s)

# 			cv2.imshow("Processed Feed", frame2)
# 			out.write(frame2)
# 			if cv2.waitKey(1) & 0xff == ord("q"):
# 				break
# 		cap3.release()
# 		cv2.destroyAllWindows()

# mask detection for streaming
def mask_detector_img(url):
    m = 0
    f = 1
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # # video_name_time_FM = datetime.now().strftime('%Y-%m-%d%H:%M:%S')
    # # vt_fm_name = time.mktime(datetime.strptime(video_name_time_FM, "%Y-%m-%d%H:%M:%S").timetuple())
    # # video_name_FM = "FM" + str(vt_fm_name) + ".mp4"
    # # base_file_path = os.path.dirname(os.path.abspath(__file__))
    # # FM_fullpath = os.path.join(base_file_path,video_name_FM)
    # out_FM = cv2.VideoWriter('mask.mp4' , fourcc, 10.0, (624, 416))
    count = 5
    alert = "Face Mask Violations detected, please wear face mask"
    with tf.device('/CPU'):
        while True:
            # print("m")
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            # if img.verify()==None:
            # 	break
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (624, 416))
            m_vio = predict(img)
            if f % count == 0:
                if (m / f) > 2:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()

                elif m_vio > 1:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()
            m += m_vio
            # cv2.imshow("Mask Detector", frame)

            f += 1
            # out_FM.write(frame)
            # cv2.imwrite('test_frame_mask.jpg', frame)
            # cv2.imwrite('test_frame_mask2.jpg', frame)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        avg_mask = m / f
        print("Average mask violations: ", avg_mask)


# distance finder for streaming
def person_dist_img(url):
    d = 0
    f = 1
    # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # out_FM = cv2.VideoWriter('distance.mp4' , fourcc, 10.0, (624, 416))
    count = 5
    alert = "Social Distance Violations Detected, Please maintain safe distance"
    with tf.device('/CPU'):
        while True:
            # print("d")
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            # if img.verify()==None:
            # 	break
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # img = cv2.resize(img, (624, 416))
            d_vio = run(img)
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
            # cv2.imshow("Distance Observer", frame)
            # f += 1
            # out_FM.write(frame)
            # cv2.imwrite('test_frame_distance.jpg', frame)
            # cv2.imwrite('test_frame_distance2.jpg', frame)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
                # avg_dist = d / f


# combine processes for image
# def combine_img(url):
# 	with tf.device('/GPU:0'):
# 		while True:
# 			response = requests.get(url)
# 			img = Image.open(BytesIO(response.content))
# 			img = np.asarray(img)
# 			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 			img = cv2.resize(img, (624, 416))
# 			img1 = run(img)
# 			cv2.imshow("Processed Feed", predict(img))

# 			if cv2.waitKey(1) & 0xff == ord("q"):
# 				break


# importing some more libraries
from multiprocessing import Process
import requests
from io import BytesIO
from PIL import Image

# url to be given for live streaming
url = 'http://192.168.43.97:8080/shot.jpg'

# name of saved video file
video = 'socialchina.mp4'


# if __name__ == '__main__':

# # # block for streaming (ACTIVATE AS REQD)
# p1 = Process(target=mask_detector_img, args=(url,))
# p2 = Process(target=person_dist_img, args=(url,))

# p1.start()
# p2.start()

# p1.join()
# p2.join()


# # # block for saved videos (ACTIVATE AS REQD)
# s = time.time()

# p1 = Process(target=mask_detector, args=(video,))
# p2 = Process(target=person_dist, args=(video,))

# p1.start()
# p2.start()

# p1.join()
# p2.join()

# print(time.time()-s)
# # # block for combining output in saved video (ACTIVATE AS REQD)
# combine(video)


# # # block for combining output in streaming (ACTIVATE AS REQD)
# combine_img(url)# importing required libraries
# from BothAvg import predict
# from BothAvg import run
# from BothAvg import violation
# from BothAvg import dist
# from detect.models import tbl_Incident_Master

# url = 'http://192.168.1.3:8080/shot.jpg'

def feedlive(url):
    # url = 'http://192.168.1.3:8080/shot.jpg'
    print('\nFUNCTION RUNNNNNNNNN\n')

    # # block for streaming (ACTIVATE AS REQD)
    p1 = Process(target=mask_detector_img, args=(url,))
    p2 = Process(target=person_dist_img, args=(url,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()


if __name__ == '__main__':
    feedlive(url)