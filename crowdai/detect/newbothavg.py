# #checking availability of GPU (optional)
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
from datetime import datetime
import pymysql
import matplotlib.pyplot as plt
import urllib.request
from PIL import Image
import requests
from io import BytesIO
from PIL import Image
from moviepy.editor import *
from moviepy.Clip import *
from moviepy.video.VideoClip import *
from detect.models import tbl_Incident_Master
from detect.models import LocationMaster
from detect.models import CameraMaster


# files for mask_detection


# detecting face masks
def predict(image):
    base_file_path = os.path.dirname(os.path.abspath(__file__))
    labelsPath = os.path.join(base_file_path, "yolo.names")
    LABELS = open(labelsPath).read().strip().split("\n")

    weightsPath = os.path.join(base_file_path, "yolov3_custom_train_3000.weights")
    configPath = os.path.join(base_file_path, "yolov3_custom_train.cfg")

    net1 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

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

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
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
    color1 = (0, 255, 0)
    color2 = (0, 0, 255)

    cv2.putText(image, text1, (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color1, 2)
    cv2.putText(image, text2, (2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)
    return image, nmc


# files for distance detection


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
def dist(centers, F, P, R):
    d = []
    p = []
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dis = ((abs(centers[i][0] - centers[j][0])) ** 2 + (abs(centers[i][1] - centers[j][1])) ** 2) ** 0.5
            d.append(int(dis))
            p.append([i + 1, j + 1])

    # print("Total number of people : ", len(centers))
    # print("\n")
    # F = 3.6
    # R = 20
    # P = 3
    act_dist = []
    ifov = (P / F) * (10) ** -3
    for i in d:
        act = R * i * ifov
        act_dist.append(round(act, 2))

    close, grp1 = violation(act_dist, p)

    df1 = pd.DataFrame(list(zip(p, act_dist, close)), columns=['Person', 'Actual Distance(meter)', 'Status'])
    df1 = df1.loc[df1['Status'] == 'Unsafe']
    # print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
    # print("People responsible for group violations : {}".format(grp1))

    return act_dist, len(df1.index)


# categorizing people as safe or unsafe wrt the distance they maintain from others
def run(image, F, P, R):
    scale = 0.00392
    base_file_path = os.path.dirname(os.path.abspath(__file__))

    classes_file = os.path.join(base_file_path, "coco.names")
    weights = os.path.join(base_file_path, "yolov3.weights")
    config_file = os.path.join(base_file_path, "yolov3.cfg")

    classes = None
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net2 = cv2.dnn.readNet(weights, config_file)
    # import pdb;pdb.set_trace()
    np.random.seed(32)
    try:
        (H, W) = image.shape[:2]
    except:
        pass

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
    act_dist, d_vio = dist(centers, F, P, R)

    # print(act_dist)

    if (len(centers) == 1):
        if (c[0] == 0):
            color = (0, 255, 0)
            cv2.rectangle(image, (centers[i][0] - int(centers[i][2] / 2), centers[i][1] - centers[i][3]),
                          (centers[i][0] + int(centers[i][2] / 2), centers[i][1]), color, 2)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            if act_dist[i] > 2.00:
                color = (0, 255, 0)
                if c[i] == 0:
                    cv2.rectangle(image, (centers[i][0] - int(centers[i][2] / 2), centers[i][1] - centers[i][3]),
                                  (centers[i][0] + int(centers[i][2] / 2), centers[i][1]), color, 2)
            # cv2.putText(image, 'safe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
            #   0.5, color, 1)
            else:
                color = (0, 0, 255)
                if c[i] == 0:
                    cv2.rectangle(image, (centers[i][0] - int(centers[i][2] / 2), centers[i][1] - centers[i][3]),
                                  (centers[i][0] + int(centers[i][2] / 2), centers[i][1]), color, 2)
                    # cv2.putText(image, 'unsafe', (x + w, y + h), cv2.FONT_HERSHEY_SIMPLEX,
                    #   0.5, color, 1)

    return image, d_vio


# mask detection for saved videos
def mask_detector(video1, fps):
    with tf.device('/CPU'):
        vid = []
        # video1 = "C:/Users/jenas/Desktop/test1.mp4"
        cap1 = cv2.VideoCapture(video1)
        cap1.open(video1)
        f = 1
        m = 0
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_name_time_FM = datetime.now().strftime('%Y%m%d%H%M')
        # vt_fm_name = time.mktime(datetime.strptime(video_name_time_FM, "%Y%m%d%H%M").timetuple())
        video_name_FM = "detect/FM" + video_name_time_FM + ".mp4"
        # out_FM = cv2.VideoWriter(video_name_FM, -1, fps, (int(cap1.get(3)),int(cap1.get(4))))

        # out = cv2.VideoWriter('output.avi', -1, 20.0, (624,416))
        count = 9
        alert = "Face Mask Violations detected, please wear face mask"
        while True:
            ret, frame = cap1.read()
            # frame = cv2.resize(frame, (624, 416))
            if ret == False:
                break
            image01, m_vio = predict(frame)

            # number of mask Violations
            m += m_vio
            if f % count == 0:
                if m > 2:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()

                elif m_vio > 1:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()

            # display(predict(frame))
            # cv2.imshow("Mask Detector", image01)
            image01 = cv2.cvtColor(image01, cv2.COLOR_BGR2RGB)
            vid.append(image01)
            # out_FM.write(image01)
            f += 1
            # if cv2.waitKey(1) & 0xff == ord("q"):
            #     break
        avg_mask = m / f
        videoclip = ImageSequenceClip(vid, fps=20)
        videoclip.write_videofile(video_name_FM, fps=20)

        # print("Average mask violations: ", avg_mask)
        cap1.release()
        cv2.destroyAllWindows()
        return round(avg_mask), video_name_FM


# distance finder for saved videos
def person_dist(video2, F, P, R, fps):
    with tf.device('/CPU'):
        vid1 = []
        cap2 = cv2.VideoCapture(video2)
        cap2.open(video2)
        d = 0
        f = 1
        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        video_name_time_SD = datetime.now().strftime('%Y%m%d%H%M')
        # vt_fm_name = time.mktime(datetime.strptime(video_name_time_FM, "%Y%m%d%H%M").timetuple())
        video_name_SD = "detect/SD" + video_name_time_SD + ".mp4"
        # video_name_SD = "SD"+ str(video_name_time_SD)
        # out_SD = cv2.VideoWriter(str(video_name_SD)+".mp4", fourcc, 30.0, (624, 416))
        # out = cv2.VideoWriter(video_name_SD, -1, fps, (int(cap2.get(3)),int(cap2.get(4))))
        count = 9
        alert = "Social Distance Violations Detected, Please maintain safe distance"
        while True:
            s = time.time()
            ret, frame = cap2.read()
            if ret == False:
                break

            # print("frame",frame)
            # frame = cv2.resize(frame, (624, 416))
            image02, d_vio = run(frame, F, P, R)
            # print(time.time() - s)

            d += d_vio
            if f % count == 0:
                if d > 10:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()

                elif d_vio > 3:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 150)
                    engine.say(alert)
                    engine.runAndWait()
            image02 = cv2.cvtColor(image02, cv2.COLOR_BGR2RGB)
            vid1.append(image02)
            # cv2.imshow("Distance Observer", image02)
            # out.write(image02)
            f += 1
            # if cv2.waitKey(1) & 0xff == ord("q"):
            #     break

        # print("Frrrrrraaammmmm", f)
        avg_dist = d / f
        # print("Average distance violations: ", avg_dist)
        videoclip1 = ImageSequenceClip(vid1, fps=20)
        videoclip1.write_videofile(video_name_SD, fps=20)
        cap2.release()
        cv2.destroyAllWindows()
        return round(avg_dist), video_name_SD


# combine processes
def combine(video3):
    df = pd.DataFrame()
    with tf.device('/CPU'):
        cap3 = cv2.VideoCapture(video3)
        cap3.open(video3)
        f = 1
        m = 0
        d = 0
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (624, 416))
        while True:
            s = time.time()
            ret, frame = cap3.read()
            frame = cv2.resize(frame, (624, 416))
            frame1, d_vio = run(frame, 3.6, 3, 10)
            frame2, m_vio = predict(frame1)
            # print(time.time() - s)

            d += d_vio
            m += m_vio
            f += 1
            #            cv2.imshow("Processed Feed", frame2)
            out.write(frame2)
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        avg_dist = d / f
        avg_mask = m / f
        df['Distance Violation'] = avg_dist
        df['Mask Violation'] = avg_mask
        # print("Average distance violations: ", avg_dist)
        # print("Average Mask violations: ", avg_mask)

        cap3.release()
        cv2.destroyAllWindows()
    return avg_dist, avg_mask


# mask detection for streaming
def mask_detector_img(url):
    m = 0
    f = 1
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_FM = cv2.VideoWriter('detect/static/assets/mask.mp4', fourcc, 20.0, (624, 416))
    with tf.device('/CPU'):
        while True:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            # if img.verify()==None:
            #   break
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (624, 416))
            frame, m_vio = predict(img)
            m += m_vio
            cv2.imshow("Mask Detector", frame)

            f += 1
            out_FM.write(frame)
            # import pdb; pdb.set_trace()
            cv2.imwrite('detect/static/assets/test_frame_mask.jpg', frame)

            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        avg_mask = m / f
        # print("Average mask violations: ", avg_mask)

        # print("Average mask violations: ", avg_mask)


def person_dist_img(url):
    d = 0
    f = 1
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_FM = cv2.VideoWriter('detect/static/assets/distance.mp4', fourcc, 20.0, (624, 416))
    with tf.device('/CPU'):
        while True:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            # if img.verify()==None:
            #   break
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (624, 416))
            frame, d_vio = run(img, 3.6, 3, 10)
            d += d_vio
            cv2.imshow("Distance Observer", frame)
            f += 1
            out_FM.write(frame)
            # import pdb; pdb.set_trace()
            cv2.imwrite('detect/static/assets/test_frame_distance.jpg', frame)

            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        avg_dist = d / f


# combine processes for image
def combine_img(url):
    d = 0
    m = 0
    f = 1
    with tf.device('/CPU'):
        while True:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (624, 416))
            img1, d_vio = run(img)
            frame, m_vio = predict(img1)

            d += d_vio
            m += m_vio
            #            cv2.imshow("Processed Feed", frame)
            f += 1
            if cv2.waitKey(1) & 0xff == ord("q"):
                break
        avg_dist = d / f
        avg_mask = m / f
        # print("Average distance violations: ", avg_dist)
        # print("Average Mask violations: ", avg_mask)


def create_datastore():
    # url to be given for live streaming
    url = 'http://192.168.43.97:8080/shot.jpg'
    base_file_path = os.path.dirname(os.path.abspath(__file__))

    # name of saved video file
    video = os.path.join(base_file_path, 'ind1.mp4')
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(start_time)
    # p1 = Process(target=mask_detector, args=(video,))
    #
    # p2 = Process(target=person_dist, args=(video,))
    # print("###########$$$$$$$$$$$$$$$$$$started$$$$$$$$$$$$$$$###############")
    # import pdb;pdb.set_trace()
    # field = 'camid'
    id1 = 0
    while id1 <= LocationMaster.objects.count():

        F = 3.6
        P = 3
        R = 10
        fps = 20.0

        obj = LocationMaster.objects.filter(LocationId=id1)
        # print('-----------', obj)

        if obj:

            f = LocationMaster._meta.get_field('camid')
            vf = f.value_from_object(obj[0])
            F = CameraMaster.objects.get(Name=vf).Focal if CameraMaster.objects.get(Name=vf).Focal else 3.6
            P = CameraMaster.objects.get(Name=vf).Pixel if CameraMaster.objects.get(Name=vf).Pixel else 3
            R = CameraMaster.objects.get(Name=vf).Range if CameraMaster.objects.get(Name=vf).Range else 10
            fps = CameraMaster.objects.get(Name=vf).FPS if CameraMaster.objects.get(Name=vf).FPS   else 20.0

            # print(F)

            # print(P)
            # print(R)
            # print(fps)


            act_dist, video_name_SD = person_dist(video, F, P, R, fps)

            act_mask, video_m_name = mask_detector(video, fps)

            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            CameraId = id1
            store_to_db(CameraId, start_time, end_time, video_name_SD, video_m_name, act_dist, act_mask)

        else:
            pass

            # F = 3.6
            # P = 3
            # R = 10
            # fps = 20.0
            # # print(F)

            # # print(P)
            # # print(R)
            # # print(fps)
            # act_dist, video_name_SD = person_dist(video, F, P, R, fps)



            # act_mask, video_m_name = mask_detector(video, fps)

            # end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # CameraId = id1
            # store_to_db(CameraId, start_time, end_time, video_name_SD,video_m_name, act_dist, act_mask)

        id1 += 1


def store_to_db(CameraId, start_time, end_time, video_name_SD, video_m_name, act_dist, act_mask):
    tbl_Incident_Master.objects.create(CameraId=CameraId, Date_Time_frm=start_time, Date_Time_to=end_time,
                                       SDVideoPath=video_name_SD, FMVideoPath=video_m_name, SocialDistance=act_dist,
                                       FaceMask=act_mask)


if __name__ == '__main__':
    url = 'http://192.168.43.97:8080/shot.jpg'
    # create_datastore()
    # # block for streaming (ACTIVATE AS REQD)
    # p1 = Process(target=mask_detector_img, args=(url,))
    # p2 = Process(target=person_dist_img, args=(url,))

    # p1.start()
    # p2.start()

    # p1.join()
    # p2.join()

    # # # block for saved videos (ACTIVATE AS REQD)

    # print("Time taken for execution is:",end_time-start_time)
    # df = tbl_Incident_Master.objects.create(
    #     'CameraId' = cameraId,
    #     'Date_Time_frm' = current_time,
    #     'Date_Time_to' = end_time,
    #     'VideoPath' = videoPath,
    #     'SocialDistance' = act_dist,
    #     'FaceMask' = act_mask
    # )
    # connection = pymysql.connect(host='localhost',
    #                              user='root',
    #                              password='root',
    #                              db='mydb')
    # cursor = connection.cursor()
    # #df = pd.read_sql("select * from detect_tbl_incident_master ORDER BY incidentId DESC LIMIT 1", connection)
    #
    #
    # CameraId = 6
    #
    # sql = "INSERT INTO `detect_tbl_incident_master` ( `CameraId`, `Date_Time_frm`, `Date_Time_to`, `SDVideoPath`, `FMVideoPath`,`SocialDistance`,`FaceMask`) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    # cursor.execute(sql, ( CameraId, CameraId, end_time, video_name_SD,video_m_name, act_dist, act_mask))
    #
    #
    #
    # connection.commit()
    # # # block for combining output in saved video (ACTIVATE AS REQD)
    # combine(video)


    # # # block for combining output in streaming (ACTIVATE AS REQD)
    # combine_img(url)