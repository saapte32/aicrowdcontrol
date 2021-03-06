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
import pandas as pd


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2] = _sigmoid(netout[..., :2])
    netout[..., 4:] = _sigmoid(netout[..., 4:])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h * grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if (objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w  # center position, unit: image width
            y = (row + y) / grid_h  # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w  # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h  # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = filename
    width, height = image.size
    # load the image with the required size
    # image = load_img(filename, target_size=shape)
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i] * 100)
                # don't break, many labels may trigger for one box
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores):
    centers = []
    # load the image
    data = filename
    # plot the image
    pyplot.imshow(data)
    # get the context for drawing boxes
    ax = pyplot.gca()
    # plot each box
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        # calculate width and height of the box
        width, height = x2 - x1, y2 - y1
        centers.append([int((x1 + x2) / 2), int(y2)])

        # create the shape
        rect = Rectangle((x1, y1), width, height, fill=False, color='white')
        # draw the box
        ax.add_patch(rect)
        # data = ImageDraw.Draw(data).rectangle([x1, y1, x2, y2], outline = 'red')
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        pyplot.text(x1, y1, label, color='white')
    # show the plot
    pyplot.show()
    return centers


    # print(df1)


def draw_box1(image, v_boxes, v_labels, v_scores):
    centers = []
    for i in range(len(v_boxes)):
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        centers.append([int((x1 + x2) / 2), int(y2)])

        ImageDraw.Draw(image).rectangle([x1, y1, x2, y2], outline="white")
        label = "%s %d (%.3f)" % (v_labels[i], i + 1, v_scores[i])
        ImageDraw.Draw(image).text((x1, y1), label, (255, 255, 255))
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return predict(frame), centers


labelsPath = os.path.join("yolo.names")
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.join("yolov3_custom_train_6000.weights")
configPath = os.path.join("yolov3_custom_train.cfg")

net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# def display_img(img, cmap=None):
#
#     fig = plt.figure(figsize=(12, 12))
#     ax = fig.add_subplot(111)
#     ax.imshow(img, cmap)


def predict(image):
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    (H, W) = image.shape[:2]

    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    centers = []
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

                centers.append([int(centerX), int(centerY - height / 2)])

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, threshold, 0.1)
    m_count = 0
    nm_count = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if LABELS[classIDs[i]] == 'masked':
                color = (0, 255, 0)
                m_count += 1
            if LABELS[classIDs[i]] == 'not_masked':
                color = (0, 0, 255)
                nm_count += 1
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

        # text = "{}".format(LABELS[classIDs[i]], confidences[i])
        #             cv2.putText(image, text, (x + w - 1, y + h + 1), cv2.FONT_HERSHEY_SIMPLEX,
        #                 0.3, color, 1)
        # def dist(centers):
        #     d = []
        #     p = []
        #     for i in range(len(centers)):
        #         for j in range(i + 1, len(centers)):
        #             dis = ((abs(centers[i][0] - centers[j][0])) ** 2 + (abs(centers[i][1] - centers[j][1])) ** 2) ** 0.5
        #             d.append(int(dis))
        #             p.append([i + 1, j + 1])
        #     return p, d
        #
        # print("Total number of people : ", len(centers))
        # print("\n")
        # p, d = dist(centers)
        #
        # F = 4
        # R = 16
        # act_dist = []
        # ifov = (5 / F) * (10) ** -3
        # for i in d:
        #     act = R * i * ifov
        #     act_dist.append(act)
        #
        # def violation(act_dist, p):
        #     close = []
        #     grp = []
        #     grp1 = []
        #     for i in range(len(act_dist)):
        #         if act_dist[i] < 1.82:
        #             close.append('Unsafe')
        #             grp.append(p[i])
        #         for sublist in grp:
        #             for val in sublist:
        #                 grp1.append(val)
        #         else:
        #             close.append('Safe')
        #
        #     return close, set(grp1)
        #
        # close, grp1 = violation(act_dist, p)
        # df1 = pd.DataFrame(list(zip(p, act_dist, close)), columns=['Person', 'Actual Distance(meter)', 'Status'])
        # # print(len(df1.index))
        # print(df1)
        # df1 = df1.loc[df1['Status'] == 'Unsafe']
        # print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
        # print("People responsible for group violations : {}".format(grp1))

        print("People with masks =", m_count)
        print("People not wearing masks =", nm_count)
        print("Alert percentage =", (nm_count * 100) / (m_count + nm_count))

    return image


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
    #     p, d = dist(centers)
    F = 4
    R = 16
    act_dist = []
    ifov = (5 / F) * (10) ** -3
    for i in d:
        act = R * i * ifov
        act_dist.append(act)

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

    close, grp1 = violation(act_dist, p)

    df1 = pd.DataFrame(list(zip(p, act_dist, close)), columns=['Person', 'Actual Distance(meter)', 'Status'])
    # print(len(df1.index))
    print(df1)
    df1 = df1.loc[df1['Status'] == 'Unsafe']
    print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
    print("People responsible for group violations : {}".format(grp1))


# load yolov3 model
#import pdb;pdb.set_trace()
model = load_model(r'model.h5')
# define the expected input shape for the model
input_w, input_h = 416, 416
# define our new photo
# photo_filename = '/content/4people.jpg'
# load and prepare image
cap = cv2.VideoCapture(r'VID-20200420-WA0006.mp4')
#cap = cv2.VideoCapture(r'VID-20200511-WA0013.mp4')
centers = list()
while cap.isOpened():
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
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    # suppress non-maximal boxes
    do_nms(boxes, 0.5)
    # define the labels
    #     labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    #     "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    #     "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    #     "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    #     "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    #     "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    #     "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    #     "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    #     "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    #     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    labels = ["person"]
    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
    # summarize what we found
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])
    # draw what we found
    # draw_boxes(frame, v_boxes, v_labels, v_scores)
    frame1, centers = draw_box1(frame, v_boxes, v_labels, v_scores)
    #predict(frame1)
    dist(centers)

    #     print(centers)

    cv2.imshow("out", frame1)
    frameTime=10
    cv2.waitKey(frameTime)
cv2.destroyAllWindows()