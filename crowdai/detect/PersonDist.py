import cv2
import numpy as np
import pandas as pd
import time

scale = 0.00392
classes_file = "coco.names"
weights = "yolov3.weights"
config_file = "yolov3.cfg"

# read class names from text file
classes = None
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# generate different colors for different classes
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet(weights, config_file)


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
    F = 3.6
    R = 9
    act_dist = []
    ifov = (1.1 / F) * (10) ** -3
    for i in d:
        act = R * i * ifov
        act_dist.append(round(act, 2))

    close, grp1 = violation(act_dist, p)

    df1 = pd.DataFrame(list(zip(p, act_dist, close)), columns=['Person', 'Actual Distance(meter)', 'Status'])
    # print(len(df1.index))
    # print(df1)
    df1 = df1.loc[df1['Status'] == 'Unsafe']
    print("\nTotal number of violations in the range : {}\n".format(len(df1.index)))
    print("People responsible for group violations : {}".format(grp1))


# return act_dist

def run(frame):
    Width = frame.shape[1]
    Height = frame.shape[0]

    # create input blob
    blob = cv2.dnn.blobFromImage(frame, scale, (288, 288), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # run inference through the network and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer get the confidence, class id,
    # bounding box params and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    centers = []
    # go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        centers.append([x, y + h / 2])
        draw_bounding_box(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        dist(centers)


# return frame


# function to get the output layer names in the architecture
def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    if class_id == 0:
        centers = []
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), (0, 255, 0), 1)

        cv2.putText(img, label, (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imshow(' ', img)


if __name__ == "__main__":
    # read frame ...
    cap = cv2.VideoCapture(r'hong.mp4')
    # cap=cv2.VideoCapture(0)
    while cap.isOpened():
        s = time.time()
        ret, frame = cap.read()
        run(frame)
        print(time.time() - s)
        if cv2.waitKey(1) & 0xff == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()