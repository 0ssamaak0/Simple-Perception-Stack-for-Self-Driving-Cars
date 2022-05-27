import numpy as np
import time
import cv2
import sys
import matplotlib.pyplot as plt

# Weights, cfg and label paths
labels = open("coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

names = net.getLayerNames()

layers_names = [names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

capture = cv2.VideoCapture("Project_data/project_video.mp4")
isTrue, frame = capture.read()

fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter("result2.mp4", fourcc, fps,
                         (frame.shape[1], frame.shape[0]))

i = 0
save = i
t1 = time.time()

while i < frame_count:
    isTrue, frame = capture.read()
    if not isTrue:
        print(percentage + "%")
        break
    # frame = cv2.imread("Project_data/test_images/test1.jpg")
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), crop=False, swapRB=False)

    net.setInput(blob)

    layers_output = net.forward(layers_names)

    boxes = []
    confidences = []
    classIDs = []

    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > 0.85:
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")

                # openCV takes left-top point

                x = int(bx - (bw / 2))
                y = int(by - (bh / 2))

                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.8)

    # x, y, w, h = [0, 0, 0, 0]
    for i in indices:
        (x, y) = boxes[i][0], boxes[i][1]
        (w, h) = boxes[i][2], boxes[i][3]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"{labels[classIDs[i]]}: {round(confidences[i] * 100, 2)}%", (x + 5, y - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)

    i += 1
    if i < save:
        i = save + 1
    save = i

    t2 = divmod(time.time() - t1, 60)

    mins = round(t2[0])
    if mins < 10:
        mins = "0" + str(mins)

    secs = round(t2[1])
    if secs < 10:
        secs = "0" + str(secs)

    percentage = round(((i * 100 / fps) / duration), 2)

    loading = ("■" * int(percentage / 2)) + ("□" * (50 - int(percentage)))

    sys.stdout.write(f"\r{percentage}% time:{mins}:{secs} {loading}")
    sys.stdout.flush()

    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
