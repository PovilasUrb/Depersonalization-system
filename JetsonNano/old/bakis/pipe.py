import cv2
import numpy as np
import time

protoFile = "pose_deploy_linevec.prototxt"
weightsFile = "pose_iter_440000.caffemodel"
nPoints = 18
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4],
    [5, 6], [6, 7], [1, 8], [8, 9],
    [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]
inWidth = 368
inHeight = 368

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    blob = cv2.dnn.blobFromImage(frame, 1.0/255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()
    
    H = output.shape[2]
    W = output.shape[3]
    
    points = []
    threshold = 0.1
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        x = (frame.shape[1] * point[0]) / W
        y = (frame.shape[0] * point[1]) / H
        if prob > threshold:
            cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(output_frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)
    
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(output_frame, points[partA], points[partB], (0, 255, 0), 2)
            cv2.circle(output_frame, points[partA], 3, (0, 0, 255), thickness=-1)
            cv2.circle(output_frame, points[partB], 3, (0, 0, 255), thickness=-1)
    
    cv2.imwrite("debug_frame_{}.jpg".format(frame_num), output_frame)
    frame_num += 1
    if frame_num > 10:
        break

cap.release()
