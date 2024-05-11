import os

import numpy as np
from ultralytics import YOLO
import cv2

classes_to_track = [
    0,  # "person",
    1,  # "bicycle",
    2,  # "car",
    3,  # "motorcycle",
    5,  # "bus",
    6,  # "train",
    7,  # "truck"
]

# load yolov8 model
model = YOLO('yolov8m.pt')

# load video
cap = cv2.VideoCapture(0)  # os.path.join("videos", "vid1.mp4"))

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        result = model.track(frame, persist=True)[0]


        frame_ = result.plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
