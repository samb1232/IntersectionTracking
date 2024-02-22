import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import random

# pip install filterpy
# pip install lap

import os
from sort import Sort


class ObjectDetection:
    classes_to_track = [
        0,  # "person",
        1,  # "bicycle",
        2,  # "car",
        3,  # "motorcycle",
        5,  # "bus",
        6,  # "train",
        7,  # "truck"
    ]

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):

        model = YOLO("yolov8m.pt")
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame, verbose=False)

        return results

    def get_results(self, results):
        detections_list = []
        classes_list = []

        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id in self.classes_to_track:
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]

                detections_list.append(merged_detection)
                classes_list.append(class_id)

        return np.array(detections_list), classes_list

    def draw_bounding_boxes_with_id(self, frame, classes_list, bboxes, ids):

        for bbox, id_, clss in zip(bboxes, ids, classes_list):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{self.CLASS_NAMES_DICT[clss[0]]}. id: {str(id_)}", (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        return frame

    def draw_bounding_boxes_without_id(self, frame, results):
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, clss in zip(boxes, classes):
            # Generate a random color for each object based on its ID
            if clss != 0:
                random.seed(int(clss))
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3],), color, 2)
                cv2.putText(
                    frame,
                    f"{self.CLASS_NAMES_DICT[clss]}",
                    (box[0], box[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
        return frame

    def __call__(self):
        cap = cv2.VideoCapture("videos/vid1.mp4")
        assert cap.isOpened()

        # SORT
        sort = Sort(max_age=100, min_hits=15, iou_threshold=0.15)

        fps_counter = 0
        start_time = time()
        fps = 0

        while True:
            fps_counter += 1
            cur_time = time()

            time_diff = cur_time - start_time
            if time_diff > 2.0:
                fps = fps_counter / np.round(time_diff)
                start_time = time()
                fps_counter = 0

            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            assert ret

            prediction = self.predict(frame)
            results = self.get_results(prediction)

            classes_list = results[1]
            detections_list = results[0]

            # SORT Tracking
            if len(detections_list) == 0:
                detections_list = np.empty((0, 5))

            res = sort.update(detections_list)

            boxes_track = res[:, :-1]
            boxes_ids = res[:, -1].astype(int)

            frame = self.draw_bounding_boxes_with_id(frame, classes_list, boxes_track, boxes_ids)
            # frame = self.draw_bounding_boxes_without_id(frame, results)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)
detector()
