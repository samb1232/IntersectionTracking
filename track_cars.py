import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

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

    def __init__(self, capture_source):
        self.capture_source = capture_source

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

        # Extract detections for classes
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)

            if class_id in self.classes_to_track:
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]

                detections_list.append(merged_detection)
                classes_list.append(class_id)

        return np.array(detections_list), np.array(classes_list)

    def draw_bounding_boxes_with_id(self, frame, bboxes, classes_list, ids):
        for bbox, id_, clss in zip(bboxes, ids, classes_list):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{self.CLASS_NAMES_DICT[clss[0]]}. id: {str(id_)}", (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1)

        return frame

    def process(self):
        cap = cv2.VideoCapture(self.capture_source)
        assert cap.isOpened()

        sort = Sort(max_age=100, min_hits=15, iou_threshold=0.15)

        # FPS counter
        fps_counter = 0
        start_time = time()
        fps = 0

        # Main cycle
        while True:
            fps_counter += 1
            cur_time = time()
            time_diff = cur_time - start_time
            if time_diff > 2.0:
                fps = fps_counter / np.round(time_diff)
                start_time = time()
                fps_counter = 0

            # Read frame
            ret, frame = cap.read()
            frame = cv2.resize(frame, (1280, 720))
            assert ret

            prediction = self.predict(frame)
            detections_list, classes_list = self.get_results(prediction)

            # SORT Tracking
            if len(detections_list) == 0:
                detections_list = np.empty((0, 5))
                classes_list = np.empty((0,), dtype=int)

            res = sort.update(detections_list)
            boxes_coordinates = res[:, :-1]
            boxes_ids = res[:, -1].astype(int)


            # Draw frame
            frame = self.draw_bounding_boxes_with_id(frame, boxes_coordinates, classes_list, boxes_ids)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            # End of cycle
            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detector = ObjectDetection(capture_source=os.path.join("videos", "vid1.mp4"))
    detector.process()
