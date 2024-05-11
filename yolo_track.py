import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

import os


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

    def track(self, frame):
        results = self.model.track(frame, persist=True)
        return results[0]

    def draw_bounding_boxes_with_id(self, frame, bboxes, classes_list, ids):
        for bbox, id_, clss in zip(bboxes, ids, classes_list):
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(frame, f"{self.CLASS_NAMES_DICT[clss]}. id: {str(int(id_))}", (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)

        return frame

    def process(self):
        cap = cv2.VideoCapture(self.capture_source)
        assert cap.isOpened()


        # FPS counter
        fps_counter = 0
        start_time = time()
        fps = 0

        ret = True

        # Main cycle
        while ret:
            fps_counter += 1
            cur_time = time()
            time_diff = cur_time - start_time
            if time_diff > 2.0:
                fps = fps_counter / np.round(time_diff)
                start_time = time()
                fps_counter = 0

            # Read frame
            ret, frame = cap.read()
            assert ret
            frame = cv2.resize(frame, (1280, 720))

            result = self.track(frame)

            res_data = result.boxes.data.cpu().numpy()

            classes_list = []
            confs_list = []
            boxes_ids = []
            boxes_coordinates = []
            for box in res_data:
                if box[6] in self.classes_to_track:
                    classes_list.append(box[6].astype(int))
                    confs_list.append(box[5])
                    boxes_ids.append(box[4])
                    boxes_coordinates.append(box[0:4])

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
