import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

# pip install filterpy
# pip install lap

from sort import Sort


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):

        model = YOLO("yolov8m.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame, verbose=False)

        return results

    def get_results(self, results):

        detections_list = []

        # Extract detections
        for result in results[0]:
            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()

            merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]

            detections_list.append(merged_detection)

        return np.array(detections_list)

    def draw_bounding_boxes_with_id(self, img, bboxes, ids):

        for bbox, id_ in zip(bboxes, ids):
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

        return img

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # SORT
        sort = Sort(max_age=40, min_hits=8, iou_threshold=0.50)

        fps_counter = 0
        start_time = time()
        fps = 0

        while True:
            fps_counter += 1
            cur_time = time()
            time_diff = cur_time - start_time
            if time_diff > 2.0:
                fps = fps_counter/np.round(time_diff)
                start_time = time()
                fps_counter = 0


            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            detections_list = self.get_results(results)

            # SORT Tracking
            if len(detections_list) == 0:
                detections_list = np.empty((0, 5))

            res = sort.update(detections_list)

            boxes_track = res[:, :-1]
            boxes_ids = res[:, -1].astype(int)

            frame = self.draw_bounding_boxes_with_id(frame, boxes_track, boxes_ids)


            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)
detector()
