import numpy as np
import torch
from ultralytics import YOLO

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time


class DetectionTrackingNodes:
    """Модуль инференса модели детекции и трекинг алгоритма"""

    def __init__(self, config) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Детекция будет производиться на {device}')

        config_yolo = config["detection_node"]
        self.model = YOLO(config_yolo["weight_pth"])
        self.model.fuse()
        self.classes = self.model.names
        self.conf = config_yolo["confidence"]
        self.iou = config_yolo["iou"]
        self.imgsz = config_yolo["imgsz"]
        self.classes_to_detect = config_yolo["classes_to_detect"]

        config_general = config["general"]
        self.tracker_conf_path = config_general["tracker_conf_path"]
        self.show_only_yolo_detections = config["show_node"]["show_only_yolo_detections"]

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"DetectionTrackingNodes | Неправильный формат входного элемента {type(frame_element)}"

        frame = frame_element.frame.copy()

        track_list = self.model.track(frame, persist=True, tracker=self.tracker_conf_path)[0].boxes.data.cpu().numpy()
        classes_list = []
        confs_list = []
        boxes_ids = []
        boxes_coordinates = []
        for box in track_list:
            if box[6] in self.classes_to_detect:
                classes_list.append(self.classes[box[6].astype(int)])
                confs_list.append(box[5])
                boxes_ids.append(box[4].astype(int))
                boxes_coordinates.append([i.astype(int) for i in box[0:4]])

        # Получение id list
        frame_element.id_list = boxes_ids
        # Получение box list
        frame_element.tracked_xyxy = boxes_coordinates
        # Получение object class names
        frame_element.tracked_cls = classes_list
        # Получение conf scores
        frame_element.tracked_conf = confs_list

        if self.show_only_yolo_detections:
            outputs = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False,
                                         iou=self.iou, classes=self.classes_to_detect)

            frame_element.detected_conf = outputs[0].boxes.conf.cpu().tolist()
            detected_cls = outputs[0].boxes.cls.cpu().int().tolist()
            frame_element.detected_cls = [self.classes[i] for i in detected_cls]
            frame_element.detected_xyxy = outputs[0].boxes.xyxy.cpu().int().tolist()

        return frame_element

    def _get_results_for_tracker(self, results) -> np.ndarray:
        # Приведение данных в правильную форму для трекера
        detections_list = []
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            # трекаем те же классы, что и детектируем
            if class_id[0] in self.classes_to_detect:
                bbox = result.boxes.xyxy.cpu().numpy()
                confidence = result.boxes.conf.cpu().numpy()

                merged_detection = [
                    bbox[0][0],
                    bbox[0][1],
                    bbox[0][2],
                    bbox[0][3],
                    confidence[0],
                    class_id[0],
                ]

                detections_list.append(merged_detection)

        return np.array(detections_list)
