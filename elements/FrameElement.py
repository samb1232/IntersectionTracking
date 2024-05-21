from __future__ import annotations

import time

import numpy as np


class FrameElement:
    # Класс, содержащий информацию о конкретном кадре видеопотока
    def __init__(
            self,
            source: str,
            frame: np.ndarray,
            timestamp: float,
            frame_num: float,
            roads_info: dict,
            frame_result: np.ndarray | None = None,
            detected_conf: list | None = None,
            detected_cls: list | None = None,
            detected_xyxy: list[list] | None = None,
            tracked_conf: list | None = None,
            tracked_cls: list | None = None,
            tracked_xyxy: list[list] | None = None,
            id_list: list | None = None,
            buffer_tracks: dict | None = None,
    ) -> None:
        self.source: str = source  # Путь к видео или номер камеры с которой берем поток
        self.frame: np.ndarray = frame  # Кадр bgr формата
        self.timestamp: float = timestamp  # Значение времени с начала потока (в секундах)
        self.frame_num: float = frame_num  # Нормер кадра с потока
        self.roads_info: dict = roads_info  # Словарь с координатми дорог, примыкающих к участку кругового движения
        self.frame_result: np.ndarray | None = frame_result  # Итоговый обработанный кадр
        self.timestamp_date: float = time.time()  # Время в момент обработки кадра unix формат (в секундах)

        # Результаты на выходе с YOLO:
        self.detected_conf: list | None = detected_conf  # Список уверенностей задетектированных объектов
        self.detected_cls: list | None = detected_cls  # Список классов задетектированных объектов
        self.detected_xyxy: list[list] | None = detected_xyxy  # Список списков с координатами xyxy боксов

        # Результаты корректировки трекинг алгоритмом:
        self.tracked_conf: list | None = tracked_conf  # Список уверенностей задетектированных объектов
        self.tracked_cls: list | None = tracked_cls  # Список классов задетектированных объектов
        self.tracked_xyxy: list[list] | None  = tracked_xyxy  # Список списков с координатами xyxy боксов
        self.id_list: list | None = id_list  # Список обнаруженных id трекуемых объектов

        # Постобработка кадра:
        self.buffer_tracks: dict | None = buffer_tracks  # Буфер актуальных треков за выбранное время анализа
        self.info: dict = {}  # Словарь с результирующей статистикой (загруженность дорог + число машин)
        self.send_info_of_frame_to_db: bool = False  # Флаг того, будет ли с это кадра инфа отправлена в бд
