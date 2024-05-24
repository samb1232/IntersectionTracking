import json
from collections import deque

import numpy as np

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time, VehiclesCounter as Vc


class CalcStatisticsNode:
    """Модуль для расчета загруженности дорог (вычисление статистик)"""

    def __init__(self, config: dict) -> None:
        config_general = config["general"]

        self.time_buffer_analytics = config_general[
            "buffer_analytics"
        ]  # размер времени буфера в минутах
        self.min_time_life_track = config_general[
            "min_time_life_track"
        ]  # минимальное время жизни трека в сек
        self.count_cars_buffer_frames = config_general["count_cars_buffer_frames"]
        self.cars_buffer = deque(maxlen=self.count_cars_buffer_frames)  # создали буфер значений
        self.road_directions_data: dict = {}

        # data format: road_from: {road_to: {"cars": 0, "busses": 0, "trucks": 0}}

        for i in range(1, 5):
            self.road_directions_data[i] = {}
            for j in range(1, 5):
                if i == j:
                    continue
                self.road_directions_data[i][j] = {"cars": Vc(), "busses": Vc(), "trucks": Vc()}

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"CalcStatisticsNode | Неправильный формат входного элемента {type(frame_element)}"

        buffer_tracks = frame_element.buffer_tracks
        self.cars_buffer.append(len(frame_element.id_list))

        info_dictionary = {"cars_amount": round(np.mean(self.cars_buffer))}

        # Посчитаем число машин которые давно живут и имеют значения дороги приезда
        for _, track_element in buffer_tracks.items():
            if (
                    track_element.timestamp_last - track_element.timestamp_init_road
                    > self.min_time_life_track
                    and track_element.start_road is not None and track_element.end_road is not None
            ):
                self.road_directions_data[track_element.start_road][track_element.end_road]["cars"].add(
                    track_element.id)
                # TODO: поменять чтобы треки считались по классам тс

        road_statistic = {}
        for i in range(1, 5):
            road_statistic[i] = {}
            for j in range(1, 5):
                if j == i:
                    continue
                road_statistic[i][j] = {}
                for vehicle_cls, counter in self.road_directions_data[i][j].items():
                    road_statistic[i][j][vehicle_cls] = counter.get_len()

        info_dictionary["vehicles_data"] = json.dumps(road_statistic)

        # Запись результатов обработки:
        frame_element.info = info_dictionary

        return frame_element

    def clear_vehicle_data(self):
        for i in range(1, 5):
            for j in range(1, 5):
                if i == j:
                    continue
                for counter in self.road_directions_data[i][j].values():
                    counter.clear()
