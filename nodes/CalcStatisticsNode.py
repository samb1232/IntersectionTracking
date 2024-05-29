import json

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

        self.road_directions_data: dict = {}

        # data format: road_from: {road_to: {"car": 0, "bus": 0, "truck": 0}}

        for i in range(1, 5):
            self.road_directions_data[i] = {}
            for j in range(1, 5):
                if i == j:
                    continue
                self.road_directions_data[i][j] = {"car": Vc(), "bus": Vc(), "truck": Vc(), "motorcycle": Vc()}

        self.people_directions_data: dict = {}

        for i in range(1, 4):
            self.people_directions_data[i] = {}
            for j in range(1, 4):
                if i == j:
                    continue
                self.people_directions_data[i][j] = Vc()

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"CalcStatisticsNode | Неправильный формат входного элемента {type(frame_element)}"

        buffer_tracks = frame_element.buffer_tracks

        info_dictionary = {"cars_on_screen": len(frame_element.id_list)}

        # Посчитаем число машин которые давно живут и имеют значения дороги приезда
        for _, track_element in buffer_tracks.items():
            if (track_element.timestamp_last - track_element.timestamp_init_road > self.min_time_life_track
                    and track_element.start_road is not None and track_element.end_road is not None):
                if track_element.cls == "person":
                    if track_element.start_road == 4:
                        track_element.start_road = None
                        continue
                    if track_element.end_road == 4:
                        track_element.end_road = None
                        continue
                        # TODO: КОСТЫЛЬ ВОНЮЧИЙ
                    self.people_directions_data[track_element.start_road][track_element.end_road].add(track_element.id)
                else:
                    self.road_directions_data[track_element.start_road][track_element.end_road][track_element.cls].add(
                        track_element.id)

        road_statistic_cars = {}
        for i in range(1, 5):
            road_statistic_cars[i] = {}
            for j in range(1, 5):
                if j == i:
                    continue
                road_statistic_cars[i][j] = {}
                for vehicle_cls, counter in self.road_directions_data[i][j].items():
                    road_statistic_cars[i][j][vehicle_cls] = counter.get_len()

        road_statistic_people = {}
        for i in range(1, 4):
            road_statistic_people[i] = {}
            for j in range(1, 4):
                if j == i:
                    continue
                road_statistic_people[i][j] = {}
                road_statistic_people[i][j] = self.people_directions_data[i][j].get_len()

        info_dictionary["vehicles_data"] = json.dumps(road_statistic_cars)
        info_dictionary["people_data"] = json.dumps(road_statistic_people)

        # Запись результатов обработки:
        frame_element.info = info_dictionary
        return frame_element

    def clear_counters_data(self):
        for i in range(1, 5):
            for j in range(1, 5):
                if i == j:
                    continue
                for counter in self.road_directions_data[i][j].values():
                    counter.clear()

        for i in range(1, 4):
            self.people_directions_data[i] = {}
            for j in range(1, 4):
                if i == j:
                    continue
                self.people_directions_data[i][j] = Vc()
