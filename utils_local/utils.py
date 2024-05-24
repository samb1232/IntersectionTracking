import logging
import time
from collections import deque

import numpy as np
from shapely.geometry import Point, Polygon

logger_profile = logging.getLogger("profile")


def profile_time(func):
    def exec_and_print_status(*args, **kwargs):
        t_start = time.time()
        out = func(*args, **kwargs)
        t_end = time.time()
        dt_msecs = (t_end - t_start) * 1000

        self = args[0]
        logger_profile.debug(
            f"{self.__class__.__name__}.{func.__name__}, time spent {dt_msecs:.2f} msecs"
        )
        return out

    return exec_and_print_status


class FPSCounter:
    def __init__(self, calc_time_period_n_frames: int) -> None:
        """Счетчик FPS по ограниченным участкам видео (скользящему окну).

        Args:
            calc_time_period_n_frames (int): количество фреймов окна подсчета статистики.
        """
        self.time_buffer = []
        self.calc_time_perion_N_frames = calc_time_period_n_frames

    def calc_FPS(self) -> float:
        """Производит рассчет FPS по нескольким кадрам видео.

        Returns:
            float: значение FPS.
        """
        time_buffer_is_full = len(self.time_buffer) == self.calc_time_perion_N_frames
        t = time.time()
        self.time_buffer.append(t)

        if time_buffer_is_full:
            self.time_buffer.pop(0)
            fps = len(self.time_buffer) / (self.time_buffer[-1] - self.time_buffer[0])
            return np.round(fps, 2)
        else:
            return 0.0


def intersects_central_point(tracked_xyxy, polygons):
    """Функция определяет присутcвие центральной точки bbox в области полигонов дорог

    Args:
        tracked_xyxy: координаты bbox
        polygons: словарь полигонов

    Returns:
        Лиибо None либо значение ключа (номер дороги - int)
    """
    # Центральная точка bbox:
    center_point = [
        (tracked_xyxy[0] + tracked_xyxy[2]) / 2,
        (tracked_xyxy[1] + tracked_xyxy[3]) / 2,
    ]
    center_point = Point(center_point)
    for key, polygon in polygons.items():
        polygon = Polygon([(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)])
        if polygon.contains(center_point):
            return int(key)
    return None


class VehiclesCounter:
    """
    Счетчик уникальных id. Хранит только последние уникальные id в очереди. При переполнении забывает старые id.

    """

    def __init__(self, capacity: int = 150):
        """
        Args:
            capacity: Длина очереди для запоминания последних id.
        """
        self.capacity = capacity
        self.queue = deque(maxlen=capacity)
        self.id_set = set()
        self.total_count = 0

    def add(self, id: int):
        if id in self.id_set:
            return

        if len(self.queue) >= self.capacity:
            oldest_id = self.queue.popleft()
            self.id_set.remove(oldest_id)

        self.queue.append(id)
        self.id_set.add(id)

        self.total_count += 1

    def get_len(self):
        return self.total_count

    def reset_len(self):
        self.total_count = 0

    def clear(self):
        self.reset_len()
        self.queue.clear()
