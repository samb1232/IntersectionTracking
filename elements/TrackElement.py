from __future__ import annotations


class TrackElement:
    # Класс, содержаций информацию о конкретном треке машины
    def __init__(
        self,
        id: int,
        timestamp_first: float,
        start_road: int | None = None,
    ) -> None:
        self.id: int = id  # Номер этого трека
        self.timestamp_first: float = timestamp_first  # Таймстемп инициализации (в сек)
        self.timestamp_last: float = timestamp_first  # Таймстемп последнего обнаружения (в сек)
        self.start_road: int | None = start_road  # Номер дороги, с которой приехал
        self.end_road: int | None = None  # Номер дороги, на которую уехал
        self.timestamp_init_road: float = timestamp_first  # Таймстемп инициализации номера дороги (в сек)
        # ps: если дорога не будет определена, то значение останется равным первому появлению

    def update(self, timestamp):
        # Обновление времени последнего обнаружения
        self.timestamp_last = timestamp
