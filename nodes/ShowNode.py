import random

import cv2
import numpy as np

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time, FPSCounter


class ShowNode:
    """Модуль отвечающий, за визуализацию результатов"""

    def __init__(self, config) -> None:
        data_colors_cars = config["general"]["colors_of_roads_cars"]
        self.colors_roads_cars = {key: tuple(value) for key, value in data_colors_cars.items()}
        data_colors_people = config["general"]["colors_of_roads_people"]
        self.colors_roads_people = {key: tuple(value) for key, value in data_colors_people.items()}

        self.buffer_analytics_sec = (
                config["general"]["buffer_analytics"] * 60
        )  # столько по времени буфер набирается и информацию о статистеке выводить рано

        config_show_node = config["show_node"]
        self.scale = config_show_node["scale"]
        self.fps_counter_N_frames_stat = config_show_node["fps_counter_N_frames_stat"]
        self.default_fps_counter = FPSCounter(self.fps_counter_N_frames_stat)
        self.draw_fps_info = config_show_node["draw_fps_info"]
        self.show_roi_cars = config_show_node["show_roi_cars"]
        self.show_roi_people = config_show_node["show_roi_cars"]
        self.overlay_transparent_mask = config_show_node["overlay_transparent_mask"]
        self.imshow = config_show_node["imshow"]
        self.show_only_yolo_detections = config_show_node["show_only_yolo_detections"]
        self.show_track_id_different_colors = config_show_node["show_track_id_different_colors"]
        self.show_info_statistics = config_show_node["show_info_statistics"]

        self.show_number_of_road = True  # отображение номеров дорог

        # Параметры для шрифтов:
        self.fontFace = 1
        self.fontScale = 1.5
        self.thickness = 2

        # Параметры для полигонов и bboxes:
        self.thickness_lines = 2

        # Параметры для экрана статистики:
        self.width_window = 700  # ширина экрана в пикселях

    @profile_time
    def process(self, frame_element: FrameElement, fps_counter=None) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"ShowNode | Неправильный формат входного элемента {type(frame_element)}"

        frame_result = frame_element.frame.copy()

        # Отображение лишь результатов детекции:
        if self.show_only_yolo_detections:
            for box, class_name in zip(frame_element.detected_xyxy, frame_element.detected_cls):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника
                cv2.rectangle(frame_result, (x1, y1), (x2, y2), (0, 0, 0), 2)
                # Добавление подписи с именем класса
                cv2.putText(
                    frame_result,
                    class_name,
                    (x1, y1 - 10),
                    fontFace=self.fontFace,
                    fontScale=self.fontScale,
                    thickness=self.thickness,
                    color=(0, 0, 255),
                )

        else:
            # Отображение результатов трекинга:
            for box, class_name, id in zip(
                    frame_element.tracked_xyxy, frame_element.tracked_cls, frame_element.id_list
            ):
                x1, y1, x2, y2 = box
                # Отрисовка прямоугольника
                if self.show_track_id_different_colors:
                    # Отображаем каждый трек своим цветом
                    random.seed(int(id))
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    # Отображаем каждый трек согласно цвету пересечения с дорогой
                    try:
                        start_road = frame_element.buffer_tracks[int(id)].start_road
                        if start_road is not None:
                            if frame_element.buffer_tracks[int(id)].cls == "person":
                                color = self.colors_roads_people[int(start_road)]
                            else:
                                color = self.colors_roads_cars[int(start_road)]
                        else:  # бокс черным цветом если еще нет информации о стартовой дороге
                            color = (0, 0, 0)
                    except KeyError as e:  # На случай если машина еще в кадре, а трек уже удален
                        color = (255, 255, 255)

                if int(id) in frame_element.buffer_tracks:
                    cv2.rectangle(frame_result, (x1, y1), (x2, y2), color, self.thickness_lines)
                    # Добавление подписи с именем класса
                    cv2.putText(
                        frame_result,
                        f"{frame_element.buffer_tracks[int(id)].cls}: {id}",
                        (x1, y1 - 10),
                        fontFace=self.fontFace,
                        fontScale=self.fontScale,
                        thickness=self.thickness,
                        color=(0, 0, 255),
                    )

        # Построение полигонов дорог для машин
        if self.show_roi_cars:
            for road_id, points in frame_element.cars_roads_info.items():
                color = self.colors_roads_cars[int(road_id)]
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame_result,
                    [points],
                    isClosed=True,
                    color=color,
                    thickness=self.thickness_lines,
                )

                if self.overlay_transparent_mask:
                    frame_result = self._overlay_transparent_mask(
                        frame_result, points, mask_color=color, alpha=0.3
                    )

                # Отображение номера дороги в залитой окружности
                if self.show_number_of_road:
                    moments = cv2.moments(points)  # Найти центр области
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])

                        (label_width, label_height), _ = cv2.getTextSize(
                            str(road_id),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale * 1.3,
                            thickness=self.thickness,
                        )
                        # Определение размеров круга
                        circle_radius = max(label_width, label_height) // 2
                        # Рисование круга
                        cv2.circle(
                            frame_result,
                            (cx, cy),
                            circle_radius + 6,  # Добавляем небольшой отступ для текста
                            (200, 200, 200),
                            -1
                        )
                        # Нанесение подписи road_id в центре области
                        cv2.putText(
                            frame_result,
                            str(road_id),
                            (cx + 2 - label_width // 2, cy + 2 + label_height // 2),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale * 1.3,
                            thickness=self.thickness,
                            color=(0, 0, 0),
                        )

        # Построение полигонов дорог для пешеходов
        if self.show_roi_people:
            for road_id, points in frame_element.people_roads_info.items():
                color = self.colors_roads_people[int(road_id)]
                points = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    frame_result,
                    [points],
                    isClosed=True,
                    color=color,
                    thickness=self.thickness_lines,
                )

                if self.overlay_transparent_mask:
                    frame_result = self._overlay_transparent_mask(
                        frame_result, points, mask_color=color, alpha=0.3
                    )

                # Отображение номера дороги в залитой окружности
                if self.show_number_of_road:
                    moments = cv2.moments(points)  # Найти центр области
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])

                        (label_width, label_height), _ = cv2.getTextSize(
                            f"{road_id}`",
                            fontFace=self.fontFace,
                            fontScale=self.fontScale * 1.3,
                            thickness=self.thickness,
                        )
                        # Определение размеров круга
                        circle_radius = max(label_width, label_height) // 2
                        # Рисование круга
                        cv2.circle(
                            frame_result,
                            (cx, cy),
                            circle_radius + 6,  # Добавляем небольшой отступ для текста
                            (200, 200, 200),
                            -1
                        )
                        # Нанесение подписи road_id в центре области
                        cv2.putText(
                            frame_result,
                            f"{road_id}`",
                            (cx + 2 - label_width // 2, cy + 2 + label_height // 2),
                            fontFace=self.fontFace,
                            fontScale=self.fontScale * 1.3,
                            thickness=self.thickness,
                            color=(0, 0, 0),
                        )

        # Подсчет fps и отрисовка
        if self.draw_fps_info:
            fps_counter = fps_counter if fps_counter is not None else self.default_fps_counter
            fps_real = fps_counter.calc_FPS()

            text = f"FPS: {fps_real:.1f}"
            (label_width, label_height), _ = cv2.getTextSize(
                text,
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
            )
            cv2.rectangle(
                frame_result, (0, 0), (10 + label_width, 35 + label_height), (0, 0, 0), -1
            )
            cv2.putText(
                img=frame_result,
                text=text,
                org=(10, 40),
                fontFace=self.fontFace,
                fontScale=self.fontScale,
                thickness=self.thickness,
                color=(255, 255, 255),
            )

        # Обработка отдельного окна с выводом статистики
        if self.show_info_statistics:
            black_image = np.zeros((frame_result.shape[0], self.width_window, 3), dtype=np.uint8)
            data_info = frame_element.info

            # Текст для количества машин
            text_cars = f"Cars on screen: {data_info['cars_on_screen']}"
            # Начальная координата для текста
            y = 55
            # Выводим текст для количества машин
            cv2.putText(
                img=black_image,
                text=text_cars,
                org=(20, y),
                fontFace=self.fontFace,
                fontScale=self.fontScale * 1.5,
                thickness=self.thickness,
                color=(255, 255, 255),
            )

            frame_result = np.hstack((frame_result, black_image))

        frame_element.frame_result = frame_result
        frame_show = cv2.resize(frame_result.copy(), (-1, -1), fx=self.scale, fy=self.scale)

        if self.imshow:
            cv2.imshow(frame_element.source, frame_show)
            cv2.waitKey(1)

        return frame_element

    @staticmethod
    def _overlay_transparent_mask(img, points, mask_color=(0, 255, 255), alpha=0.3):
        img_shape = img.shape
        binary_mask = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
        binary_mask = cv2.fillPoly(binary_mask, pts=[points], color=1)
        colored_mask = (binary_mask[:, :, np.newaxis] * mask_color).astype(np.uint8)
        return cv2.addWeighted(img, 1, colored_mask, alpha, 0)
