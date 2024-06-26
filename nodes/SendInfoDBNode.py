import logging
import time

import psycopg2

from elements.FrameElement import FrameElement
from elements.VideoEndBreakElement import VideoEndBreakElement
from utils_local.utils import profile_time

logger = logging.getLogger(__name__)


class SendInfoDBNode:
    """Модуль для отправки актуальной информации о трафике в базу данных"""

    def __init__(self, config: dict) -> None:
        config_db = config["send_info_db_node"]
        self.update_period = config_db["update_period"] * 60
        self.table_name = config_db["table_name"]
        self.last_db_update = time.time()

        # Параметры подключения к базе данных
        db_connection = config_db["connection_info"]
        conn_params = {
            "user": db_connection["user"],
            "password": db_connection["password"],
            "host": db_connection["host"],
            "port": str(db_connection["port"]),
            "database": db_connection["database"],
        }

        self.buffer_analytics_sec = (
            config["general"]["buffer_analytics"] * 60
        )  # столько по времени буфер набирается и информацию о статистике выводить рано

        # Подключение к базе данных
        try:
            self.connection = psycopg2.connect(**conn_params)
            print("Connected to PostgreSQL")
        except (Exception, psycopg2.Error) as error:
            print("Error while connecting to PostgreSQL:", error)

        # Создание курсора для выполнения SQL-запросов
        self.cursor = self.connection.cursor()

        # SQL-запрос для удаления таблицы, если она уже существует
        drop_table_query = f"DROP TABLE IF EXISTS {self.table_name};"

        # Удаление таблицы, если она уже существует
        try:
            self.cursor.execute(drop_table_query)
            self.connection.commit()
        except (Exception, psycopg2.Error) as error:
            logger.error(
                f"Error while dropping table:: {error}"
            ) 

        # SQL-запрос для создания таблицы
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            timestamp INTEGER,
            timestamp_date TIMESTAMP,
            cars_on_screen INTEGER,
            vehicles_data JSONB,
            people_data JSONB
        );
        """

        # Создание таблицы
        try:
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info(
                f"Table {self.table_name} created successfully"
            )
        except (Exception, psycopg2.Error) as error:
            logger.error(
                f"Error while creating table: {error}"
            )

    @profile_time
    def process(self, frame_element: FrameElement) -> FrameElement:
        # Выйти из обработки если это пришел VideoEndBreakElement а не FrameElement
        if isinstance(frame_element, VideoEndBreakElement):
            return frame_element
        assert isinstance(
            frame_element, FrameElement
        ), f"SendInfoDBNode | Неправильный формат входного элемента {type(frame_element)}"

        # Получение значений для записи в бд новой строки:
        info_dictionary = frame_element.info
        timestamp = frame_element.timestamp
        timestamp_date = frame_element.timestamp_date

        # Проверка, нужно ли отправлять информацию в базу данных
        current_time = time.time()
        if current_time - self.last_db_update >= self.update_period:
            self._insert_in_db(info_dictionary, timestamp, timestamp_date)
            frame_element.send_info_of_frame_to_db = True
            self.last_db_update = (
                current_time  # Обновление времени последнего обновления базы данных
            )

        return frame_element

    def _insert_in_db(self, info_dictionary: dict, timestamp: float, timestamp_date: float) -> None:
        # Формирование и выполнение SQL-запроса для вставки данных в бд
        insert_query = (
            f"INSERT INTO {self.table_name} "
            "(timestamp, timestamp_date, cars_on_screen, vehicles_data, people_data) "
            "VALUES (%s, to_timestamp(%s), %s, %s, %s);"
        )
        try:
            self.cursor.execute(
                insert_query,
                (
                    timestamp,
                    timestamp_date,
                    info_dictionary["cars_on_screen"],
                    info_dictionary["vehicles_data"],
                    info_dictionary["people_data"]
                ),
            )
            self.connection.commit()
            logger.info(
                f"Successfully inserted data into PostgreSQL"
            )   
        except (Exception, psycopg2.Error) as error:
            logger.error(
                f"Error while inserting data into PostgreSQL: {error}"
            )
            self.connection.rollback()

