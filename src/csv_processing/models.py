import pandas as pd
import numpy as np


class Point(object):
    """
      Класс точек местоположения метеостанций
      Attributes:
          id: номер метеостанции.
          name: неаименование населённого пункта.
          x: широта.
          y: долгота.
          z: высота.
      """

    def __init__(self, id, name, x, y, z):
        self.id = id
        self.name = name
        self.x = x
        self.y = y
        self.z = z

    def distance(self, point1, point2):
        """
        Функция вычисления дистанции между точками
        Args:
            point1 (Point): Первая точка.
            point2 (Point): Вторая точка.
        """
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2 + (point1.z - point2.z) ** 2)


class Element(object):
    """
        Класс пропуска в данных
        Attributes:
        index: индекс строки с пропуском из исходного dataframe.
        station_number: номер метеостанции.
        neighbors_list: список номеров соседних метеостанций.
        dataframe: данные соседних метеостанций за соответствующий год.
        datarow: строка с пропуском.
    """
    neighbors_list = list()
    dataframe = pd.DataFrame(
        columns=['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'per_year',
                 'station_Number', 'type_data', 'Index', 'Name', 'y', 'x', 'z', 'Country'])
    datarow = pd.DataFrame(
        columns=['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'per_year',
                 'station_Number', 'type_data', 'Index', 'Name', 'y', 'x', 'z', 'Country'])

    def __init__(self, index, station_number, neighbors_list, datarow):
        self.index = index
        self.station_number = station_number
        self.neighbors_list = neighbors_list
        self.dataframe = pd.DataFrame(
            columns=['year', 'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
                     'per_year', 'station_Number', 'type_data', 'Index', 'Name', 'y', 'x', 'z', 'Country'])
        self.datarow = datarow
