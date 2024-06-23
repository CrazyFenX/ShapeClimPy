from random import randint

import pandas as pd
from tqdm import tqdm
import seaborn as sns
import numpy as np
import scipy.spatial.distance

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
import statistics
import statsmodels.api as sm
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
from scipy.interpolate import splprep, splev
import pykrige as pk

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_squared_error
from pandas.core.frame import DataFrame

import matplotlib.pyplot as plt
from geopy.distance import geodesic

from models import Element, Point

stations = pd.read_csv('list_of_stations_full.csv')
clean_data = pd.read_csv('output_clean_data_full.csv')

# Объединение таблиц
joined_data = clean_data.merge(stations[["Latitude", "Longitude", "Elevation", "Index"]], left_on='station_Number', right_on='Index')

# Удаление скобок из столбца
joined_data['Elevation'] = joined_data['Elevation'].str.replace(r'"', '')

# Создание словаря для отображения старых и новых имен столбцов
columns_mapping = {
    'Year': 'year',
    'Jan': 'jan',
    'Feb': 'feb',
    'March': 'mar',
    'April': 'apr',
    'May': 'may',
    'June': 'jun',
    'July': 'jul',
    'August': 'aug',
    'September': 'sep',
    'October': 'oct',
    'November': 'nov',
    'December': 'dec',
    'Annual': 'per_year',
    'Station_Number': 'station_number',
    'Latitude': 'x',
    'Longitude': 'y',
    'Elevation': 'z',
    'Type_data': 'data_type',
}

# Переименование столбцов с использованием метода rename()
joined_data = joined_data.rename(columns=columns_mapping)

# Запись данных о станциях в список точек
points = []
for index, row in stations.iterrows():
    points.append(Point(row['Index'], row['Name'], row['Latitude'], row['Longitude'], row['Elevation']))
    # points.append(Point(row['Index'], row['Name'], row['x'], row['y'], row['z']))
print(len(points))


def get_neighbour_elements(dt, value_df, radius, station_col_name='station_Number', year_col_name='year', data_type_col_name='type_data'):
    """
    Получает список соседей для каждой точки в заданном DataFrame.

    :param dt: (DataFrame) Исходный DataFrame с данными о точках.
    :param value_df: (DataFrame) DataFrame со значениями точек.
    :param radius: (float) Радиус для поиска соседей.
    :param station_col_name: (str) Наименование колонки, содержащей номера метеостанций.
    :param year_col_name: (str) Наименование колонки, содержащей информацию о годе.
    :param data_type_col_name: (str) Наименование колонки, содержащей информацию о типе данных.

    :return (list) Список объектов Element, содержащих информацию о каждой точке и ее соседях.
    """
    rows_ids = []  # Список для отслеживания уже обработанных точек
    rows_neighbours = []  # Список объектов Element с информацией о каждой точке и ее соседях

    for rowIndex, row in dt.iterrows():
        if ((row[year_col_name], row[station_col_name])) in rows_ids:
            # Пропускаем точку, если она уже была обработана
            print("skip", row[year_col_name], row[station_col_name])
            continue

        neighbors_list = []  # Список для хранения соседних точек
        is_neighbours_found = False  # Флаг, указывающий, были ли найдены соседи

        for columnIndex, value in row.items():
            if pd.isnull(value) and not is_neighbours_found:
                rows_ids.append((row[year_col_name], row[station_col_name]))
                # Получаем список соседей для текущей точки
                neighbors_list = get_neighbours(points, row[station_col_name], radius)
                # Создаем объект Element для текущей точки и ее соседей, сохраняем в rows_neighbours
                rows_neighbours.append(Element(rowIndex, row[station_col_name], neighbors_list.copy(), row))
                is_neighbours_found = True

    for point_id in rows_neighbours:
        for point_id_x in point_id.neighbors_list:
            # Получаем соответствующие значения из value_df для каждого соседа
            tmp_df = value_df[
                (value_df[station_col_name] == point_id_x.id) & (value_df[year_col_name] == point_id.datarow[year_col_name]) & (
                            value_df[data_type_col_name] == point_id.datarow[data_type_col_name])].dropna()

            # Добавляем значения соседей в свойство dataframe объекта point_id
            point_id.dataframe = pd.concat([point_id.dataframe, tmp_df], ignore_index=True)
    return rows_neighbours


def get_neighbours(input_points, station_id, km):
    """
    Возвращает коллекцию соседних станций с номером id в радиусе km километров.

    :param input_points: (list) Список точек, в котором выполняется поиск соседей.
    :param station_id: (int) Номер станции, для которой выполняется поиск соседей.
    :param km: (float) Радиус поиска соседей в километрах.

    :return (list) Список соседних станций в радиусе km километров от станции с номером id.
    """

    ret_list = []  # Список для хранения соседних станций
    cur_point = None  # Переменная для хранения текущей точки

    for point in input_points:
        if point.id == station_id:
            cur_point = point
            break

    if cur_point is not None:
        for point in input_points:
            try:
                # Вычисляем расстояние между текущей точкой и остальными точками в списке points
                dist = geodesic((cur_point.x, cur_point.y), (point.x, point.y)).kilometers
                if 0 < dist <= km:
                    ret_list.append(point)
            except:
                pass

    return ret_list


def create_test_data_gaps(test_data_gaps, cols, gaps_count, start_index, end_index):
    """
    Создает тестовые пропуски в данных.

    :param test_data_gaps: (DataFrame) Исходный DataFrame с данными.
    :param cols: (list) Список колонок, в которых необходимо добавить пропуски.
    :param gaps_count: (int) Количество пропусков, которые нужно создать.
    :param start_index: (int) Начальный индекс строкового промежутка.
    :param end_index: (int) Конечный индекс строкового промежутка.

    :return (DataFrame) DataFrame с созданными тестовыми пропусками.
    """
    missing_indexes = []
    for col_index in cols:
        for x in range(gaps_count):
            while True:
                # Выбираем случайный индекс строки в заданном диапазоне
                rowid = randint(start_index, end_index)

                # Проверяем, что выбранный индекс не содержится в списке пропусков и существует в DataFrame
                if rowid not in missing_indexes and rowid in test_data_gaps.index:
                    # Устанавливаем значение пропуска (np.nan) для выбранной ячейки
                    test_data_gaps.loc[rowid, col_index] = np.nan

                    # Добавляем индекс строки в список пропусков
                    missing_indexes.append(rowid)

                    # Выходим из цикла, так как пропуск успешно создан
                    break

    return test_data_gaps


def recover_missing_values(rows_neighbours, test_data, station_col_name='station_Number', year_col_name='year', data_type_col_name='type_data'):
    """
    Восстанавливает отсутствующие значения в данных, используя различные модели.

    :param rows_neighbours: (list) Список строк, содержащих информацию о соседних значениях.
    :param test_data: (DataFrame) Тестовые данные о температуре.
    :param station_col_name: Наименование колонки, содержащей номера метеостанций.
    :param year_col_name: Наименование колонки, содержащей информацию о годе.
    :param data_type_col_name: Наименование колонки, содержащей информацию о типе данных.

    :return (DataFrame) DataFrame с коэффициентами и оценками для каждой модели. (Int) число возникших при выполнении ошибок.
    """

    # Создание пустого DataFrame для хранения коэффициентов и оценок
    coefs = pd.DataFrame()
    error_count = 0

    for gaprow in rows_neighbours:
        nan_list = gaprow.datarow.isnull()  # Булева маска пропущенных значений в строке

        for col in nan_list.index:
            if nan_list[col]:
                data = gaprow.dataframe[[col, 'y', 'x', 'z']]  # Извлечение данных из gaprow

                test_xy_df = test_data[(test_data[station_col_name] == gaprow.datarow[station_col_name]) & (test_data[year_col_name] == gaprow.datarow[year_col_name]) & (test_data[data_type_col_name] == gaprow.datarow[data_type_col_name])]
                X_train = data.drop(col, axis=1)
                y_train = data[col]
                X_test = test_xy_df[['y', 'x', 'z']].copy()
                y_test = test_xy_df[col].copy()

                # Преобразование колонок x, y и z в числовой формат
                X_test['x'] = pd.to_numeric(X_test['x'])
                X_test['y'] = pd.to_numeric(X_test['y'])
                X_test['z'] = pd.to_numeric(X_test['z'])
                X_train['x'] = pd.to_numeric(X_train['x'])
                X_train['y'] = pd.to_numeric(X_train['y'])
                X_train['z'] = pd.to_numeric(X_train['z'])

                if len(X_train) == 0 or len(y_train) == 0:
                    print("skipped:", gaprow.datarow[station_col_name])
                    continue
                try:
                    x = X_train['x'].values
                    y = X_train['y'].values
                    z = X_train['z'].values

                    points = np.column_stack((x, y, z))  # Создание массива точек из координат x, y и z
                    delaunay_pred = []  # Пустой список для хранения предсказанных значений методом интерполяции Delaunay

                    if len(points) >= 5:  # Если количество точек достаточно для построения треугольной сетки
                        tri = Delaunay(points)  # Создание треугольной сетки Delaunay
                        interp_model = LinearNDInterpolator(tri, y_train)  # Создание модели интерполяции
                        delaunay_pred = interp_model((X_test.x, X_test.y, X_test.z))  # Предсказание значений на тестовых данных
                    else:
                        delaunay_pred.append(999)  # Если количество точек недостаточно, добавляем значение 999 в качестве отсутствующего предсказания

                    krig = pk.UniversalKriging3D(X_train.x, X_train.y, X_train.z, y_train)  # Создание объекта универсальной кригинг-интерполяции
                    k = krig.execute("grid", X_test.x, X_test.y, X_test.z)  # Интерполяция значений на тестовых данных

                    est = sm.OLS(np.asarray(y_train, dtype=float), np.asarray(X_train, dtype=float), missing='drop')  # Создание модели линейной регрессии
                    est2 = est.fit()  # Обучение модели
                    regr_pred = est2.predict(X_test)  # Предсказание значений на тестовых данных

                    tmp_coef = pd.DataFrame({"coef_xyz": est2.params, "p_value_xyz": est2.pvalues}, index=X_train.columns)  # Создание DataFrame с коэффициентами и p-value

                    mean_pred = y_train.mean()  # Вычисление среднего значения y_train

                    regressor = KNeighborsRegressor(n_neighbors=len(X_train), weights='distance')  # Создание модели метода ближайших соседей
                    regressor.fit(X_train, y_train)  # Обучение модели
                    reconstructed_values = regressor.predict(X_test)  # Предсказание значений на тестовых данных

                except Exception as e:
                    error_count += 1
                    print("EXCEPTION: ", gaprow.datarow[station_col_name], str(e))
                    continue

                value_by_regr = y_test.values[0] - regr_pred.values[0]  # Вычисление разницы между фактическим и предсказанным значением с помощью линейной регрессии
                value_by_mean = y_test.values[0] - mean_pred  # Вычисление разницы между фактическим и средним значением y_train
                value_by_delaunay = y_test.values[0] - delaunay_pred[0] if delaunay_pred[0] < 999 else np.nan  # Вычисление разницы между фактическим и предсказанным значением с помощью интерполяции Delaunay
                value_by_krig = y_test.values[0] - k[0][0][0][0]  # Вычисление разницы между фактическим и предсказанным значением с помощью универсальной кригинг-интерполяции
                value_by_IDW = y_test.values[0] - reconstructed_values[0]  # Вычисление разницы между фактическим и предсказанным значением с помощью метода ближайших соседей

                tmp_coef["regr_pred_data"] = regr_pred.values[0]
                tmp_coef["mean_pred_data"] = mean_pred
                tmp_coef["delaunay_pred_data"] = delaunay_pred[0]
                tmp_coef["IDW_pred_data"] = reconstructed_values[0]
                tmp_coef["krig_pred_data"] = k[0][0][0][0]
                tmp_coef["real_data"] = y_test.values[0]

                tmp_coef["radius"] = 250
                tmp_coef["month"] = col

                tmp_coef['station_Number'] = gaprow.datarow['station_Number']

                tmp_coef["BE_by_regr"] = value_by_regr # ошибка для регрессии
                tmp_coef["AE_by_regr"] = abs(value_by_regr) # абсолютная ошибка для регрессии
                tmp_coef["RSE_by_regr"] = value_by_regr**2
                tmp_coef["APE_by_regr"] = abs(value_by_regr) / y_test.values[0]

                tmp_coef["BE_by_mean"] = value_by_mean # ошибка для метода среднего
                tmp_coef["AE_by_mean"] = abs(value_by_mean) # абсолютная ошибка для метода среднего
                tmp_coef["RSE_by_mean"] = value_by_mean**2
                tmp_coef["APE_by_mean"] = abs(value_by_mean) / y_test.values[0]

                tmp_coef["BE_by_delaunay"] = value_by_delaunay # ошибка для
                tmp_coef["AE_by_delaunay"] = abs(value_by_delaunay) # абсолютная ошибка для
                tmp_coef["RSE_by_delaunay"] = value_by_delaunay**2
                tmp_coef["APE_by_delaunay"] = abs(value_by_delaunay) / y_test.values[0]

                tmp_coef["BE_by_IDW"] = value_by_IDW # ошибка для IDW
                tmp_coef["AE_by_IDW"] = abs(value_by_IDW) # абсолютная ошибка для IDW
                tmp_coef["RSE_by_IDW"] = value_by_IDW**2
                tmp_coef["APE_by_IDW"] = abs(value_by_IDW) / y_test.values[0]

                tmp_coef["BE_by_krig"] = value_by_krig # ошибка для кригинга
                tmp_coef["AE_by_krig"] = abs(value_by_krig) # абсолютная ошибка для кригинга
                tmp_coef["RSE_by_krig"] = value_by_krig**2
                tmp_coef["APE_by_krig"] = abs(value_by_krig) / y_test.values[0]

                coefs = pd.concat([coefs, tmp_coef])
    return coefs, error_count


def filling_gaps_models_test(test_data, cols, gaps_count, start_index, end_index, start_year, end_year, year_col='year', radius=250):
    """
    Функция запуска тестирования моделей

    :param test_data: (DataFrame) Тестовый набор данных
    :param cols: (list) Список колонок, в которых необходимо добавить пропуски.
    :param gaps_count: Количество пропусков, которые нужно создать.
    :param start_index: Начальный индекс строкового промежутка.
    :param end_index: Конечный индекс строкового промежутка.
    :param start_year: Год начала периода.
    :param end_year: Год окончания периода.
    :param year_col: Название столбца хранящего год.
    :param radius: Радиус для поиска соседних метеостанций.

    :return (DataFrame) DataFrame с коэффициентами и оценками для каждой модели. (Int) число возникших при выполнении ошибок.
    """
    test_gaps_indexes = []  # Номера строк с пропусками

    test_data = test_data.dropna()
    test_data_gaps = test_data
    test_data_gaps = create_test_data_gaps(test_data_gaps, cols, gaps_count, start_index, end_index)

    rows_neighbours = get_neighbour_elements(test_data_gaps[(test_data_gaps[year_col] > start_year) & (test_data_gaps[year_col] < end_year)], test_data, radius)

    return recover_missing_values(rows_neighbours, test_data)
