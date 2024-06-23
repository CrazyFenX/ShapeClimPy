import numpy as np
import pandas as pd
from tqdm import tqdm
import datetime
import xarray as xr


def crop_netcdf_dataset(input_filename, output_filename, lon_min, lon_max, lat_min, lat_max):
    """
    Функция обрезки NetCDF файла

    :param input_filename: путь к NetCDF исходному файлу
    :param output_filename: путь для сохранения итогового файла
    :param lon_min: минимальная долгота
    :param lon_max: максимальная долгота
    :param lat_min: минимальная широта
    :param lat_max: максимальная широта
    """

    # Загрузка исходного файла NetCDF
    data = xr.open_dataset(input_filename)

    # Нахождение индексов соответствующих координатам
    lon_indices = (data['lon'] >= lon_min) & (data['lon'] <= lon_max)
    lat_indices = (data['lat'] >= lat_min) & (data['lat'] <= lat_max)

    # Обрезка данных по заданному квадрату
    data_cropped = data.sel(lon=lon_indices, lat=lat_indices)

    # Сохранение обрезанных данных в новый файл NetCDF
    data_cropped.to_netcdf(output_filename)


def resample_tas_netcdf_dataset(input_filename, output_filename):
    """
    Функция пересчета значений переменной 'tas' NetCDF файла

    :param input_filename: путь к NetCDF исходному файлу
    :param output_filename: путь для сохранения итогового файла
    """
    data = xr.open_dataset(input_filename)

    # Преобразуем значения температуры из Кельвинов в градусы Цельсия
    new_values = data['tas'] - 273.15

    # Создаем новый NetCDF файл
    new_dataset = xr.Dataset(
      data_vars={'tas': new_values},
      coords=data.coords
    )

    # Сохраняем новый NetCDF файл
    new_dataset.to_netcdf(output_filename)


def resample_pr_netcdf_dataset(input_file_name, output_file_name):
    """
    Функция пересчета значений переменной 'pr' NetCDF файла

    :param input_file_name: путь к NetCDF исходному файлу
    :param output_file_name: путь для сохранения итогового файла
    """
    data = xr.open_dataset(input_file_name)

    # данные в pr - kg m-2 s-1 - - килограмм на метр квадратный в секунду
    # перевод в мм/сутки=мм/24часа = x*86400 = x*24*60*60
    new_values = data['pr'] * 86400

    # Создаем новый NetCDF файл
    new_dataset = xr.Dataset(
      data_vars={'pr': new_values},
      coords=data.coords
    )

    # Сохраняем новый NetCDF файл
    new_dataset.to_netcdf(output_file_name)


def concatenate_netcdf_dataset(input_file_paths, output_filename):
    """
    Функция объединения NetCDF файлов

    :param input_file_paths: Список путей к NetCDF файлам, которые нужно объединить
    :param output_filename: путь для сохранения объединенного файла
    """
    datasets = [xr.open_dataset(file_path) for file_path in input_file_paths]

    # Выполняем конкатенацию по временной оси
    combined_dataset = xr.concat(datasets, dim='time')

    # Сохраняем объединенный NetCDF файл
    combined_dataset.to_netcdf(output_filename)


def get_value_by_netcdf_data(nc_data, start_date, year, month, lat, lon, target_property):
    """
        Функция восстановления пропущенных значений на основе данных растра

        :param nc_data: набор данных растра
        :param start_date: начальная дата периода данных растра
        :param year: год
        :param month: месяц
        :param lat: широта
        :param lon: долгота
        :param target_property: целевой атрибут сравнения (tas, pr)

        :return значение данных в пикселе растра, соответствующего координатам и периоду
    """
    # Получение значений широты и долготы из NetCDF файла
    lats = nc_data.variables['lat'][:]
    lons = nc_data.variables['lon'][:]

    if month + 1 == 12:
        date = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
    else:
        date = datetime.datetime(year, month + 2, 1) - datetime.timedelta(days=1)

    time_index = (date.year - start_date.year) * 12 + (date.month - start_date.month)

    lat_index = np.abs(lats - lat).argmin()
    lon_index = np.abs(lons - lon).argmin()

    nc_temp = nc_data.variables[target_property][time_index, lat_index, lon_index]

    return nc_temp


def recover_gaps_by_netcdf_data(data_with_gaps, nc_data, start_date, months, target_property):
    """
        Функция восстановления пропущенных значений на основе данных растра

        :param data_with_gaps: набор данных с пропусками
        :param nc_data: набор данных растра
        :param start_date: набор данных с пропусками
        :param months: список месяцев для агрегации
        :param target_property: целевой атрибут сравнения (tas, pr)

        :return набор метеорологических данных с присоединенными данными растра
    """
    # Получение значений широты и долготы из NetCDF файла
    lats = nc_data.variables['lat'][:]
    lons = nc_data.variables['lon'][:]

    data_with_gaps_tmp = data_with_gaps.copy()

    for index, row in tqdm(data_with_gaps_tmp.iterrows(), total=len(data_with_gaps_tmp)):
        year = row['year']
        for month in range(12):
            if pd.isna(row[months[month]]):
                lat = row['Latitude']
                lon = row['Longitude']

                if month + 1 == 12:
                    date = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
                else:
                    date = datetime.datetime(year, month + 2, 1) - datetime.timedelta(days=1)

                time_index = (date - start_date).days
                time_index = (date.year - start_date.year) * 12 + (date.month - start_date.month)

                lat_index = np.abs(lats - lat).argmin()
                lon_index = np.abs(lons - lon).argmin()

                nc_temp = nc_data.variables[target_property][time_index, lat_index, lon_index]

                data_with_gaps_tmp.loc[index, months[month]] = nc_temp

    return data_with_gaps_tmp


def recover_gaps_per_year(data, months, target_property, agregate_func='sum'):
    """
        Функция восстановления значений за год

        :param data: набор данных метеорологических станций
        :param months: список месяцев для агрегации
        :param target_property: целевой атрибут сравнения (tas, pr)
        :param agregate_func: функция агрегации годовых значений (sum/mean)

        :return набор метеорологических данных с присоединенными данными растра
    """
    data_tmp = data.copy()

    for index, row in tqdm(data_tmp.iterrows(), total=len(data_tmp)):
        year = row['year']

        per_year = 0

        if agregate_func == "sum":
            per_year = row[months].sum()
        elif agregate_func == "mean":
            per_year = row[months].mean()

    data_tmp.loc[index, target_property] = per_year

    return data_tmp


def join_netcdf_data(data_csv, nc_data, start_date, months, target_property):
    """
        Функция объединения NetCDF файлов

        :param data_csv: набор данных метеорологических станций
        :param nc_data: набор растровых данных
        :param start_date: начальная дата сравнения
        :param months: список месяцев
        :param target_property: целевой атрибут сравнения (tas, pr)

        :return набор метеорологических данных с присоединенными данными растра
    """
    # Получение значений широты и долготы из NetCDF файла
    lats = nc_data.variables['lat'][:]
    lons = nc_data.variables['lon'][:]

    # Создание пустого DataFrame для хранения разницы данных
    diff_df = pd.DataFrame(columns=['year', 'station'])

    for index, row in tqdm(data_csv.iterrows(), total=len(data_csv)):
        year = row['year']

        # Создание нового DataFrame с данными для добавления
        new_data = pd.DataFrame({'year': [year], "station": [row['station']]})

        for month in range(12):
            # month = 1
            lat = row['Latitude']
            lon = row['Longitude']

            # Вычисление индекса времени для данного года и месяца
            if month + 1 == 12:
              date = datetime.datetime(year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
              date = datetime.datetime(year, month + 2, 1) - datetime.timedelta(days=1)

            time_index = (date - start_date).days
            time_index = (date.year - start_date.year) * 12 + (date.month - start_date.month)

            # Нахождение ближайшего значения широты и долготы в NetCDF файле
            lat_index = np.abs(lats - lat).argmin()
            lon_index = np.abs(lons - lon).argmin()

            # Получение данных в точке lat, lon из NetCDF файла за тот же год и месяц
            nc_temp = nc_data.variables[target_property][time_index, lat_index, lon_index]

            new_data[months[month]] = nc_temp

        # Объединение DataFrame diff_df и new_data
        diff_df = pd.concat([diff_df, new_data], ignore_index=True)
    return diff_df
