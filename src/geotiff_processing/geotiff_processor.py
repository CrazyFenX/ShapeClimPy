import os
import geopandas as gpd
import pandas as pd
from rasterstats import zonal_stats
import datetime
import rasterio
from tqdm import tqdm


def get_zonal_stats_std_mean(zones, directory, start_year, end_year, months, data_type, out_col_name, join_col_name):
    gdf = gpd.read_file(zones)

    gdf_out = pd.DataFrame()

    # Перебор всех файлов в директории
    for filename in os.listdir(directory):
        if filename.endswith(f'.tif'):
            # Определение года и месяца из имени файла
            year = int(filename.split('_')[2].split('-')[0])
            month = int(filename.split('_')[2].split('-')[1])

            # Проверка, что год находится в заданном диапазоне
            if start_year <= year <= end_year:
                # Вычисление статистик для среза данных
                stats_year = zonal_stats(gdf, os.path.join(directory, filename), stats=["std", "mean"])
                stats = []

                # Добавление года в статистики
                for stat in stats_year:
                    stat["year"] = year

                # Добавление статистик в список
                stats.extend(stats_year)

                # Создание GeoDataFrame из списка статистик
                gdf_stats = pd.DataFrame(stats)

                # Присоединение столбца с годом к GeoDataFrame
                gdf_stats["year"] = gdf_stats["year"].astype(int)
                gdf_stats[join_col_name] = list(gdf[join_col_name]) * int(len(gdf_stats)/len(gdf))
                gdf_stats = gdf_stats.rename(columns={'std': f'{out_col_name}_{months[month - 1]}_std', 'mean': f"{out_col_name}_{months[month - 1]}"})

                # Присоединение статистик к исходному GeoDataFrame с использованием join_col_name
                gdf_out = pd.concat([gdf_out, gdf_stats])

    return gdf_out


# Функция для чтения данных из GeoTIFF растра в точке с заданными координатами
def read_geotiff_data(file_path, lon, lat):
    with rasterio.open(file_path) as src:
        lon_index, lat_index = src.index(lon, lat)
        array = src.read(1)
        data = array[lat_index, lon_index]
        return data


def join_geotiff_data(directory, months, dataframe, datatype):
    # Создание нового DataFrame для хранения объединенных данных
    output_dataframe = dataframe.copy()

    # Обход каждой станции
    for index, row in dataframe.iterrows():
        lon = row['Longitude']
        lat = row['Latitude']

        # Обход каждого месяца
        for month in months:
            # Формирование имени файла GeoTIFF для текущего месяца
            tiff_file = os.path.join(directory, f'wc2.1_2.5m_{datatype}_1960-{datetime.datetime.strptime(month, "%b").strftime("%m")}.tif')

            # Чтение данных из GeoTIFF растра в точке с заданными координатами
            tiff_data = read_geotiff_data(tiff_file, lon, lat)

            # Обновление соответствующего атрибута в DataFrame
            output_dataframe.at[index, f'{month}_wc'] = tiff_data

    # Вывод результата
    print(output_dataframe)


def split_multiband_geotiff(input_file, output_prefix):
    """
    Разделяет многоканальный GeoTIFF-файл на отдельные одноканальные GeoTIFF-файлы.

    :param (str) input_file : Путь к входному многоканальному GeoTIFF-файлу
    :param (str) output_prefix: Префикс для имен выходных одноканальных GeoTIFF-файлов
    """
    with rasterio.open(input_file) as src:
        for i in tqdm(range(src.count)):
            output_file = f"{output_prefix}_{i + 1}.tif"

            profile = src.profile
            profile.update(count=1, dtype=src.dtypes[i])

            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(src.read(i + 1), 1)


def get_zonal_stats_std_mean_1(zones, directory, start_year, end_year, months, data_type, model_type, ssp_type, out_col_name, join_col_name):
    # Инициализация пустого списка для хранения статистик
    gdf = gpd.read_file(zones)

    gdf_out = pd.DataFrame()

    for month in tqdm(range(1, 13)):
        # Формирование строки с именем файла для текущего года и месяца
        filename = f'{directory}wc2.1_2.5m_{data_type}_{model_type}_{ssp_type}_{start_year}-{end_year}/wc2.1_2.5m_{data_type}_{model_type}_{ssp_type}_{start_year}-{end_year}_{month}.tif'

        # Вычисление статистик для среза данных
        stats_year = zonal_stats(gdf, filename, stats=["std", "mean"])
        stats = []

        # Добавление года в статистики
        for stat in stats_year:
            stat["model"] = f"{model_type}"
            stat["years"] = f"{start_year} - {end_year}"

        # Добавление статистик в список
        stats.extend(stats_year)

        # Создание DataFrame из списка статистик
        gdf_stats = pd.DataFrame(stats)

        # Присоединение столбца с годом к DataFrame со статистиками
        gdf_stats[join_col_name] = list(gdf[join_col_name]) * int(len(gdf_stats) / len(gdf))
        # Присоединение статистик к исходному DataFrame
        gdf_stats[join_col_name] = gdf_stats[join_col_name]
        gdf_stats = gdf_stats.rename(columns={'std': f'{out_col_name}_{months[month - 1]}_std',
                                              'mean': f"{out_col_name}_{months[month - 1]}"})

        if month == 1:
            gdf_out = gdf_stats
        else:
            # Присоединение статистик к исходному DataFrame с использованием join_col_name
            gdf_out = gdf_out.merge(gdf_stats, on=["model", "years", join_col_name], how="outer")

    return gdf_out
