import os
from osgeo import gdal, gdal_array
import numpy as np


def merge_hgt_files(hgt_folder, output_file, cols_count, rows_count):
    """
    Объединяет HGT-файлы в один файл GeoTIFF с объединенными данными в одном канале.

    :param hgt_folder -- путь к папке с HGT-файлами
    :param output_file -- путь для сохранения объединенного файла GeoTIFF
    :param cols_count -- количество фрагментов растра по горизонтали
    :param rows_count -- количество фрагментов растра по вертикали

    Example:
        hgt_folder = '/content/'
        output_file = '/content/I45.tif'
        cols_count = 6
        rows_count = 4
        merge_hgt_files(hgt_folder, output_file, cols_count, rows_count)
    """
    # Получаем список HGT-файлов в папке и сортируем их по имени
    hgt_files = sorted([file for file in os.listdir(hgt_folder) if file.endswith('.hgt')])

    if len(hgt_files) == 0:
        raise ValueError("HGT-файлы не найдены в указанной папке")

    hgt_files = [file for file in os.listdir(hgt_folder) if file.endswith('.hgt')]

    print(int(hgt_files[0][1:3]), int(hgt_files[0][4:7]))

    sorted_files = sorted(hgt_files, key=lambda x: (-int(x[1:3]), int(x[4:7])))

    for file_name in sorted_files:
        print(file_name)

    # Читаем первый HGT-файл, чтобы получить информацию о размере и геопривязке
    first_hgt_file = os.path.join(hgt_folder, sorted_files[0])

    dataset = gdal.Open(first_hgt_file, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize
    print(rows, cols)
    print(rows*rows_count, cols*cols_count)
    geotransform = dataset.GetGeoTransform()
    data_type = dataset.GetRasterBand(1).DataType

    # Создаем выходной массив для объединенных данных
    merged_data = np.zeros((rows*rows_count, cols*cols_count), dtype=np.float32)
    start_col = 0
    start_row_1 = rows*rows_count

    # Стартовая точка
    i1 = 31
    j1 = 83

    # Объединяем HGT-файлы в один канал с использованием широты и долготы
    for hgt_file in hgt_files:
        hgt_path = os.path.join(hgt_folder, hgt_file)

        numbers = hgt_file.split(".")[0].split("N")[1].split("E")

        i = i1 - int(numbers[0])
        j = int(numbers[1]) - j1

        dataset = gdal.Open(hgt_path, gdal.GA_ReadOnly)
        data = dataset.GetRasterBand(1).ReadAsArray()

        # Получаем геопривязку для текущего HGT-файла
        hgt_geotransform = dataset.GetGeoTransform()
        hgt_cols = dataset.RasterXSize
        hgt_rows = dataset.RasterYSize

        # Вычисляем индексы пикселей в выходном массиве, соответствующие текущему HGT-файлу
        end_col = j * hgt_cols

        start_row = start_row_1 + i * hgt_rows

        start_col = end_col - hgt_cols
        end_row = start_row + hgt_rows

        # Объединяем данные текущего HGT-файла в выходной массив
        merged_data[start_row:end_row, start_col:end_col] = data

    # Создаем драйвер для GeoTIFF
    driver = gdal.GetDriverByName("GTiff")

    # Создаем выходной файл GeoTIFF
    output_dataset = driver.Create(output_file, rows*cols_count, cols*rows_count, 1, data_type)

    # Устанавливаем геопривязку для выходного файла
    output_dataset.SetGeoTransform(geotransform)

    # Записываем объединенные данные в выходной канал
    output_dataset.GetRasterBand(1).WriteArray(merged_data)

    output_dataset.FlushCache()