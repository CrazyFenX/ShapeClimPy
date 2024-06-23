import numpy as np
from pykrige.ok import OrdinaryKriging
from netCDF4 import Dataset

# Задание размера пикселя и размера сетки значений растра
# pixel_size = 50
# grid_size = 100

# Задание координат точек
# lon = np.array([100.0, 200.0, 300.0])
# lat = np.array([10.0, 20.0, 30.0])

# Задание значений в точках
# values = np.array([5.0, 10.0, 15.0])

# Задание координаты, в которой нужно выполнить кригинг
# new_lon = 250.0
# new_lat = 15.0


def kriging(lon, lat, values, new_lon, new_lat, grid_size, output_filename='output.nc'):
    # Создание экземпляра класса OrdinaryKriging и выполнение кригинга
    ok = OrdinaryKriging(lon, lat, values)
    z, ss = ok.execute('grid', [new_lon], [new_lat])

    # Создание пустого растра
    raster = np.zeros((grid_size, grid_size))

    # Заполнение растра значениями из сетки значений
    raster[grid_size, grid_size] = z[0]

    # Запись растра в NetCDF файл
    nc_filename = output_filename
    nc_file = Dataset(nc_filename, 'w', format='NETCDF4')

    # Создание размерностей
    nc_file.createDimension('x', grid_size)
    nc_file.createDimension('y', grid_size)

    # Создание переменной для растра
    raster_var = nc_file.createVariable('raster', 'f4', ('x', 'y'))
    raster_var[:] = raster

    nc_file.close()
