# тестированиее методов восстановления
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import pykrige as pk
from enum import Enum, auto


def recover_missing_values(rows_neighbours, test_data, interpolation_type):
    """
    Восстановление отсутствующих значений в данных, используя различные модели.

    :param rows_neighbours:(list) Список строк, содержащих информацию о соседних значениях
    :param test_data:(DataFrame) Тестовые данные о температуре
    :param interpolation_type:(InterpolationStrategyType) Тип интерполяционной модели

    :return (DataFrame) DataFrame с восстановленными значениями.
    """

    # Создание пустого DataFrame для хранения восстановленных значений
    recovered_data = test_data.copy()

    error_count = 0

    for gaprow in rows_neighbours:
        nan_list = gaprow.datarow.isnull()  # Булева маска пропущенных значений в строке

        for col in nan_list.index:
            if nan_list[col]:
                data = gaprow.dataframe[[col, 'y', 'x', 'z']]  # Извлечение данных из gaprow

                test_xy_df = test_data[(test_data['station_Number'] == gaprow.datarow['station_Number']) & (test_data['year'] == gaprow.datarow['year']) & (test_data['type_data'] == gaprow.datarow['type_data'])]
                x_train = data.drop(col, axis=1)
                y_train = data[col]
                x_test = test_xy_df[['y', 'x', 'z']].copy()
                y_test = test_xy_df[col].copy()

                # Преобразование колонок x, y и z в числовой формат
                x_test['x'] = pd.to_numeric(x_test['x'])
                x_test['y'] = pd.to_numeric(x_test['y'])
                x_test['z'] = pd.to_numeric(x_test['z'])
                x_train['x'] = pd.to_numeric(x_train['x'])
                x_train['y'] = pd.to_numeric(x_train['y'])
                x_train['z'] = pd.to_numeric(x_train['z'])

                if len(x_train) == 0 or len(y_train) == 0:
                    print("skipped:", gaprow.datarow['station_Number'])
                    continue
                try:
                    interpolator = Interpolator(MeanInterpolation())
                    interpolator = Interpolator(interpolator.get_strategy_by_type(interpolation_type))
                    result = interpolator.restore_data(y_train, x_train, x_test)
                    recovered_data.loc[(recovered_data['station_Number'] == gaprow.datarow['station_Number']) & (
                        recovered_data['year'] == gaprow.datarow['year']) & (
                        recovered_data['type_data'] == gaprow.datarow['type_data']), col] = result
                except Exception as e:
                    error_count += 1
                    print("EXCEPTION: ", gaprow.datarow['station_Number'], str(e))
                    continue

    return recovered_data, error_count


def try_all_interpolation_models(rows_neighbours, test_data):
    """
    Тестовое восстановление отсутствующих значений в данных, используя различные модели

    :param: Rows_neighbours (list) Список строк, содержащих информацию о соседних значениях
    :param: Test_data (DataFrame) Тестовые данные о температуре.

    :return: (DataFrame) DataFrame с коэффициентами и оценками для каждой модели.

    """

    # Создание пустого DataFrame для хранения коэффициентов и оценок
    coefs = pd.DataFrame()
    error_count = 0

    for gaprow in rows_neighbours:
        nan_list = gaprow.datarow.isnull()  # Булева маска пропущенных значений в строке

        for col in nan_list.index:
            if nan_list[col]:
                data = gaprow.dataframe[[col, 'y', 'x', 'z']]  # Извлечение данных из gaprow

                test_xy_df = test_data[(test_data['station_Number'] == gaprow.datarow['station_Number']) & (test_data['year'] == gaprow.datarow['year']) & (test_data['type_data'] == gaprow.datarow['type_data'])]
                x_train = data.drop(col, axis=1)
                y_train = data[col]
                x_test = test_xy_df[['y', 'x', 'z']].copy()
                y_test = test_xy_df[col].copy()

                # Преобразование колонок x, y и z в числовой формат
                x_test['x'] = pd.to_numeric(x_test['x'])
                x_test['y'] = pd.to_numeric(x_test['y'])
                x_test['z'] = pd.to_numeric(x_test['z'])
                x_train['x'] = pd.to_numeric(x_train['x'])
                x_train['y'] = pd.to_numeric(x_train['y'])
                x_train['z'] = pd.to_numeric(x_train['z'])

                if len(x_train) == 0 or len(y_train) == 0:
                    print("skipped:", gaprow.datarow['station_Number'])
                    continue
                try:
                    # Создание модели линейной регрессии
                    est = sm.OLS(np.asarray(y_train, dtype=float), np.asarray(x_train, dtype=float), missing='drop')
                    est2 = est.fit()  # Обучение модели

                    # Создание DataFrame с коэффициентами и p-value
                    tmp_coef = pd.DataFrame({"coef_xyz": est2.params, "p_value_xyz": est2.pvalues}, index=x_test.columns)

                    interpolator = Interpolator(LinearInterpolation())
                    all_results = interpolator.try_all_strategies(y_train, x_train, x_test)

                except Exception as e:
                    error_count += 1
                    print("EXCEPTION: ", gaprow.datarow['station_Number'], str(e))
                    continue

                # Вычисление разницы между фактическим и средним значением y_train
                value_by_mean = y_test.values[0] - all_results[0]
                # Вычисление разницы между фактическим и предсказанным значением с помощью линейной регрессии
                value_by_regr = y_test.values[0] - all_results[1]
                # Вычисление разницы между фактическим и предсказанным значением с помощью интерполяции Delaunay
                value_by_delaunay = y_test.values[0] - all_results[2]
                # Вычисление разницы между фактическим и предсказанным значением с помощью метода ближайших соседей
                value_by_IDW = y_test.values[0] - all_results[3]
                # Вычисление разницы между фактическим и предсказанным значением с помощью универсального кригинга
                value_by_krig = y_test.values[0] - all_results[4]

                tmp_coef["mean_pred_data"] = all_results[0]
                tmp_coef["regr_pred_data"] = all_results[1]
                tmp_coef["delaunay_pred_data"] = all_results[2]
                tmp_coef["IDW_pred_data"] = all_results[3]
                tmp_coef["krig_pred_data"] = all_results[4]
                tmp_coef["real_data"] = y_test.values[0]

                tmp_coef["radius"] = 250
                tmp_coef["month"] = col

                tmp_coef['station_Number'] = gaprow.datarow['station_Number']

                tmp_coef["BE_by_mean"] = value_by_mean  # ошибка для метода среднего
                tmp_coef["AE_by_mean"] = abs(value_by_mean)  # абсолютная ошибка для метода среднего
                tmp_coef["RSE_by_mean"] = value_by_mean**2
                tmp_coef["APE_by_mean"] = abs(value_by_mean) / y_test.values[0]

                tmp_coef["BE_by_regr"] = value_by_regr  # ошибка для регрессии
                tmp_coef["AE_by_regr"] = abs(value_by_regr)  # абсолютная ошибка для регрессии
                tmp_coef["RSE_by_regr"] = value_by_regr**2
                tmp_coef["APE_by_regr"] = abs(value_by_regr) / y_test.values[0]

                tmp_coef["BE_by_delaunay"] = value_by_delaunay  # ошибка для
                tmp_coef["AE_by_delaunay"] = abs(value_by_delaunay)  # абсолютная ошибка для
                tmp_coef["RSE_by_delaunay"] = value_by_delaunay**2
                tmp_coef["APE_by_delaunay"] = abs(value_by_delaunay) / y_test.values[0]

                tmp_coef["BE_by_IDW"] = value_by_IDW  # ошибка для IDW
                tmp_coef["AE_by_IDW"] = abs(value_by_IDW)  # абсолютная ошибка для IDW
                tmp_coef["RSE_by_IDW"] = value_by_IDW**2
                tmp_coef["APE_by_IDW"] = abs(value_by_IDW) / y_test.values[0]

                tmp_coef["BE_by_krig"] = value_by_krig  # ошибка для кригинга
                tmp_coef["AE_by_krig"] = abs(value_by_krig)  # абсолютная ошибка для кригинга
                tmp_coef["RSE_by_krig"] = value_by_krig**2
                tmp_coef["APE_by_krig"] = abs(value_by_krig) / y_test.values[0]

                coefs = pd.concat([coefs, tmp_coef])

    return coefs, error_count


class InterpolationStrategyBase:
    """Базовый класс для стратегий интерполяции"""
    def interpolate(self, y_train, x_train, x_test):
        pass


class MeanInterpolation(InterpolationStrategyBase):
    """Стратегия интерполяции методом среднего M0 (Mean)"""
    def interpolate(self, y_train, x_train, x_test):
        return y_train.mean()


class LinearInterpolation(InterpolationStrategyBase):
    """Стратегия линейной интерполяции M1 (LM)"""
    def interpolate(self, y_train, x_train, x_test):
        est = sm.OLS(np.asarray(y_train, dtype=float), np.asarray(x_train, dtype=float),
                     missing='drop')  # Создание модели линейной регрессии
        est2 = est.fit()  # Обучение модели
        regr_pred = est2.predict(x_test)  # Предсказание значений на тестовых данных
        return regr_pred.values[0]


class IdwInterpolation(InterpolationStrategyBase):
    """Стратегия интерполяции обратно взвешенных расстояний M2 (IDW)"""
    def interpolate(self, y_train, x_train, x_test):
        # Создание модели метода ближайших соседей
        regressor = KNeighborsRegressor(n_neighbors=len(x_train), weights='distance')
        # Обучение модели
        regressor.fit(x_train, y_train)
        # Предсказание значений на тестовых данных
        reconstructed_values = regressor.predict(x_test)

        return reconstructed_values[0]


class DelaunayInterpolation(InterpolationStrategyBase):
    """Стратегия интерполяции триангуляцией Делоне M3 (Delaunay)"""
    def interpolate(self, y_train, x_train, x_test):
        _x = x_train['x'].values
        _y = x_train['y'].values
        _z = x_train['z'].values

        points = np.column_stack((_x, _y, _z))  # Создание массива точек из координат x, y и z
        delaunay_pred = []  # Пустой список для хранения предсказанных значений методом интерполяции Delaunay

        if len(points) >= 5:  # Если количество точек достаточно для построения треугольной сетки
            tri = Delaunay(points)  # Создание треугольной сетки Delaunay
            interp_model = LinearNDInterpolator(tri, y_train)  # Создание модели интерполяции
            delaunay_pred = interp_model((x_test.x, x_test.y, x_test.z))  # Предсказание значений на тестовых данных
        else:
            # Если количество точек недостаточно, добавляем значение 999 в качестве отсутствующего предсказания
            delaunay_pred.append(999)

        return delaunay_pred[0] if delaunay_pred[0] < 999 else np.nan


class UniversalKrigingInterpolation(InterpolationStrategyBase):
    """Стратегия интерполяции универсальным кригингом M4 (UK)"""
    def interpolate(self, y_train, x_train, x_test):
        # Создание объекта универсальной кригинг-интерполяции
        krig = pk.UniversalKriging3D(x_train.x, x_train.y, x_train.z, y_train)
        k = krig.execute("grid", x_test.x, x_test.y, x_test.z)  # Интерполяция значений на тестовых данных
        return k[0][0][0][0]


class InterpolationStrategyType(Enum):
    MEAN = auto()
    LINEAR = auto()
    IDW = auto()
    DELAUNAY = auto()
    UNIVERSAL_KRIGING = auto()


class Interpolator:
    """Контекст, использующий стратегию интерполяции"""

    all_interpolation_types = [
        MeanInterpolation,
        LinearInterpolation,
        IdwInterpolation,
        DelaunayInterpolation,
        UniversalKrigingInterpolation
    ]

    def __init__(self, strategy):
        self.strategy = strategy

    def restore_data(self, y_train, x_train, x_test):
        return self.strategy.interpolate(y_train, x_train, x_test)

    def try_all_strategies(self, y_train, x_train, x_test):
        results = []
        for strategy_type in self.all_interpolation_types:
            strategy = strategy_type()
            results.append(self.restore_data(y_train, x_train, x_test))
        return results

    def try_strategies(self, interpolation_types, y_train, x_train, x_test):
        results = []
        for strategy_type in interpolation_types:
            strategy = strategy_type()
            results.append(self.restore_data(y_train, x_train, x_test))
        return results

    def get_strategy_by_type(self, strategy_type: InterpolationStrategyType):
        strategy_classes = {
            InterpolationStrategyType.MEAN: MeanInterpolation,
            InterpolationStrategyType.LINEAR: LinearInterpolation,
            InterpolationStrategyType.IDW: IdwInterpolation,
            InterpolationStrategyType.DELAUNAY: DelaunayInterpolation,
            InterpolationStrategyType.UNIVERSAL_KRIGING: UniversalKrigingInterpolation
        }
        strategy_class = strategy_classes.get(strategy_type)
        if strategy_class:
            return strategy_class()
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")
