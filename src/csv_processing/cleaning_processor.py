import pandas as pd
import numpy as np


def cleaning(df, min_value, max_value, columns_to_check=['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'per_year'], errors_codes=[999, -999, 9999], is_normal=True):
    """
    Функция очищает указанные столбцы в pandas DataFrame, заменяя значения,
    которые соответствуют определенным критериям, на NaN (Not a Number).

    Параметры:
    - df (pandas.DataFrame): Входной DataFrame, который необходимо очистить.
    - columns_to_check (list): Список имен столбцов, которые необходимо очистить.
    - errors_codes (list): Список кодов ошибок, которые необходимо заменить на NaN.
    - min_value (float): Минимальное допустимое значение для столбцов.
    - max_value (float): Максимальное допустимое значение для столбцов.
    - is_normal (bool, необязательный): Если True, значения за пределами 3-сигм
    будут заменены на NaN. По умолчанию True.

    Возвращает:
    pandas.DataFrame: Очищенный DataFrame.
    """
    for column in columns_to_check:
        # Преобразование столбца в числовой тип данных
        df[column] = pd.to_numeric(df[column], errors='coerce')

        # Замена значений, которые больше max_value и меньше min_value
        df[column] = np.where(df[column].isin(errors_codes), np.nan, df[column])

        # Замена значений, которые находятся в списке ошибок errors_codes
        df.loc[(df[column] > max_value) | (df[column] < min_value), column] = np.nan

        if is_normal:
            # Вычисление среднего значения и стандартного отклонения для каждого столбца
            means = df[columns_to_check].mean()
            stds = df[columns_to_check].std()

            # Определение границ трех сигм для каждого столбца
            lower_bounds = means - 3 * stds
            upper_bounds = means + 3 * stds

            # Замена значений, которые находятся за пределами трех сигм, на NaN
            df[columns_to_check] = df[columns_to_check].where(
                (df[columns_to_check] >= lower_bounds) &
                (df[columns_to_check] <= upper_bounds),
                other=np.nan
            )
    return df
