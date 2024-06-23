import os
import pandas as pd
import numpy as np
from scipy import stats


class ModelsComparer:
    def __init__(self, models):
        self.models = models
        self.max_values = []
        self.median_values = []
        self.min_values = []
        self.max_models = []
        self.median_models = []
        self.min_models = []
        self.months = []
        self.zones = []
        self.confidence_intervals = []

    def get_result_df(self):
        return pd.DataFrame({'zone': self.zones, 'month': self.months, 'min_value': self.min_values, 'min_model': self.min_models, 'median_value': self.median_values, 'median_model': self.median_models, 'max_value': self.max_values, 'max_model': self.max_models, 'confidence_intervals': self.confidence_intervals})

    def print_result_df(self):
        tmp = self.get_result_df()
        print(tmp)

    def models_compare(self, folder, project_dir, data_type, ssp_type, start_year, end_year):
        """
        Функция сравнения различных моделей указанного SSP-сценария
        :param folder -- путь к папке с HGT-файлами
        :param project_dir -- путь к общей папке
        :param data_type -- тип данных
        :param ssp_type -- тип SSP-сценария
        :param start_year -- начальный год периода
        :param end_year -- конечный год периода
        remarks: параметры применяются в маске файлов моделей:
            "project_dir + "\\" + folder + f"\\wc2.1_2.5m_{data_type}_{model}_{ssp_type}_{start_year}-{end_year}.csv")
        """
        folder_path = os.path.abspath(folder)  # Получение абсолютного пути к папке

        full_df = pd.DataFrame()

        for model in self.models:
            tmp_df = pd.read_csv(project_dir + "\\" + folder + f"\\wc2.1_2.5m_{data_type}_{model}_{ssp_type}_{start_year}-{end_year}.csv")
            # Слияние DataFrame
            full_df = pd.concat([full_df, tmp_df])

        for month in full_df.columns[3:]:
            print("-", month)
            for zone_name in full_df['NAME_EN'].unique():
                print("---", zone_name)
                values = []
                values.extend(full_df[month][full_df['NAME_EN'] == zone_name].dropna().values)

                print(values)

                max_value = np.max(values)
                min_value = np.min(values)
                median_value = np.median(values)

                if len(full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == max_value)]["model"].values) > 1:
                    model_max = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == max_value)]["model"].values
                else:
                    model_max = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == max_value)]["model"].values[0]

                if len(full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == median_value)]["model"].values) > 1:
                    model_median = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == median_value)]["model"].values
                else:
                    model_median = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == median_value)]["model"].values[0]

                if len(full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == min_value)]["model"].values) > 1:
                    model_min = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == min_value)]["model"].values
                else:
                    model_min = full_df[(full_df['NAME_EN'] == zone_name) & (full_df[month] == min_value)]["model"].values[0]

                confidence_interval = stats.t.interval(0.95, len(values) - 1, loc=np.mean(values), scale=stats.sem(values))

                self.max_values.append(max_value)
                self.median_values.append(median_value)
                self.min_values.append(min_value)
                self.max_models.append(model_max)
                self.median_models.append(model_median)
                self.min_models.append(model_min)
                self.confidence_intervals.append(confidence_interval)
                self.months.append(month)
                self.zones.append(zone_name)
