import pandas as pd
from tqdm import tqdm


def get_weather_info(station_id):
    """
    P1-1. Функция для получения метеорологических данных с портала "Погода и климат"
    :param station_id: (str) Пятизначный синоптический индекс метеостанции
    :return: (Dataframe) Набор данных по температуре и осадкам
    """
    fail_station_id = []

    try:
        # Получить средние месячные и годовые температуры воздуха
        air_temperature = pd.concat(
            pd.read_html(f"http://www.pogodaiklimat.ru/history/{station_id}.htm"),
            axis=1
        )
        # Добавить года и месяца в заголовки
        air_temperature = pd.DataFrame(air_temperature.values[1:],
                                    columns=air_temperature.iloc[0])
        # Добавить тип данных и номер метеостанции
        air_temperature['номер_станции'] = station_id
        air_temperature['вид_данных'] = 'Av_Temp'

        # Получить суммы выпавших осадков
        sum_of_weather = pd.concat(
            pd.read_html(f"http://www.pogodaiklimat.ru/history/{station_id}_2.htm"),
            axis=1
        )
        # Добавить года и месяца в заголовки
        sum_of_weather = pd.DataFrame(sum_of_weather.values[1:],
                                    columns=sum_of_weather.iloc[0])
        # Добавить тип данных и номер метеостанции
        sum_of_weather['номер_станции'] = station_id
        sum_of_weather['вид_данных'] = 'Sum_Precip'

        # Вернуть результирующий DataFrame для конкретной станции station_id
        return pd.concat([air_temperature, sum_of_weather], axis=0)
    except Exception:
        fail_station_id.append(station_id)
        return None


def get_weather_data(station_ids, output_filepath='output_weather.csv'):
    """
    P1-1. Функция получения данных с портала "Погода и климат" и последующего сохранения в csv файл
    :param station_ids: (list(str)) Список Пятизначных синоптических индексов метеостанций
    :param output_filepath: (str) путь сохранения выходного файла
    """
    weather_info = []

    for station_id in tqdm(station_ids):
        weather_info.append(get_weather_info(station_id))

    # Обработка полученных данных
    output_weather = pd.concat(weather_info, axis=0)
    output_weather = output_weather[['год', 'янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек', 'за год', 'номер_станции', 'вид_данных']]
    output_weather.rename(columns={'год': 'Year', 'янв': 'Jan',  'фев': 'Feb', 'мар': 'March', 'апр': 'April', 'май': 'May', 'июн': 'June', 'июл': 'July', 'авг': 'August', 'сен': 'September', 'окт': 'October', 'ноя': 'November', 'дек': 'December', 'за год': 'Annual', 'номер_станции': 'Station_Number', 'вид_данных': 'Type_data'}, inplace=True)

    output_path = output_filepath

    # Сохранение результата
    output_weather.to_csv(output_path, index=False)
