import pandas as pd
import numpy as np

import pickle
import json

import re

def read_pkl(source: str):
    "Считывание объекта из pickle формата"

    with open(source, "rb") as f:
        return pickle.load(f)
    

def read_json(source: str):
    "Считывание объекта из json формата"

    with open(source, "r") as f:
        return json.load(f)
    

def is_nm_kgm(x):
    """
    Функция, которая определяет единицы измерения torque.
    Делает значение 9.81, если kgm, и 1, если nm
    """
    if isinstance(x, str):
        if 'kgm' in x.lower():
            return 9.81
        else:
            return 1
    else:
        return x
    

def extract_nums(x):
    if isinstance(x, str):
        return [float(num.replace(',', '').replace('.', '.')) for num in re.findall(r'\d+(?:\.\d+)?(?:,\d{3})*', x)]
    else:
        return x
    

def transform_csv(df: pd.DataFrame) -> pd.DataFrame:
    "Преобразование входного датасета для подачи в модель"

    # Создам колонку sellimg_price, тк обучал OHE вместе с ней, в конце ее дропну
    df['selling_price'] = 0

    # Генерим фичи
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = df[col].apply(lambda x: re.sub(r'[^0-9.]', '', x) if isinstance(x, str) else x)
        df[col] = df[col].replace('', None)
        df[col] = df[col].astype('float')

    df['is_nm_kgm'] = df['torque'].apply(is_nm_kgm)
    df['torque_nums'] = df['torque'].apply(extract_nums)
    df['torque_first'] = df['torque_nums'].apply(lambda x: x[0] if isinstance(x, list) else x)
    df['max_torque'] = df['torque_nums'].apply(lambda x: max(x[1:]) if isinstance(x, list) and len(x[1:]) > 0 else None)
    df['torque'] = df['torque_first'] * df['is_nm_kgm']
    df = df.drop(['is_nm_kgm', 'torque_nums', 'torque_first'], axis=1)

    # Удаляем пропуски
    medians = read_json('../models/medians.json')
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col].fillna(medians[col], inplace=True)

    # Преобразование столбцов
    df['engine'] = df['engine'].astype('int')
    df['seats'] = df['seats'].astype('object')

    # Преобразуем колонку name
    df['name'] = df['name'].apply(lambda x: x.split()[0])

    ohe = read_pkl('../models/ohe_trained.pkl')
    cat_cols = df.select_dtypes(include=['object']).columns

    df_ohe = ohe.transform(df[cat_cols])
    df_ohe = pd.DataFrame(df_ohe, columns=ohe.get_feature_names_out())
    df_ohe = pd.concat([df[numeric_cols], df_ohe], axis=1)
    df_ohe = df_ohe.drop(['seats', 'selling_price'], axis=1)

    return df_ohe
