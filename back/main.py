from fastapi import FastAPI, UploadFile, File
import io
import uvicorn
from pydantic import BaseModel
from typing import List, Union
import csv

import pandas as pd

from utils import *

import copy

import warnings
warnings.filterwarnings("ignore")

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    ex = pd.DataFrame([dict(item)])
    ex = transform_csv(ex)

    model = read_pkl('../models/ridge_regr_trained.pkl')

    return round(model.predict(ex)[0], 2)


@app.post("/predict_items")
def predict_items(items: UploadFile = File(...)) -> Union[List[float], int]:

    # Создаю копию, тк мне нужно проходиться по строкам
    # объекта для валидации, затем подать на вход
    # функции, которая рассчитывает предсказания
    items_copy = copy.deepcopy(items)

    df = pd.read_csv(items.file)
    items.file.close()

    # Произвожу валидацию строк датасета
    contents = items_copy.file.read()
    
    decoded_contents = contents.decode('utf-8')

    buffer = io.StringIO(decoded_contents)
    reader = csv.DictReader(buffer)
    
    for row in reader:
        try:
            Item(**row)
        except Exception:
            return 0

    items_copy.file.close()
    
    # Нахожу вектор предсказаний
    df_trans = transform_csv(df.copy())

    model = read_pkl('../models/ridge_regr_trained.pkl')
    pred = model.predict(df_trans)
    
    return pred


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=80)