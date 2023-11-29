from fastapi import FastAPI,UploadFile, File
from pydantic import BaseModel
import re
import pickle
import numpy as np
import pandas as pd
import io
from typing import List

with open('ridge_weights_gridsearch.pkl', 'rb') as f:
    ridge_model = pickle.load(f)

with open('standard_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

class CSVUpload(BaseModel):
    file: UploadFile = File(..., description="CSV file to upload")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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

class Items(BaseModel):
    objects: List[Item]


def extract_number(text):
    match = re.search(r'[\d.]+', text)
    return float(match.group()) if match else None

# Функция для преобразования объекта Item в формат, подходящий для модели предсказания
def transform_item_to_predict_format(item: Item):
    
    mileage_value = extract_number(item.mileage)
    engine_value = extract_number(item.engine)
    max_power_value = extract_number(item.max_power)
    transformed_item = {
        'year': item.year,
        'km_driven': item.km_driven,
        'mileage': mileage_value,
        'engine': engine_value,
        'max_power': max_power_value,
        'fuel_Diesel': 1 if item.fuel == 'Diesel' else 0,
        'fuel_LPG': 1 if item.fuel == 'LPG' else 0,
        'fuel_Petrol': 1 if item.fuel == 'Petrol' else 0,
        'seller_type_Individual': 1 if item.seller_type == 'Individual' else 0,
        'seller_type_Trustmark_Dealer': 1 if item.seller_type == 'Trustmark Dealer' else 0,
        'transmission_Manual': 1 if item.transmission == 'Manual' else 0,
        'owner_Fourth_Above_Owner': 1 if item.owner == 'Fourth & Above Owner' else 0,
        'owner_Second_Owner': 1 if item.owner == 'Second Owner' else 0,
        'owner_Test_Drive_Car': 1 if item.owner == 'Test Drive Car' else 0,
        'owner_Third_Owner': 1 if item.owner == 'Third Owner' else 0,
        'seats_2': 1 if item.seats == 2 else 0,
        'seats_4': 1 if item.seats == 4 else 0,
        'seats_5': 1 if item.seats == 5 else 0,
        'seats_6': 1 if item.seats == 6 else 0,
        'seats_7': 1 if item.seats == 7 else 0,
        'seats_8': 1 if item.seats == 8 else 0,
        'seats_9': 1 if item.seats == 9 else 0,
        'seats_14': 1 if item.seats == 14 else 0,
    }
 
    
    return [list(transformed_item.values())]

def csv_to_list_of_items(csv_file):
    contents = csv_file.file.read().decode("utf-8")
    
    df = pd.read_csv(io.StringIO(contents))
    items = []
    for index, row in df.iterrows():
        item = Item(
            name=row["name"],
            year=row["year"],
            selling_price=row["selling_price"],
            km_driven=row["km_driven"],
            fuel=row["fuel"],
            seller_type=row["seller_type"],
            transmission=row["transmission"],
            owner=row["owner"],
            mileage=row["mileage"],
            engine=row["engine"],
            max_power=row["max_power"],
            torque=row["torque"],
            seats=row["seats"]
        )
        items.append(item)
    return items    


app = FastAPI()




@app.post("/predict_price")
def predict_price(item: Item):
    transformed_item = transform_item_to_predict_format(item)

    scaled_item = scaler.transform(transformed_item)
    
    prediction = ridge_model.predict(scaled_item)
    return prediction.tolist()[0]

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    items = csv_to_list_of_items(file)
    predictions = []
    for item in items:
        transformed_item = transform_item_to_predict_format(item)
        scaled_item = scaler.transform(transformed_item)
        
        prediction = ridge_model.predict(scaled_item)
        predictions.append(prediction[0])

    return predictions