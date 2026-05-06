import time
import pandas as pd
import requests

'''
Script to simulate a data stream
'''

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/fraudTrain.csv"

df = pd.read_csv(DATA_PATH).sample(1000, random_state = 42) # random state for reproducibility

for _, row in df.iterrows():
    payload = {
        "amt": float(row["amt"]),
        "lat": float(row["lat"]),
        "long": float(row["long"]),
        "merch_lat": float(row["merch_lat"]),
        "merch_long": float(row["merch_long"]),
        "city_pop": int(row["city_pop"]),
        "category": row["category"],
        "gender": row["gender"],
        "state": row["state"],
        "merchant": row["merchant"],
        "trans_date_trans_time": row["trans_date_trans_time"],
        "dob": row["dob"],
    }

    response = requests.post(API_URL, json = payload)
    print(response.json())

    time.sleep(1)