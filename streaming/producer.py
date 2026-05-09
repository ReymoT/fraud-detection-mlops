import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "fraud-transactions"

producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8")
)

df = pd.read_csv("data/fraudTrain.csv").sample(100, random_state = 42)

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

    producer.send(TOPIC, value = payload)

    print("Sent: ", payload)

    time.sleep(1)

producer.flush()