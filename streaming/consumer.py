import json
import requests
from kafka import KafkaConsumer

TOPIC = "fraud-transactions"
API_URL = "http://localhost:8000/predict"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers = "localhost:9092",
    auto_offset_reset = "earliest",
    value_deserializer = lambda m: json.loads(m.decode("utf-8"))
)

print("Listening for transactions...")

for message in consumer:

    payload = message.value

    response = requests.post(API_URL, json = payload)

    print(response.json())