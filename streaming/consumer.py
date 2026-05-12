import json
import requests
from kafka import KafkaConsumer, KafkaProducer

RAW_TOPIC = "fraud-transactions"
SCORED_TOPIC = "scored-transactions"

API_URL = "http://localhost:8000/predict"

consumer = KafkaConsumer(
    RAW_TOPIC,
    bootstrap_servers = "localhost:9092",
    auto_offset_reset = "earliest",
    enable_auto_commit = True,
    group_id = "fraud-scoring-consumer-v1",
    value_deserializer = lambda m: json.loads(m.decode("utf-8")),
)

producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8"),
)

print("Listening for raw transactions...")

for message in consumer:
    transaction = message.value

    try:
        response = requests.post(API_URL, json = transaction, timeout = 10)
        response.raise_for_status()

        prediction = response.json()

        scored_event = {
            "transaction": transaction,
            "prediction": prediction,
        }

        producer.send(SCORED_TOPIC, value = scored_event)
        producer.flush()

        print("Scored transaction sent: ", prediction)

    except Exception as e:
        print("Error scoring transaction: ", e)