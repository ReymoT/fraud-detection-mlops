import json
import requests
from kafka import KafkaConsumer, KafkaProducer
from datetime import datetime, timezone
import os

RAW_TOPIC = "fraud-transactions"
SCORED_TOPIC = "scored-transactions"
DLQ_TOPIC = "transactions-dlq"

API_URL = "http://localhost:8000/predict"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
API_KEY = os.getenv("API_KEY", "dev-secret-key")

consumer = KafkaConsumer(
    RAW_TOPIC,
    bootstrap_servers = "localhost:9092",
    auto_offset_reset = "earliest",
    enable_auto_commit = True,
    group_id = "fraud-scoring-consumer-v1",
    value_deserializer = lambda m: json.loads(m.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers = "localhost:9092",
    value_serializer = lambda v: json.dumps(v).encode("utf-8")
)

def publish_to_dlq(original_message, error):
    dlq_event = {
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "source_topic": RAW_TOPIC,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "payload": original_message
    }

    producer.send(DLQ_TOPIC, dlq_event)
    producer.flush()

    print(f"Sent failed message to DLQ: {error}")

print("Listening for raw transactions...")

for message in consumer:
    transaction = message.value

    try:
        response = requests.post(
            API_URL,
            json = transaction,
            timeout = 10,
            headers = {"x-api-key": API_KEY}
        )
        response.raise_for_status()

        prediction = response.json()

        scored_event = {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "transaction": transaction,
            "prediction": prediction
        }

        producer.send(SCORED_TOPIC, value = scored_event)
        producer.flush()

        print("Scored transaction sent: ", prediction)

    except Exception as e:
        print("Error scoring transaction: ", e)
        publish_to_dlq(transaction, e)