import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "transactions-dlq",
    bootstrap_servers = "localhost:9092",
    value_deserializer = lambda m: json.loads(m.decode("utf-8")),
    auto_offset_reset = "earliest",
    enable_auto_commit = True,
    group_id = "fraud-dlq-monitor"
)

print("Listening for DLQ events...")

for message in consumer:
    event = message.value
    print("\nDLQ EVENT")
    print(json.dumps(event, indent = 2))