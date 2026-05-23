import json
import os
from datetime import datetime, timezone

import pandas as pd
from kafka import KafkaConsumer

SCORED_TOPIC = "scored-transactions"
ALERT_LOG_PATH = "logs/high_risk_alerts.csv"

consumer = KafkaConsumer(
    SCORED_TOPIC,
    bootstrap_servers = "localhost:9092",
    auto_offset_reset = "earliest",
    enable_auto_commit = True,
    group_id = "fraud-alert-consumer-v1",
    value_deserializer = lambda m: json.loads(m.decode("utf-8"))
)

print("Listening for scored transactions...")

def save_alert(event):
    os.makedirs("logs", exist_ok = True)

    transaction = event["transaction"]
    prediction = event["prediction"]

    row = {
        "alert_time": datetime.now(timezone.utc).isoformat(),
        **transaction,
        "fraud_probability": prediction["fraud_probability"],
        "threshold": prediction["threshold"],
        "risk_level": prediction["risk_level"],
        "flag": prediction["flag"],
        "top_reasons": json.dumps(prediction.get("top_reasons", []))
    }

    df = pd.DataFrame([row])

    file_exists = os.path.exists(ALERT_LOG_PATH)
    df.to_csv(ALERT_LOG_PATH, mode = "a", header = (not file_exists), index = False)


for message in consumer:
    event = message.value
    prediction = event["prediction"]

    if prediction["flag"] == 1 or prediction["risk_level"] == "HIGH":
        print("HIGH RISK ALERT:", prediction)
        save_alert(event)
    else:
        print("Not high risk:", prediction["fraud_probability"])