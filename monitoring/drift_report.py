import os
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

REPORT_DIR = "monitoring/reports"
REFERENCE_PATH = "data/reference_transactions.csv"
CURRENT_PATH = "scored_transactions_drifted.csv" # simulated data drift

MONITORING_COLUMNS = [
    "amt",
    "distance",
    "city_pop",
    "trans_hour",
    "trans_dayofweek",
    "trans_month",
    "is_night",
    "age",
    "fraud_probability",
    "flag",
]

def main():
    os.makedirs(REPORT_DIR, exist_ok = True)

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    reference = reference[MONITORING_COLUMNS].dropna()
    current = current[MONITORING_COLUMNS].dropna()

    report = Report([
        DataDriftPreset(drift_share = 0.3)
    ])

    result = report.run(
        reference_data = reference,
        current_data = current
    )

    result.save_html(f"{REPORT_DIR}/data_drift_report.html")

    print(f"Saved drift report to {REPORT_DIR}/data_drift_report.html")

if __name__ == "__main__":
    main()