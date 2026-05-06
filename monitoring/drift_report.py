import os
import pandas as pd

from evidently import Report, ColumnMapping
from evidently.presets import DataDriftPreset, DataSummaryPreset
from scipy.stats import wasserstein_distance

REPORT_DIR = "monitoring/reports"
REFERENCE_PATH = "data/reference_transactions.csv"
CURRENT_PATH = "scored_transactions_drifted.csv" # simulated data drift, 

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

DRIFT_THRESHOLD = 0.3

DRIFT_COLUMNS = ["amt", "distance", "fraud_probability"]

def compute_simple_drift(reference, current):
    drifted = []

    for col in DRIFT_COLUMNS:
        ref = reference[col].dropna()
        cur = current[col].dropna()

        if ref.std() == 0:
            continue

        distance = wasserstein_distance(ref, cur) / ref.std()

        if distance > DRIFT_THRESHOLD:
            drifted.append(col)

    return drifted

def main():
    os.makedirs(REPORT_DIR, exist_ok = True)

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(CURRENT_PATH)

    reference = reference[MONITORING_COLUMNS].dropna()
    current = current[MONITORING_COLUMNS].dropna()

    drifted_cols = compute_simple_drift(reference, current)

    with open(f"{REPORT_DIR}/drift_status.txt", "w") as f:
        if drifted_cols:
            f.write("DRIFT_DETECTED\n")
            f.write(",".join(drifted_cols))
        else:
            f.write("NO_DRIFT\n")

    print("Drifted columns:", drifted_cols)

    column_mapping = ColumnMapping(
        prediction = "fraud_probability",
        target = None
    )

    report = Report([
        DataDriftPreset(drift_share = 0.3),
        DataSummaryPreset()
    ])

    result = report.run(
        reference_data = reference,
        current_data = current,
        column_mapping = column_mapping
    )

    result.save_html(f"{REPORT_DIR}/data_drift_report.html")

    print(f"Saved drift report to {REPORT_DIR}/data_drift_report.html")

if __name__ == "__main__":
    main()