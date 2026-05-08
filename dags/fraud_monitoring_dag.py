from datetime import datetime, timedelta
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_drift_report():
    subprocess.run(["python", "/opt/airflow/monitoring/drift_report.py"], check = True)


def retrain_if_drift():
    subprocess.run(["python", "/opt/airflow/monitoring/retrain.py"], check = True)


default_args = {
    "owner": "reimo",
    "retries": 1,
    "retry_delay": timedelta(minutes = 2),
}

with DAG(
    dag_id = "fraud_monitoring_and_retraining",
    default_args = default_args,
    description = "Run drift monitoring and trigger retraining if drift is detected.",
    start_date = datetime(2025, 1, 1),
    schedule = "@hourly",
    catchup = False,
    is_paused_upon_creation = False,
) as dag:

    drift_report = PythonOperator(
        task_id = "run_drift_report",
        python_callable = run_drift_report,
    )

    retrain = PythonOperator(
        task_id = "retrain_if_drift",
        python_callable = retrain_if_drift,
    )

    drift_report >> retrain