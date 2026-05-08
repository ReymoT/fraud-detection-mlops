import time
import schedule
import subprocess

'''
Script to schedule monitoring every hour

REPLACED BY AIRFLOW
'''

def run_monitoring():
    print("Running drift monitoring...")
    subprocess.run(["python", "monitoring/drift_report.py"])

schedule.every(1).hours.do(run_monitoring) 

while True:
    schedule.run_pending()
    time.sleep(60)