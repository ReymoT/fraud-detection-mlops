import time
import schedule
import subprocess

'''
Script to schedule monitoring every hour
'''

def run_monitoring():
    print("Running drift monitoring...")
    subprocess.run(["python", "monitoring/drift_report.py"])

# FOR PROD RUN:
#schedule.every(1).hours.do(run_monitoring) 
schedule.every(1).minutes.do(run_monitoring) # For demo purposes

while True:
    schedule.run_pending()
    time.sleep(60)