import subprocess
from pathlib import Path

'''
Script to retrain model if necessary
'''

STATUS_PATH = Path("monitoring/reports/drift_status.txt")

def main():
    if not STATUS_PATH.exists():
        print("No drift status found. Run drift_report.py first.")
        return

    status = STATUS_PATH.read_text()

    # retrain if drift has been detected in the data
    if "DRIFT_DETECTED" in status:
        print("Drift detected. Triggering retraining...")
        subprocess.run(["python", "-m", "src.train"], check = True)
    else:
        print("No drift detected. No retraining needed.")

if __name__ == "__main__":
    main()