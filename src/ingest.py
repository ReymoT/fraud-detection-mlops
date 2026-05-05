import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    return df