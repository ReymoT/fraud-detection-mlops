import numpy as np

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * R * np.arcsin(np.sqrt(a))

def build_features(df):
    df = df.copy()

    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
    df["trans_month"] = df["trans_date_trans_time"].dt.month

    df["age"] = (
        (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    )

    df["distance"] = haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    df["is_night"] = (df["trans_hour"] < 6).astype("int8")
    df["amt_log"] = np.log1p(df["amt"])

    return df