import pandas as pd
from src.features import build_features

def test_build_features_creates_expected_columns():
    df = pd.DataFrame({
        "trans_date_trans_time": pd.to_datetime(["2020-01-01 03:00:00"]),
        "dob": pd.to_datetime(["1990-01-01"]),
        "lat": [40.0],
        "long": [-74.0],
        "merch_lat": [41.0],
        "merch_long": [-75.0],
        "amt": [100.0],
    })

    result = build_features(df)

    assert "amt_log" in result.columns
    assert "distance" in result.columns
    assert "is_night" in result.columns