import pandas as pd
from src.ingest import load_data


def test_load_data_parses_dates(tmp_path):
    csv_path = tmp_path / "sample.csv"

    df = pd.DataFrame({
        "trans_date_trans_time": ["2020-01-01 03:00:00"],
        "dob": ["1990-01-01"],
        "amt": [100.0],
    })

    df.to_csv(csv_path, index=False)

    result = load_data(str(csv_path))

    assert pd.api.types.is_datetime64_any_dtype(result["trans_date_trans_time"])
    assert pd.api.types.is_datetime64_any_dtype(result["dob"])