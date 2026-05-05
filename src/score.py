import joblib
import pandas as pd
import numpy as np

from src.ingest import load_data
from src.features import build_features
from src.config import DATA_PATH, FEATURE_COLS

MODEL_DIR = "models"

def main():
    # load data
    df = load_data(DATA_PATH)
    df = build_features(df)

    # load artifacts
    model = joblib.load(f"{MODEL_DIR}/fraud_model.pkl")
    merchant_freq = joblib.load(f"{MODEL_DIR}/merchant_freq.pkl")
    model_columns = joblib.load(f"{MODEL_DIR}/model_columns.pkl")
    threshold = joblib.load(f"{MODEL_DIR}/threshold.pkl")

    # keep raw fields for dashboard
    raw_df = df.copy()

    # prepare features (same as training)
    new_df = df[FEATURE_COLS].dropna()

    new_df = pd.get_dummies(
        new_df,
        columns=["category", "gender", "state"]
    )

    # merchant frequency
    new_df["merchant_freq"] = new_df["merchant"].map(merchant_freq).fillna(0)

    # drop unused columns
    new_df = new_df.drop(["merchant", "unix_time"], axis=1)

    # align columns
    new_df = new_df.reindex(columns=model_columns, fill_value=0)

    # predict
    scores = model.predict_proba(new_df)[:, 1]

    # add results to raw dataframe
    raw_df["fraud_probability"] = scores
    raw_df["flag"] = (scores >= threshold).astype(int)

    # keep useful columns for dashboard
    output = raw_df[
        [
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
    ]

    output.to_csv("scored_transactions.csv", index = False)

    print("Saved scored_transactions.csv")

    reference = raw_df[
        [
            "amt",
            "distance",
            "city_pop",
            "trans_hour",
            "trans_dayofweek",
            "trans_month",
            "is_night",
            "age",
        ]
    ].copy()

    reference["fraud_probability"] = scores
    reference["flag"] = (scores >= threshold).astype(int)

    reference.sample(50000, random_state = 42).to_csv(
        "data/reference_transactions.csv",
        index = False
    )

if __name__ == "__main__":
    main()