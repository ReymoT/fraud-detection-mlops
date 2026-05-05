DATA_PATH = "data/fraudTrain.csv"
MODEL_DIR = "models"

TEST_SIZE = 0.2
TOP_PERCENTILE = 99.5
RANDOM_STATE = 42

FEATURE_COLS = [
    "amt_log", "distance", "city_pop",
    "trans_hour", "trans_dayofweek", "trans_month",
    "is_night", "age",
    "category", "gender", "state",
    "merchant", "unix_time"
]

EXPERIMENT_NAME = "fraud-detection-xgboost"