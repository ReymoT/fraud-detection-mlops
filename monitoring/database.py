import os
import pandas as pd
from sqlalchemy import create_engine

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://fraud:fraud@app-postgres:5432/fraud_predictions",
)

engine = create_engine(DATABASE_URL, pool_pre_ping = True)


def load_recent_predictions(limit = 5000):
    query = f"""
        SELECT *
        FROM predictions
        ORDER BY logged_at DESC
        LIMIT {limit}
    """

    return pd.read_sql(query, engine)