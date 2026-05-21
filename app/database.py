import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, pool_pre_ping = True)

def init_db():
    with engine.begin() as conn:
        conn.execute(text("SELECT pg_advisory_lock(123456789);")) # prevent race condition
        try:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    logged_at TIMESTAMPTZ DEFAULT NOW(),
                    amt DOUBLE PRECISION,
                    lat DOUBLE PRECISION,
                    long DOUBLE PRECISION,
                    merch_lat DOUBLE PRECISION,
                    merch_long DOUBLE PRECISION,
                    city_pop INTEGER,
                    category TEXT,
                    gender TEXT,
                    state TEXT,
                    merchant TEXT,
                    trans_date_trans_time TEXT,
                    dob TEXT,
                    trans_hour INTEGER,
                    trans_dayofweek INTEGER,
                    trans_month INTEGER,
                    age DOUBLE PRECISION,
                    is_night INTEGER,
                    distance DOUBLE PRECISION,
                    fraud_probability DOUBLE PRECISION,
                    threshold DOUBLE PRECISION,
                    risk_level TEXT,
                    flag INTEGER
                );
            """))
        finally:
            conn.execute(text("SELECT pg_advisory_unlock(123456789);"))