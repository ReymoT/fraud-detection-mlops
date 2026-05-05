import pandas as pd
import numpy as np

# Simulate data drift for Evidently
df = pd.read_csv("scored_transactions.csv")

drifted = df.copy()

# simulate realistic drift
drifted["amt"] = drifted["amt"] * 3
drifted["distance"] = drifted["distance"] * 2
drifted["fraud_probability"] = np.clip(
    drifted["fraud_probability"] + 0.2,
    0,
    1
)

drifted.to_csv("scored_transactions_drifted.csv", index=False)

print("Saved scored_transactions_drifted.csv")