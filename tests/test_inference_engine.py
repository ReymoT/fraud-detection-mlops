import asyncio
import numpy as np
import pandas as pd

from app.inference_engine import DynamicBatcher


class DummyModel:
    def predict_proba(self, X):
        probs = np.full(len(X), 0.7)
        return np.column_stack([1 - probs, probs])

def test_dynamic_batcher_returns_scores():
    async def run_test():
        model = DummyModel()
        batcher = DynamicBatcher(model, max_batch_size = 4, batch_timeout_ms = 1)
        await batcher.start()

        df = pd.DataFrame({"x": [1]})

        score = await batcher.predict(df)

        assert score == 0.7
        assert batcher.metrics()["total_requests"] == 1

    asyncio.run(run_test())