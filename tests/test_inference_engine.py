import asyncio
import numpy as np
import pandas as pd
from app.inference_engine import DynamicBatcher


class DummyModel:
    def predict_proba(self, X):
        probs = np.full(len(X), 0.7)
        return np.column_stack([1 - probs, probs])


def test_dynamic_batcher_returns_score():
    async def run_test():
        model = DummyModel()
        batcher = DynamicBatcher(
            model = model,
            max_batch_size = 4,
            batch_timeout_ms = 1
        )

        await batcher.start()

        df = pd.DataFrame({"x": [1]})

        score = await asyncio.wait_for(
            batcher.predict(df),
            timeout = 2
        )

        assert score == 0.7

        metrics = batcher.metrics()

        assert metrics["total_requests"] == 1
        assert metrics["total_batches"] >= 1
        assert metrics["avg_batch_size"] >= 1

    asyncio.run(run_test())


def test_dynamic_batcher_batches_multiple_requests():
    async def run_test():
        model = DummyModel()
        batcher = DynamicBatcher(
            model = model,
            max_batch_size = 8,
            batch_timeout_ms = 5
        )

        await batcher.start()

        df = pd.DataFrame({"x": [1]})

        scores = await asyncio.wait_for(
            asyncio.gather(
                *[batcher.predict(df) for _ in range(4)]
            ),
            timeout = 2
        )

        assert scores == [0.7, 0.7, 0.7, 0.7]

        metrics = batcher.metrics()

        assert metrics["total_requests"] == 4
        assert metrics["total_batches"] >= 1
        assert metrics["avg_batch_size"] >= 1

    asyncio.run(run_test())