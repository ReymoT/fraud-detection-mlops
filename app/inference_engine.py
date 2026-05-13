import asyncio
import time
from dataclasses import dataclass
import pandas as pd


@dataclass
class InferenceRequest:
    features: pd.DataFrame
    future: asyncio.Future
    created_at: float


class DynamicBatcher:
    def __init__(
        self,
        model,
        max_batch_size: int = 64,
        batch_timeout_ms: int = 10,
        max_queue_size: int = 1000
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize = max_queue_size)
        self.running = False

        self.total_requests = 0
        self.total_batches = 0
        self.total_batch_size = 0

        self.rejected_requests = 0
        self.timed_out_requests = 0

    async def start(self): # start the engine
        if not self.running:
            self.running = True
            asyncio.create_task(self._batch_loop()) # runs independently

    async def predict(self, features: pd.DataFrame):
        if self.queue.full():
            self.rejected_requests += 1
            raise RuntimeError("Inference queue is full")
        
        loop = asyncio.get_running_loop() # get the running event loop
        future = loop.create_future()

        request = InferenceRequest(
            features = features,
            future = future,
            created_at = time.time()
        )

        await self.queue.put(request) # add request to the queue, batching loop will pick it up
        self.total_requests += 1

        return await future
    
    def record_timeout(self):
        self.timed_out_requests += 1

    async def _batch_loop(self):
        while self.running:
            first_request = await self.queue.get()
            batch = [first_request]

            start = time.time() # initialize timestamp

            while len(batch) < self.max_batch_size: # run until max batch size is met or until we time out
                elapsed_ms = (time.time() - start) * 1000
                remaining_ms = self.batch_timeout_ms - elapsed_ms

                if remaining_ms <= 0:
                    break

                try:
                    request = await asyncio.wait_for(
                        self.queue.get(), # wait for another request until we time out
                        timeout = remaining_ms / 1000,
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break

            try:
                batch_df = pd.concat(
                    [req.features for req in batch],
                    ignore_index = True,
                ) # create a collective dataframe out of the batched request dataframes

                scores = self.model.predict_proba(batch_df)[:, 1]

                self.total_batches += 1
                self.total_batch_size += len(batch)

                for req, score in zip(batch, scores):
                    if not req.future.cancelled(): # thread safety
                        req.future.set_result(float(score)) # set future as the predicted score

            except Exception as e:
                if not req.future.cancelled():
                    for req in batch:
                        req.future.set_exception(e)

    def metrics(self):
        avg_batch_size = (
            self.total_batch_size / self.total_batches
            if self.total_batches > 0
            else 0
        )

        return {
            "queue_depth": self.queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "avg_batch_size": avg_batch_size,
            "rejected_requests": self.rejected_requests,
            "timed_out_requests": self.timed_out_requests,
            "max_batch_size": self.max_batch_size,
            "batch_timeout_ms": self.batch_timeout_ms
        }