import argparse
import asyncio
import time
import statistics
import httpx
import json
import os
from datetime import datetime


PAYLOAD = {
    "amt": 5000,
    "lat": 40.7128,
    "long": -74.006,
    "merch_lat": 34.0522,
    "merch_long": -118.2437,
    "city_pop": 1000,
    "category": "shopping_net",
    "gender": "M",
    "state": "CA",
    "merchant": "unknown_rare_store",
    "trans_date_trans_time": "2020-12-25 02:30:00",
    "dob": "2003-01-01",
}


async def send_request(client, url, latencies, errors):
    start = time.perf_counter()

    try:
        response = await client.post(url, json = PAYLOAD)
        response.raise_for_status() # raise exception in case of error

        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    except Exception as e:
        errors.append(type(e).__name__)


async def run_benchmark(endpoint, requests, concurrency, warmup):
    url = f"http://127.0.0.1:8000{endpoint}"

    latencies = []
    errors = []

    limits = httpx.Limits(
        max_connections = concurrency,
        max_keepalive_connections = concurrency
    )

    timeout = httpx.Timeout(
        connect = 10.0,
        read = 60.0,
        write = 10.0,
        pool = 60.0
    )

    async with httpx.AsyncClient(timeout = timeout, limits = limits) as client:
        # Warmup phase
        warmup_latencies = []

        for i in range(0, warmup, concurrency):
            batch_size = min(concurrency, warmup - i)

            tasks = [
                send_request(client, url, warmup_latencies, errors) for _ in range(batch_size)
            ]

            await asyncio.gather(*tasks)

        # Measured phase
        latencies = []

        start = time.perf_counter()

        for i in range(0, requests, concurrency):
            batch_size = min(concurrency, requests - i)

            tasks = [
                send_request(client, url, latencies, errors) for _ in range(batch_size)
            ]

            await asyncio.gather(*tasks)

        end = time.perf_counter()

    total_time = end - start

    latencies_sorted = sorted(latencies)

    def percentile(p):
        index = int((p / 100) * len(latencies_sorted)) - 1
        index = max(0, min(index, len(latencies_sorted) - 1))
        return latencies_sorted[index]

    results = {
        "endpoint": endpoint,
        "requests": requests,
        "concurrency": concurrency,
        "throughput_rps": requests / total_time,
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
        "avg_ms": statistics.mean(latencies),
        "total_time_sec": total_time,
        "warmup_requests": warmup,
        "error_types": dict(
            (e, errors.count(e)) for e in set(errors)
        )
    }

    return results


def print_results(results):
    print("\nBenchmark Results")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

def save_results(results):
    os.makedirs("benchmarks/results", exist_ok = True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"benchmarks/results/"
        f"{results['endpoint'].replace('/', '')}_"
        f"{results['requests']}_"
        f"{results['concurrency']}_"
        f"{timestamp}.json"
    )

    with open(filename, "w") as f:
        json.dump(results, f, indent = 2)

    print(f"\nSaved benchmark to: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--endpoint", required = True)
    parser.add_argument("--requests", type = int, default = 1000)
    parser.add_argument("--concurrency", type = int, default = 100)
    parser.add_argument("--warmup", type = int, default = 100)

    args = parser.parse_args()

    results = asyncio.run(
        run_benchmark(
            endpoint = args.endpoint,
            requests = args.requests,
            concurrency = args.concurrency,
            warmup = args.warmup
        )
    )

    print_results(results)
    save_results(results)