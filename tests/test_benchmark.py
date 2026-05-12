from benchmarks.benchmark_api import print_results

def test_print_results_runs_without_error(capsys):
    results = {
        "endpoint": "/predict",
        "requests": 1000,
        "concurrency": 50,
        "throughput_rps": 123.45,
        "p50_ms": 10.1,
        "p95_ms": 20.2,
        "p99_ms": 30.3,
        "avg_ms": 15.5,
        "total_time_sec": 8.1,
    }

    print_results(results)

    captured = capsys.readouterr()

    assert "Benchmark Results" in captured.out
    assert "throughput_rps" in captured.out
    assert "/predict" in captured.out