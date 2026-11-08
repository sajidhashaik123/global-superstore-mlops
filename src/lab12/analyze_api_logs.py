import json
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

from lab12.config import LAB12_REPORT_DIR, MONITORING_LOG_DIR


LOG_FILE = MONITORING_LOG_DIR / "api_requests.jsonl"
REPORT_FILE = LAB12_REPORT_DIR / "api_performance_report.json"


def analyze_api_logs():
    if not LOG_FILE.exists():
        raise FileNotFoundError(f"API log file not found: {LOG_FILE}")

    records = []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("No API request records found.")

    total_requests = len(records)

    successful_requests = sum(
        1 for record in records if record["success"]
    )

    failed_requests = total_requests - successful_requests

    latencies = sorted(
        float(record["latency_ms"]) for record in records
    )

    status_codes = Counter(
        str(record["status_code"]) for record in records
    )

    endpoint_counts = Counter(
        record["endpoint"] for record in records
    )

    timestamps = [
        datetime.fromisoformat(record["timestamp"])
        for record in records
    ]

    time_span_seconds = (
        max(timestamps) - min(timestamps)
    ).total_seconds()

    def percentile(values, percent):
        return round(
            float(__import__("numpy").percentile(values, percent)),
            3
        )

    report = {
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "failure_rate_percent": round(
            failed_requests / total_requests * 100, 3
        ),
        "average_latency_ms": round(
            sum(latencies) / total_requests, 3
        ),
        "minimum_latency_ms": round(min(latencies), 3),
        "maximum_latency_ms": round(max(latencies), 3),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "time_span_seconds": round(time_span_seconds, 3),
        "throughput_requests_per_second": round(
            total_requests / time_span_seconds, 3
        ) if time_span_seconds > 0 else None,
        "status_code_distribution": dict(status_codes),
        "endpoint_distribution": dict(endpoint_counts),
    }

    LAB12_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nAPI PERFORMANCE REPORT")
    print("-" * 40)

    for key, value in report.items():
        print(f"{key}: {value}")

    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    analyze_api_logs()