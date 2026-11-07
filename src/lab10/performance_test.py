
import time
import json
import statistics
from pathlib import Path

import requests
import pandas as pd


# ==================================================
# 1. CONFIGURATION
# ==================================================

BASE_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{BASE_URL}/predict"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "lab10"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

TOTAL_REQUESTS = 100


# ==================================================
# 2. SAMPLE INPUT
# ==================================================

SAMPLE_ORDER = {
    "Sales": 250.0,
    "Quantity": 3,
    "Discount": 0.1,
    "Shipping Cost": 15.0,
    "Order Year": 2017,
    "Order Month": 11,
    "Ship Mode": "Second Class",
    "Segment": "Consumer",
    "City": "Los Angeles",
    "State": "California",
    "Country": "United States",
    "Market": "US",
    "Region": "West",
    "Category": "Technology",
    "Sub-Category": "Phones",
    "Order Priority": "High"
}


# ==================================================
# 3. PERFORMANCE TEST
# ==================================================

def run_performance_test():

    response_times = []
    successful_requests = 0
    failed_requests = 0
    predictions = []

    print("\nStarting API performance testing...")
    print(f"Total requests: {TOTAL_REQUESTS}\n")

    # Warm-up request
    try:
        requests.post(
            PREDICT_URL,
            json=SAMPLE_ORDER,
            timeout=10
        )
    except requests.RequestException as error:
        print(f"API connection failed: {error}")
        return

    # Repeated inference requests
    for i in range(TOTAL_REQUESTS):

        start_time = time.perf_counter()

        try:
            response = requests.post(
                PREDICT_URL,
                json=SAMPLE_ORDER,
                timeout=10
            )

            elapsed = time.perf_counter() - start_time
            response_times.append(elapsed * 1000)

            if response.status_code == 200:
                successful_requests += 1
                result = response.json()
                predictions.append(result["profitability"])
            else:
                failed_requests += 1

        except requests.RequestException:
            failed_requests += 1

        if (i + 1) % 20 == 0:
            print(f"Completed {i + 1}/{TOTAL_REQUESTS} requests")

    # ==================================================
    # 4. PERFORMANCE METRICS
    # ==================================================

    if not response_times:
        print("No valid response times recorded.")
        return

    total_time = sum(response_times) / 1000

    avg_latency = statistics.mean(response_times)
    min_latency = min(response_times)
    max_latency = max(response_times)
    median_latency = statistics.median(response_times)

    throughput = successful_requests / total_time

    reliability = (
        successful_requests / TOTAL_REQUESTS
    ) * 100

    consistency = (
        len(set(predictions)) <= 1
        if predictions else False
    )

    # ==================================================
    # 5. DISPLAY RESULTS
    # ==================================================

    results = {
        "total_requests": TOTAL_REQUESTS,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "average_latency_ms": round(avg_latency, 3),
        "median_latency_ms": round(median_latency, 3),
        "minimum_latency_ms": round(min_latency, 3),
        "maximum_latency_ms": round(max_latency, 3),
        "throughput_requests_per_second": round(throughput, 3),
        "reliability_percent": round(reliability, 2),
        "response_consistency": consistency
    }

    print("\n========== API PERFORMANCE RESULTS ==========")

    for key, value in results.items():
        print(f"{key}: {value}")

    # ==================================================
    # 6. SAVE RESULTS
    # ==================================================

    with open(
        REPORT_DIR / "performance_results.json",
        "w"
    ) as file:
        json.dump(results, file, indent=4)

    pd.DataFrame([results]).to_csv(
        REPORT_DIR / "performance_results.csv",
        index=False
    )

    print("\nPerformance results saved successfully.")
    print(f"Report directory: {REPORT_DIR}")


if __name__ == "__main__":
    run_performance_test()
