import json
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from threading import Lock

from fastapi import Request
from fastapi.responses import JSONResponse

from lab10.api import app as original_app

from lab12.config import MONITORING_LOG_DIR


app = original_app

LOG_FILE = MONITORING_LOG_DIR / "api_requests.jsonl"

request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_latency_ms": 0.0,
    "status_codes": Counter(),
}

stats_lock = Lock()


@app.middleware("http")
async def monitor_api_requests(request: Request, call_next):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        raise

    finally:
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        success = 200 <= status_code < 400

        record = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "method": request.method,
            "endpoint": request.url.path,
            "status_code": status_code,
            "success": success,
            "latency_ms": round(latency_ms, 3),
        }

        with stats_lock:
            request_stats["total_requests"] += 1

            if success:
                request_stats["successful_requests"] += 1
            else:
                request_stats["failed_requests"] += 1

            request_stats["total_latency_ms"] += latency_ms
            request_stats["status_codes"][str(status_code)] += 1

        MONITORING_LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")


@app.get("/monitoring/metrics")
def monitoring_metrics():
    with stats_lock:
        total = request_stats["total_requests"]
        successful = request_stats["successful_requests"]
        failed = request_stats["failed_requests"]

        average_latency = (
            request_stats["total_latency_ms"] / total
            if total else 0.0
        )

        return {
            "total_requests": total,
            "successful_requests": successful,
            "failed_requests": failed,
            "failure_rate_percent": (
                failed / total * 100 if total else 0.0
            ),
            "average_latency_ms": round(
                average_latency, 3
            ),
            "status_codes": dict(
                request_stats["status_codes"]
            ),
            "log_file": str(LOG_FILE),
        }