
import json
from datetime import datetime, timezone

from src.lab13.config import (
    LAB13_MODEL_DIR,
    LAB13_REPORT_DIR,
    BASELINE_MODEL_VERSION,
    RETRAINED_MODEL_VERSION,
    INITIAL_PRODUCTION_MODEL,
    RETRAINED_MODEL_STATUS,
)


LIFECYCLE_FILE = LAB13_REPORT_DIR / "model_lifecycle.json"

LAB13_MODEL_VERSION = "lab13-v3"


def create_initial_registry():
    """Create the initial model lifecycle registry."""

    LAB13_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LAB13_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    registry = {
        "project": "Global Superstore MLOps",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "current_production_model": INITIAL_PRODUCTION_MODEL,
        "models": {
            BASELINE_MODEL_VERSION: {
                "version": BASELINE_MODEL_VERSION,
                "status": "production",
                "artifact_location": "models/lab7/v1",
                "description": "Existing baseline profitability model",
            },
            RETRAINED_MODEL_VERSION: {
                "version": RETRAINED_MODEL_VERSION,
                "status": RETRAINED_MODEL_STATUS,
                "artifact_location": "models/lab12/v2",
                "description": (
                    "Retrained candidate model; not promoted "
                    "because baseline evaluation metrics were higher"
                ),
            },
            LAB13_MODEL_VERSION: {
                "version": LAB13_MODEL_VERSION,
                "status": "candidate_not_promoted",
                "artifact_location": "models/lab13/v3",
                "description": (
                    "Experiment 13 retrained candidate model; "
                    "not promoted because baseline evaluation "
                    "metrics were higher"
                ),
            },
        },
    }

    with open(LIFECYCLE_FILE, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    return registry


def update_registry():
    """Safely add Lab13 model without overwriting existing entries."""

    registry = load_registry()

    registry["models"][LAB13_MODEL_VERSION] = {
        "version": LAB13_MODEL_VERSION,
        "status": "candidate_not_promoted",
        "artifact_location": "models/lab13/v3",
        "description": (
            "Experiment 13 retrained candidate model; "
            "not promoted because baseline evaluation "
            "metrics were higher"
        ),
    }

    # Preserve the existing production model.
    registry["current_production_model"] = INITIAL_PRODUCTION_MODEL

    with open(LIFECYCLE_FILE, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=4)

    return registry


def load_registry():
    """Load the existing model lifecycle registry."""

    if not LIFECYCLE_FILE.exists():
        raise FileNotFoundError(
            f"Lifecycle registry not found: {LIFECYCLE_FILE}"
        )

    with open(LIFECYCLE_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def get_model_status(model_version):
    """Get the lifecycle status of a model version."""

    registry = load_registry()

    if model_version not in registry["models"]:
        raise ValueError(f"Unknown model version: {model_version}")

    return registry["models"][model_version]["status"]


if __name__ == "__main__":
    registry = update_registry()

    print("Model lifecycle registry updated successfully.")
    print("Production model:", registry["current_production_model"])

    for version, details in registry["models"].items():
        print(version, "->", details["status"])
