import json

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, chi2_contingency

from config import (
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    LAB12_REPORT_DIR,
)


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index."""

    reference = np.asarray(reference)
    current = np.asarray(current)

    breakpoints = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(breakpoints) < 2:
        return 0.0

    reference_counts, _ = np.histogram(
        reference, bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current, bins=breakpoints
    )

    reference_pct = (
        reference_counts + 0.001
    ) / (len(reference) + 0.001 * len(reference_counts))

    current_pct = (
        current_counts + 0.001
    ) / (len(current) + 0.001 * len(current_counts))

    return float(
        np.sum(
            (current_pct - reference_pct)
            * np.log(current_pct / reference_pct)
        )
    )


def detect_numeric_drift(reference, current):
    """Detect numerical feature drift using KS test and PSI."""

    results = []

    for column in NUMERICAL_FEATURES:
        ref = reference[column].dropna()
        cur = current[column].dropna()

        statistic, p_value = ks_2samp(ref, cur)

        psi = calculate_psi(ref, cur)

        results.append({
            "feature": column,
            "feature_type": "numerical",
            "ks_statistic": float(statistic),
            "p_value": float(p_value),
            "psi": psi,
            "drift_detected": bool(p_value < 0.05 or psi >= 0.2),
        })

    return results


def detect_categorical_drift(reference, current):
    """Detect categorical feature drift using chi-square test."""

    results = []

    for column in CATEGORICAL_FEATURES:
        ref_counts = reference[column].fillna(
            "Missing"
        ).value_counts()

        cur_counts = current[column].fillna(
            "Missing"
        ).value_counts()

        categories = sorted(
            set(ref_counts.index) | set(cur_counts.index)
        )

        observed = np.array([
            [ref_counts.get(category, 0) for category in categories],
            [cur_counts.get(category, 0) for category in categories],
        ])

        statistic, p_value, _, _ = chi2_contingency(
            observed
        )

        results.append({
            "feature": column,
            "feature_type": "categorical",
            "chi_square_statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05),
        })

    return results


def detect_prediction_drift(reference_predictions, current_predictions):
    """Compare reference and current prediction distributions."""

    statistic, p_value = ks_2samp(
        reference_predictions,
        current_predictions,
    )

    reference_rate = float(
        np.mean(np.asarray(reference_predictions) >= 0.5)
    )

    current_rate = float(
        np.mean(np.asarray(current_predictions) >= 0.5)
    )

    return {
        "ks_statistic": float(statistic),
        "p_value": float(p_value),
        "reference_positive_rate": reference_rate,
        "current_positive_rate": current_rate,
        "drift_detected": bool(p_value < 0.05),
    }


def detect_concept_drift(y_true, y_pred):
    """Measure performance against actual target labels."""

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
    }


def save_drift_report(results, filename="drift_report.json"):
    LAB12_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = LAB12_REPORT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(f"Drift report saved: {output_path}")