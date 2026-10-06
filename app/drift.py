"""Input drift monitoring: are live requests still like the data the model was trained on?

Two checks run for every feature:
1. Mean shift: |live_mean - baseline_mean| / baseline_std. Simple and easy to explain,
   but only sees a change in the average.
2. Two-sample Kolmogorov-Smirnov test between the live values and a stored sample of
   training values. It compares whole distributions and gives a p-value.
A feature counts as drifted if either check fires.
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

from app.schemas import FEATURE_NAMES, DriftReport, FeatureDrift


def load_baseline(path: Path) -> dict:
    """Read the training statistics written by training/train.py."""
    if not path.exists():
        raise FileNotFoundError(f"Drift baseline not found at {path}. Run training/train.py first.")
    return json.loads(path.read_text())


def clean_live_inputs(live: pd.DataFrame, baseline: dict) -> pd.DataFrame:
    """Apply the training-time cleaning (0 = missing -> training median) so we compare like with like."""
    cleaned = live.astype(float).copy()
    for column, median in baseline["zero_is_missing"].items():
        cleaned[column] = cleaned[column].replace(0, median)
    return cleaned


def overall_status(n_drifted: int) -> str:
    """Map the number of drifted features to a verdict."""
    if n_drifted == 0:
        return "ok"
    if n_drifted <= 2:
        return "warning"
    return "retrain_recommended"


def compute_drift(
    live: pd.DataFrame,
    baseline: dict,
    threshold: float = 2.0,
    ks_p_value: float = 0.01,
    min_samples: int = 30,
) -> DriftReport:
    """Compare live inputs (one column per feature) against the training baseline."""
    samples = len(live)
    if samples < min_samples:
        return DriftReport(status="insufficient_data", samples=samples)

    cleaned = clean_live_inputs(live[FEATURE_NAMES], baseline)
    results = []
    for name in FEATURE_NAMES:
        stats = baseline["features"][name]
        live_values = cleaned[name]
        live_mean = float(live_values.mean())
        # Guard against a zero standard deviation (a constant feature).
        drift_score = abs(live_mean - stats["mean"]) / max(stats["std"], 1e-9)
        p_value = float(ks_2samp(live_values, stats["sample"]).pvalue)
        results.append(FeatureDrift(
            name=name,
            live_mean=round(live_mean, 4),
            baseline_mean=round(stats["mean"], 4),
            drift_score=round(drift_score, 4),
            ks_p_value=round(p_value, 6),
            mean_shift_drifted=drift_score > threshold,
            ks_drifted=p_value < ks_p_value,
            drifted=drift_score > threshold or p_value < ks_p_value,
        ))

    drifted = [r.name for r in results if r.drifted]
    return DriftReport(
        status=overall_status(len(drifted)), samples=samples, drifted_features=drifted, features=results
    )
