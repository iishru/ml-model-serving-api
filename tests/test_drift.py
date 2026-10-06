"""Drift monitor tests. These call compute_drift directly, so no API or database is needed."""

import pandas as pd

from app.drift import compute_drift
from app.schemas import FEATURE_NAMES


def baseline_like_data(baseline: dict, n: int) -> pd.DataFrame:
    """Take n rows of the stored training sample: by construction it matches the baseline."""
    return pd.DataFrame({name: baseline["features"][name]["sample"][:n] for name in FEATURE_NAMES})


def test_baseline_matching_data_is_ok(baseline: dict) -> None:
    """Data drawn from the training distribution raises no alarm."""
    report = compute_drift(baseline_like_data(baseline, 100), baseline)

    assert report.status == "ok"
    assert report.samples == 100
    assert report.drifted_features == []
    assert len(report.features) == len(FEATURE_NAMES)


def test_strongly_shifted_data_recommends_retraining(baseline: dict) -> None:
    """Shifting three features far from training values triggers retrain_recommended."""
    live = baseline_like_data(baseline, 100)
    live["Glucose"] += 60
    live["BMI"] += 12
    live["Age"] += 20

    report = compute_drift(live, baseline)

    assert report.status == "retrain_recommended"
    assert set(report.drifted_features) == {"Glucose", "BMI", "Age"}


def test_one_shifted_feature_is_a_warning(baseline: dict) -> None:
    """A single drifted feature gives a warning, not a retrain recommendation."""
    live = baseline_like_data(baseline, 100)
    live["Glucose"] += 80

    assert compute_drift(live, baseline).status == "warning"


def test_too_few_samples_is_insufficient_data(baseline: dict) -> None:
    """Under 30 samples there is no verdict, because the statistics would be too noisy."""
    report = compute_drift(baseline_like_data(baseline, 29), baseline)

    assert report.status == "insufficient_data"
    assert report.samples == 29
    assert report.features == []
