"""Checks on the saved model file itself."""

import pandas as pd

from app.config import get_settings
from app.model_loader import load_model, predict
from app.schemas import FEATURE_NAMES, PatientFeatures
from tests.conftest import VALID_PATIENT


def test_model_loads_and_predicts_valid_shapes() -> None:
    """The saved Pipeline loads and returns one class and two probabilities per row."""
    pipeline = load_model(get_settings().model_path)
    rows = pd.DataFrame([VALID_PATIENT, {**VALID_PATIENT, "Glucose": 0}], columns=FEATURE_NAMES)

    probabilities = pipeline.predict_proba(rows)
    classes = pipeline.predict(rows)

    assert probabilities.shape == (2, 2)
    assert classes.shape == (2,)
    assert set(classes) <= {0, 1}


def test_predict_helper_matches_pipeline() -> None:
    """The API's predict() gives the same class as the Pipeline's own predict()."""
    pipeline = load_model(get_settings().model_path)
    prediction, probability = predict(pipeline, PatientFeatures(**VALID_PATIENT))

    expected = int(pipeline.predict(pd.DataFrame([VALID_PATIENT], columns=FEATURE_NAMES))[0])
    assert prediction == expected
    assert 0.0 <= probability <= 1.0
