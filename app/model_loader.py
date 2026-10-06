"""Loading the trained Pipeline and running a single prediction."""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.schemas import FEATURE_NAMES, PatientFeatures

logger = logging.getLogger(__name__)


def load_model(path: Path) -> Pipeline:
    """Load the saved imputer + Random Forest Pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Run training/train.py first.")
    pipeline = joblib.load(path)
    # Training used all CPU cores; for one row at a time, starting worker threads costs more
    # than it saves, so serve with a single thread (this cut latency from ~150 ms to ~30 ms).
    pipeline.set_params(model__n_jobs=1)
    logger.info("Loaded model from %s", path)
    return pipeline


def predict(pipeline: Pipeline, features: PatientFeatures) -> tuple[int, float]:
    """Return (predicted class, probability of the high-risk class) for one patient."""
    row = pd.DataFrame([features.model_dump()], columns=FEATURE_NAMES)
    probability = float(pipeline.predict_proba(row)[0, 1])
    # Same rule as RandomForestClassifier.predict (argmax), without running the forest twice.
    prediction = int(probability > 0.5)
    return prediction, probability
