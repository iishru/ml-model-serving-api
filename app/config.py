"""Application settings, read once from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    """All runtime configuration for the API. Nothing secret is hard-coded here."""

    model_path: Path
    baseline_path: Path
    model_version: str
    database_url: str | None
    db_connect_timeout_s: float
    drift_window: int
    drift_threshold: float
    drift_ks_p_value: float
    drift_min_samples: int


def get_settings() -> Settings:
    """Build Settings from environment variables, using safe defaults for non-secrets."""
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "models" / "model.joblib")),
        baseline_path=Path(os.getenv("BASELINE_PATH", PROJECT_ROOT / "models" / "baseline_stats.json")),
        model_version=os.getenv("MODEL_VERSION", "unversioned"),
        # No default: the database location and password must come from the environment.
        database_url=os.getenv("DATABASE_URL") or None,
        db_connect_timeout_s=float(os.getenv("DB_CONNECT_TIMEOUT_S", "30")),
        drift_window=int(os.getenv("DRIFT_WINDOW", "100")),
        drift_threshold=float(os.getenv("DRIFT_THRESHOLD", "2.0")),
        drift_ks_p_value=float(os.getenv("DRIFT_KS_P_VALUE", "0.01")),
        drift_min_samples=int(os.getenv("DRIFT_MIN_SAMPLES", "30")),
    )
