"""Shared pytest fixtures. Tests run without Docker: the database is a temporary SQLite file."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.drift import load_baseline
from app.main import app

VALID_PATIENT = {
    "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62, "SkinThickness": 35,
    "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.127, "Age": 47,
}


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """An API client whose prediction log is a fresh SQLite database for each test."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MODEL_VERSION", "test-version")
    # Entering the context runs the lifespan: model loaded, tables created.
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def baseline() -> dict:
    """The drift baseline produced by training."""
    return load_baseline(get_settings().baseline_path)
