"""API tests using FastAPI's TestClient (no server, no Docker)."""

from fastapi.testclient import TestClient

from tests.conftest import VALID_PATIENT


def test_predict_valid_input_returns_prediction(client: TestClient) -> None:
    """A valid request returns 200 and every documented field."""
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "label", "probability", "model_version", "latency_ms"}
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("high risk" if body["prediction"] == 1 else "low risk")
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test-version"
    assert body["latency_ms"] > 0


def test_predict_is_logged(client: TestClient) -> None:
    """Each prediction is written to the database and counted in the summary."""
    client.post("/predict", json=VALID_PATIENT)
    client.post("/predict", json=VALID_PATIENT)

    summary = client.get("/metrics-summary").json()
    assert summary["total_predictions"] == 2
    assert summary["drift"]["status"] == "insufficient_data"


def test_predict_missing_field_returns_422(client: TestClient) -> None:
    """Leaving out a feature is rejected and the error names the missing field."""
    patient = {k: v for k, v in VALID_PATIENT.items() if k != "Glucose"}
    response = client.post("/predict", json=patient)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "Glucose"]


def test_predict_out_of_range_returns_422(client: TestClient) -> None:
    """An impossible value (Age 150) is rejected with a clear message."""
    response = client.post("/predict", json={**VALID_PATIENT, "Age": 150})

    assert response.status_code == 422
    assert "less than or equal to 120" in response.json()["detail"][0]["msg"]


def test_logging_failure_does_not_break_prediction(client: TestClient) -> None:
    """If the database write fails, the caller still gets their prediction."""

    def broken_session_factory() -> None:
        raise RuntimeError("database is down")

    client.app.state.session_factory = broken_session_factory
    response = client.post("/predict", json=VALID_PATIENT)

    assert response.status_code == 200


def test_health_ok(client: TestClient) -> None:
    """Health reports the model and the database as available."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "database_connected": True}
