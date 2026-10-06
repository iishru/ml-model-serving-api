"""FastAPI application: serves the diabetes risk model and logs every prediction."""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database import (
    create_db_engine, init_database, is_connected, log_prediction, recent_inputs, serving_stats,
)
from app.drift import compute_drift, load_baseline
from app.model_loader import load_model, predict
from app.schemas import DriftReport, HealthResponse, MetricsSummary, PatientFeatures, PredictionResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and connect to the database once, when the server starts."""
    settings = get_settings()
    app.state.settings = settings
    app.state.model = load_model(settings.model_path)
    app.state.baseline = load_baseline(settings.baseline_path)
    # Last drift report and the prediction count it was computed at (see current_drift_report).
    app.state.drift_cache = None

    app.state.engine = None
    app.state.session_factory = None
    if settings.database_url:
        engine = create_db_engine(settings.database_url)
        init_database(engine, settings.db_connect_timeout_s)
        app.state.engine = engine
        app.state.session_factory = sessionmaker(bind=engine)
    else:
        logger.warning("DATABASE_URL is not set; predictions will not be logged")

    yield

    if app.state.engine is not None:
        app.state.engine.dispose()


app = FastAPI(title="Diabetes Risk Model API", version="1.0.0", lifespan=lifespan)


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(features: PatientFeatures, request: Request) -> PredictionResponse:
    """Predict diabetes risk for one patient, then log the request to the database."""
    state = request.app.state
    start = time.perf_counter()
    prediction, probability = predict(state.model, features)
    latency_ms = (time.perf_counter() - start) * 1000

    response = PredictionResponse(
        prediction=prediction,
        label="high risk" if prediction == 1 else "low risk",
        probability=round(probability, 4),
        model_version=state.settings.model_version,
        latency_ms=round(latency_ms, 3),
    )

    # Logging is best-effort: a database problem must never cost the caller their prediction.
    if state.session_factory is not None:
        try:
            log_prediction(
                state.session_factory, features, prediction, probability, latency_ms,
                state.settings.model_version,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not log prediction: %s", exc)
    return response


def require_engine(request: Request) -> Engine:
    """Return the database engine, or answer 503 if the database is not reachable."""
    engine = request.app.state.engine
    if not is_connected(engine):
        raise HTTPException(status_code=503, detail="Database is not available")
    return engine


def run_drift_check(request: Request, total_predictions: int) -> DriftReport:
    """Compute a fresh drift report from the most recent logged requests and cache it."""
    state = request.app.state
    settings = state.settings
    live = recent_inputs(state.engine, settings.drift_window)
    report = compute_drift(
        live, state.baseline, settings.drift_threshold, settings.drift_ks_p_value, settings.drift_min_samples
    )
    state.drift_cache = (total_predictions, report)
    return report


@app.get("/metrics-summary", response_model=MetricsSummary)
def metrics_summary(request: Request) -> MetricsSummary:
    """Serving statistics plus the drift report.

    The drift report is reused while no new predictions have arrived, so polling this
    endpoint is cheap; it is recomputed as soon as the prediction count changes.
    """
    engine = require_engine(request)
    stats = serving_stats(engine)
    cache = request.app.state.drift_cache
    if cache is not None and cache[0] == stats["total_predictions"]:
        report = cache[1]
    else:
        report = run_drift_check(request, stats["total_predictions"])
    return MetricsSummary(**stats, drift=report)


@app.post("/drift/check", response_model=DriftReport)
def drift_check(request: Request) -> DriftReport:
    """Force a fresh drift calculation and return the full report."""
    engine = require_engine(request)
    return run_drift_check(request, serving_stats(engine)["total_predictions"])


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Is the model in memory, and does the database answer right now?"""
    return HealthResponse(
        status="ok",
        model_loaded=getattr(request.app.state, "model", None) is not None,
        database_connected=is_connected(request.app.state.engine),
    )
