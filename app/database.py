"""PostgreSQL access: engine, the prediction_logs table, and the queries the API needs."""

import logging
import math
import time
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import DateTime, Engine, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from app.schemas import FEATURE_NAMES, PatientFeatures

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base class for ORM models."""


class PredictionLog(Base):
    """One row per prediction served: the inputs, the output and how long it took."""

    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    pregnancies: Mapped[float] = mapped_column(Float)
    glucose: Mapped[float] = mapped_column(Float)
    blood_pressure: Mapped[float] = mapped_column(Float)
    skin_thickness: Mapped[float] = mapped_column(Float)
    insulin: Mapped[float] = mapped_column(Float)
    bmi: Mapped[float] = mapped_column(Float)
    diabetes_pedigree_function: Mapped[float] = mapped_column(Float)
    age: Mapped[float] = mapped_column(Float)
    prediction: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float] = mapped_column(Float)
    latency_ms: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64))


# API/training feature name -> database column name.
FEATURE_COLUMNS = dict(zip(FEATURE_NAMES, [
    "pregnancies", "glucose", "blood_pressure", "skin_thickness",
    "insulin", "bmi", "diabetes_pedigree_function", "age",
]))


def create_db_engine(database_url: str) -> Engine:
    """Create the SQLAlchemy engine. SQLite (used by the tests) needs one extra option."""
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)


def init_database(engine: Engine, timeout_s: float) -> bool:
    """Wait for the database (it may start after the API in Docker), then create tables.

    Returns False if the database is still unreachable after timeout_s seconds; the API
    then keeps serving predictions without logging them.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            Base.metadata.create_all(engine)
            logger.info("Database ready; prediction_logs table exists")
            return True
        except Exception as exc:  # noqa: BLE001 - any driver error means "not ready yet"
            if time.monotonic() >= deadline:
                logger.error("Database not reachable after %.0fs: %s", timeout_s, exc)
                return False
            logger.warning("Database not ready, retrying in 2s (%s)", type(exc).__name__)
            time.sleep(2)


def is_connected(engine: Engine | None) -> bool:
    """Run a trivial query to check the database answers right now."""
    if engine is None:
        return False
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False


def log_prediction(
    session_factory: sessionmaker[Session],
    features: PatientFeatures,
    prediction: int,
    probability: float,
    latency_ms: float,
    model_version: str,
) -> None:
    """Insert one prediction into prediction_logs."""
    values = {FEATURE_COLUMNS[name]: value for name, value in features.model_dump().items()}
    with session_factory() as session:
        session.add(PredictionLog(
            **values,
            prediction=prediction,
            probability=probability,
            latency_ms=latency_ms,
            model_version=model_version,
        ))
        session.commit()


def recent_inputs(engine: Engine, limit: int) -> pd.DataFrame:
    """Return the input features of the most recent `limit` predictions, named as in training."""
    columns = [getattr(PredictionLog, column) for column in FEATURE_COLUMNS.values()]
    query = select(*columns).order_by(PredictionLog.id.desc()).limit(limit)
    with engine.connect() as connection:
        frame = pd.DataFrame(connection.execute(query).all(), columns=list(FEATURE_COLUMNS.values()))
    return frame.rename(columns={v: k for k, v in FEATURE_COLUMNS.items()})


def serving_stats(engine: Engine) -> dict[str, float | int | None]:
    """Total predictions, average and p95 latency, and the share of high-risk predictions."""
    with engine.connect() as connection:
        total, avg_latency, positive_rate = connection.execute(
            select(
                func.count(PredictionLog.id),
                func.avg(PredictionLog.latency_ms),
                func.avg(PredictionLog.prediction * 1.0),
            )
        ).one()
        p95 = None
        if total:
            # Nearest-rank p95: sort latencies and take the value at position ceil(0.95 * n).
            offset = math.ceil(0.95 * total) - 1
            p95 = connection.execute(
                select(PredictionLog.latency_ms).order_by(PredictionLog.latency_ms).offset(offset).limit(1)
            ).scalar_one()
    return {
        "total_predictions": int(total),
        "avg_latency_ms": round(float(avg_latency), 3) if avg_latency is not None else None,
        "p95_latency_ms": round(float(p95), 3) if p95 is not None else None,
        "positive_rate": round(float(positive_rate), 4) if positive_rate is not None else None,
    }
