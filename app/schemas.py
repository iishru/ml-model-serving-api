"""Pydantic request and response models.

Field names match the training data columns exactly, so a request can be turned into
the DataFrame the model expects without any renaming.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]


class PatientFeatures(BaseModel):
    """One patient's measurements. Out-of-range or missing values are rejected with HTTP 422."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "Pregnancies": 2, "Glucose": 138, "BloodPressure": 62, "SkinThickness": 35,
                "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.127, "Age": 47,
            }
        },
    )

    Pregnancies: int = Field(ge=0, le=20, description="Number of pregnancies")
    Glucose: float = Field(ge=0, le=300, description="Plasma glucose (mg/dL); 0 = not measured")
    BloodPressure: float = Field(ge=0, le=200, description="Diastolic blood pressure (mm Hg); 0 = not measured")
    SkinThickness: float = Field(ge=0, le=100, description="Triceps skin fold (mm); 0 = not measured")
    Insulin: float = Field(ge=0, le=900, description="2-hour serum insulin (mu U/ml); 0 = not measured")
    BMI: float = Field(ge=0, le=80, description="Body mass index; 0 = not measured")
    DiabetesPedigreeFunction: float = Field(ge=0, le=3, description="Family-history score")
    Age: int = Field(ge=1, le=120, description="Age in years")


class PredictionResponse(BaseModel):
    """What /predict returns."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: Literal[0, 1]
    label: Literal["low risk", "high risk"]
    probability: float = Field(description="Model probability of the high-risk class")
    model_version: str
    latency_ms: float = Field(description="Time spent in the model call, in milliseconds")


class FeatureDrift(BaseModel):
    """Drift result for one input feature."""

    name: str
    live_mean: float
    baseline_mean: float
    drift_score: float = Field(description="|live_mean - baseline_mean| / baseline_std")
    ks_p_value: float = Field(description="Two-sample Kolmogorov-Smirnov test p-value")
    mean_shift_drifted: bool = Field(description="drift_score above the threshold")
    ks_drifted: bool = Field(description="KS p-value below the significance level")
    drifted: bool = Field(description="Either check fired")


class DriftReport(BaseModel):
    """Overall drift verdict. 'features' is empty when there is not enough data."""

    status: Literal["ok", "warning", "retrain_recommended", "insufficient_data"]
    samples: int
    drifted_features: list[str] = []
    features: list[FeatureDrift] = []


class MetricsSummary(BaseModel):
    """Serving statistics computed from the prediction log."""

    total_predictions: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    positive_rate: float | None
    drift: DriftReport


class HealthResponse(BaseModel):
    """What /health returns."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok"]
    model_loaded: bool
    database_connected: bool
