"""Train the diabetes risk model, track the run in MLflow, and promote the best model.

Usage:
    python training/train.py --n-estimators 200 --max-depth 6 --min-samples-leaf 2

Every run is logged to the MLflow experiment "diabetes-risk-model". After logging,
the script asks MLflow which run has the highest test F1. If this run is the best
(or no model has been saved yet), it is written to models/model.joblib together with
the drift baseline models/baseline_stats.json.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "diabetes.csv"
MODEL_DIR = PROJECT_ROOT / "models"
EXPERIMENT_NAME = "diabetes-risk-model"
RANDOM_STATE = 42
TARGET = "Outcome"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
# In these columns a value of 0 is physically impossible, so 0 means "missing".
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
# Number of training values kept per feature for the drift KS test.
BASELINE_SAMPLE_SIZE = 500


def parse_args() -> argparse.Namespace:
    """Read the hyperparameters from the command line."""
    parser = argparse.ArgumentParser(description="Train the diabetes risk model.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=None, help="Omit for unlimited depth.")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    return parser.parse_args()


def build_pipeline(n_estimators: int, max_depth: int | None, min_samples_leaf: int) -> Pipeline:
    """Imputer + Random Forest in one Pipeline, so serving applies the same preprocessing.

    The imputer treats 0 as missing in ZERO_IS_MISSING and fills it with the column median
    learned from the training data only (fit happens on the training split).
    """
    preprocess = ColumnTransformer(
        transformers=[("zero_imputer", SimpleImputer(missing_values=0, strategy="median"), ZERO_IS_MISSING)],
        remainder="passthrough",
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        # About 35% of patients are positive; "balanced" stops the model ignoring that class.
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute the test-set metrics."""
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    return {
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_precision": precision_score(y_test, predictions),
        "test_recall": recall_score(y_test, predictions),
        "test_f1": f1_score(y_test, predictions),
        "test_roc_auc": roc_auc_score(y_test, probabilities),
    }


def build_baseline(pipeline: Pipeline, x_train: pd.DataFrame) -> dict:
    """Summarise the cleaned training data per feature for the drift monitor."""
    cleaned = x_train.copy()
    imputer = pipeline.named_steps["preprocess"].named_transformers_["zero_imputer"]
    impute_values = dict(zip(ZERO_IS_MISSING, imputer.statistics_))
    for column, median in impute_values.items():
        cleaned[column] = cleaned[column].replace(0, median)

    sample = cleaned.sample(n=min(BASELINE_SAMPLE_SIZE, len(cleaned)), random_state=RANDOM_STATE)
    features = {}
    for column in FEATURES:
        features[column] = {
            "mean": float(cleaned[column].mean()),
            "std": float(cleaned[column].std()),
            "min": float(cleaned[column].min()),
            "max": float(cleaned[column].max()),
            "sample": [round(float(v), 4) for v in sample[column]],
        }
    return {"zero_is_missing": impute_values, "n_rows": len(cleaned), "features": features}


def is_best_run(run_id: str) -> bool:
    """Return True if this run has the highest test F1 in the experiment (ROC-AUC breaks ties)."""
    best = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["metrics.test_f1 DESC", "metrics.test_roc_auc DESC"],
        max_results=1,
    )
    return not best.empty and best.iloc[0]["run_id"] == run_id


def main() -> None:
    """Train, evaluate, log to MLflow and promote the model if it is the best so far."""
    # MLflow prints emoji; on Windows, piped output defaults to cp1252 and would crash on them.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    data = pd.read_csv(DATA_PATH)
    x, y = data[FEATURES], data[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
    }
    pipeline = build_pipeline(**params)

    # Cross-validation on the training split only; the test set stays untouched.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, x_train, y_train, cv=cv, scoring="f1")

    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)
    metrics["cv_f1_mean"] = float(cv_scores.mean())
    metrics["cv_f1_std"] = float(cv_scores.std())
    baseline = build_baseline(pipeline, x_train)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run:
        mlflow.log_params({**params, "class_weight": "balanced", "random_state": RANDOM_STATE, "test_size": 0.2})
        mlflow.log_metrics(metrics)
        mlflow.log_dict(baseline, "baseline_stats.json")
        mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=x_train.head(3).astype(float))
        run_id, run_name = run.info.run_id, run.info.run_name

    print(f"Run '{run_name}' ({run_id}) params={params}")
    for name, value in metrics.items():
        print(f"  {name:<15} {value:.4f}")

    model_path = MODEL_DIR / "model.joblib"
    if is_best_run(run_id) or not model_path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(pipeline, model_path)
        (MODEL_DIR / "baseline_stats.json").write_text(json.dumps(baseline, indent=2))
        print(f"WINNER so far: run '{run_name}' (test F1={metrics['test_f1']:.4f}) -> saved to {model_path}")
    else:
        print("Not the best run in the experiment; models/model.joblib left unchanged.")


if __name__ == "__main__":
    main()
