#!/bin/sh
# Train several hyperparameter combinations so they can be compared in the MLflow UI.
# train.py promotes the run with the highest test F1 to models/model.joblib.
# Usage (from the project root, with MLFLOW_TRACKING_URI set):  sh training/run_experiments.sh
set -e
PYTHON="${PYTHON:-python}"

"$PYTHON" training/train.py --n-estimators 100 --max-depth 4  --min-samples-leaf 1
"$PYTHON" training/train.py --n-estimators 200 --max-depth 6  --min-samples-leaf 2
"$PYTHON" training/train.py --n-estimators 300 --max-depth 8  --min-samples-leaf 4
"$PYTHON" training/train.py --n-estimators 300 --min-samples-leaf 1
"$PYTHON" training/train.py --n-estimators 500 --max-depth 10 --min-samples-leaf 3

echo "Done. The last 'WINNER so far' line names the best run; compare all runs in the MLflow UI."
