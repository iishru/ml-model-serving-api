@echo off
REM Windows version of run_experiments.sh. Run from the project root with the venv active
REM and MLFLOW_TRACKING_URI set, e.g.:  set MLFLOW_TRACKING_URI=http://127.0.0.1:5000
REM train.py promotes the run with the highest test F1 to models\model.joblib.

python training\train.py --n-estimators 100 --max-depth 4  --min-samples-leaf 1 || exit /b 1
python training\train.py --n-estimators 200 --max-depth 6  --min-samples-leaf 2 || exit /b 1
python training\train.py --n-estimators 300 --max-depth 8  --min-samples-leaf 4 || exit /b 1
python training\train.py --n-estimators 300 --min-samples-leaf 1 || exit /b 1
python training\train.py --n-estimators 500 --max-depth 10 --min-samples-leaf 3 || exit /b 1

echo Done. Look for the last "WINNER so far" line, and compare all runs in the MLflow UI.
