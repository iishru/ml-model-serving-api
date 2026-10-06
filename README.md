# Diabetes Risk Model Serving API with Monitoring

A Scikit-learn Random Forest predicts diabetes risk from eight routine measurements (Pima Indians
Diabetes dataset, 768 patients). Training runs are tracked in **MLflow** and the best run is promoted
to `models/model.joblib`. A **FastAPI** service loads that model once at startup and serves it on
`POST /predict`, with **Pydantic** rejecting invalid input. Every prediction is logged to
**PostgreSQL** with its latency. A **drift monitor** compares the most recent inputs with the
training data and flags when the incoming population has shifted, which is the signal to consider
retraining. The whole system (database, MLflow, API) starts with `docker compose up --build`.

## Architecture

```
                      training/train.py ──── params, metrics, model ────► MLflow tracking server (:5000)
                             │
                             └── writes models/model.joblib + models/baseline_stats.json
                                                │                  │
                                                ▼                  ▼
 client ── POST /predict ──► FastAPI (:8000) ──► model     drift monitor (app/drift.py)
 (curl, simulate_traffic.py)     │                                 ▲
                                 │ insert one row per prediction   │ reads the last N requests
                                 ▼                                 │
                           PostgreSQL: prediction_logs ────────────┘
```

| Path | What it does |
|---|---|
| `training/train.py` | Cleans data, trains the imputer + Random Forest Pipeline, logs to MLflow, promotes the best run |
| `training/run_experiments.sh` / `.bat` | Trains 5 hyperparameter combinations |
| `app/main.py` | FastAPI app and endpoints; loads model, baseline and database at startup |
| `app/schemas.py` | Pydantic request/response models with range checks |
| `app/model_loader.py` | Loads the Pipeline, runs one prediction |
| `app/database.py` | SQLAlchemy engine, `prediction_logs` table, startup retry, queries |
| `app/drift.py` | Mean-shift score and Kolmogorov-Smirnov test per feature |
| `app/config.py` | Settings from environment variables |
| `scripts/simulate_traffic.py` | Sends normal or deliberately drifted requests |
| `tests/` | pytest suite (runs without Docker, using SQLite) |

## Prerequisites

- **Docker route:** Docker Desktop (includes Docker Compose v2).
- **Local route:** Python 3.11 and a PostgreSQL server you can connect to (any local install, or
  `docker run -d -p 5432:5432 -e POSTGRES_USER=mlapi -e POSTGRES_PASSWORD=change-me -e POSTGRES_DB=predictions postgres:16`).
- For the traffic simulation and tests on the host: the Python packages in `requirements.txt`.

## Run it with Docker (recommended)

```bash
cp .env.example .env            # then edit POSTGRES_PASSWORD in .env
docker compose up --build       # db, mlflow and api; add -d to run in the background
```

- API: http://localhost:8000 (interactive docs at http://localhost:8000/docs)
- MLflow UI: http://localhost:5000

The image already contains the trained model from `models/`. To retrain inside Docker (logs the
runs to the `mlflow` service and writes the winner back to `./models`), then rebuild the API so the
new model is baked into the image:

```bash
docker compose run --rm --build trainer
docker compose up --build -d api
```

`--build` matters: `docker compose run` reuses an existing trainer image, so without it a changed
`train.py` would not be picked up. MLflow prints a harmless hint about integer columns in the
model's output schema (the predicted class is 0 or 1).

Stop everything with `docker compose down` (add `-v` to also delete the logged predictions).

## Run it locally (no Docker)

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # set DATABASE_URL to your PostgreSQL
```

Load `.env` into the current shell (the app reads plain environment variables):

```bash
set -a; source .env; set +a          # macOS / Linux
```
```powershell
# Windows PowerShell
Get-Content .env | Where-Object { $_ -match '^[A-Z_]+=' } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
```

1. Start the MLflow server (terminal 1):
   ```bash
   mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow_data/mlflow.db --artifacts-destination ./mlflow_data/artifacts
   ```
2. Train (terminal 2). Either all experiments or a single run:
   ```bash
   sh training/run_experiments.sh                 # Windows: training\run_experiments.bat
   python training/train.py --n-estimators 500 --max-depth 10 --min-samples-leaf 3
   ```
3. Start the API (terminal 2):
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
4. Run the tests (no database or server needed):
   ```bash
   pytest -q
   ```

## Endpoints

On Windows PowerShell, type `curl.exe` instead of `curl`, or use the interactive docs at `/docs`.

### `POST /predict`

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":138,"BloodPressure":62,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.127,"Age":47}'
```
```json
{"prediction":1,"label":"high risk","probability":0.5597,"model_version":"rf-v1","latency_ms":31.815}
```

A value of 0 in Glucose, BloodPressure, SkinThickness, Insulin or BMI means "not measured" and is
imputed exactly as in training. Invalid input returns HTTP 422 naming the problem:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"Pregnancies":2,"Glucose":138,"BloodPressure":62,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.127,"Age":150}'
```
```json
{"detail":[{"type":"less_than_equal","loc":["body","Age"],"msg":"Input should be less than or equal to 120","input":150,"ctx":{"le":120}}]}
```

### `GET /health`

```bash
curl http://localhost:8000/health
```
```json
{"status":"ok","model_loaded":true,"database_connected":true}
```

### `GET /metrics-summary`

```bash
curl http://localhost:8000/metrics-summary
```
```json
{
  "total_predictions": 203,
  "avg_latency_ms": 31.603,
  "p95_latency_ms": 44.246,
  "positive_rate": 0.6552,
  "drift": {
    "status": "retrain_recommended",
    "samples": 100,
    "drifted_features": ["Glucose", "BMI", "Age"],
    "features": [
      {"name": "Glucose", "live_mean": 182.16, "baseline_mean": 121.6735, "drift_score": 2.0228,
       "ks_p_value": 0.0, "mean_shift_drifted": true, "ks_drifted": true, "drifted": true},
      "... one entry per feature ..."
    ]
  }
}
```

The drift report is recomputed whenever new predictions have arrived since the last check.
Returns 503 if the database is unavailable.

### `POST /drift/check`

Forces a fresh drift calculation and returns the same report as the `drift` field above.

```bash
curl -X POST http://localhost:8000/drift/check
```

With fewer than 30 logged predictions it returns `{"status":"insufficient_data","samples":12,"drifted_features":[],"features":[]}`.

## Drift demo

With the API running (either route):

```bash
python scripts/simulate_traffic.py --n 100 --mode normal --seed 42
curl http://localhost:8000/metrics-summary      # drift.status: "ok"

python scripts/simulate_traffic.py --n 100 --mode drifted --seed 42
curl http://localhost:8000/metrics-summary      # drift.status: "retrain_recommended"
```

`normal` mode samples real rows from the dataset with 2% noise. `drifted` mode adds +60 to Glucose,
+12 to BMI and +20 to Age, simulating an older, heavier, more diabetic population. The monitor
looks at the last 100 requests, so after the drifted batch it sees only drifted traffic. Real output:

```
Mode: normal | sent: 100 | succeeded: 100 | failed: 0
Average model latency: 32.04 ms | high-risk predictions: 38/100 | wall time: 6.2 s
drift status: ok | samples: 100 | drifted: []

Mode: drifted | sent: 100 | succeeded: 100 | failed: 0
Average model latency: 31.05 ms | high-risk predictions: 92/100 | wall time: 5.9 s
drift status: retrain_recommended | samples: 100 | drifted: ['Glucose', 'BMI', 'Age']
```

`--seed` makes a run repeatable. Without it, normal traffic occasionally reports `warning` (see
"limits" below for why).

## Model results

Five runs from `run_experiments` (test set = 154 patients, stratified 80/20 split, `random_state=42`):

| n_estimators | max_depth | min_samples_leaf | test F1 | test ROC-AUC | 5-fold CV F1 |
|---|---|---|---|---|---|
| **500** | **10** | **3** | **0.679** | **0.827** | 0.679 |
| 100 | 4 | 1 | 0.672 | 0.822 | 0.699 |
| 200 | 6 | 2 | 0.672 | 0.819 | 0.678 |
| 300 | 8 | 4 | 0.667 | 0.824 | 0.692 |
| 300 | None | 1 | 0.606 | 0.817 | 0.628 |

The winner (bold) has accuracy 0.766, precision 0.655 and recall 0.704. `class_weight="balanced"`
is used because only about 35% of patients are positive; without it recall was around 0.54.

## Design decisions

**Why FastAPI.** It turns Python type hints into request validation, JSON serialisation and
OpenAPI docs (`/docs`) automatically, it is fast, and its lifespan hook is a clean place to load
the model once.

**Why Pydantic validation.** A model will happily return a number for nonsense input (Age -5,
Glucose 5000). Validating at the edge with explicit ranges rejects bad requests with a 422 that
says exactly which field is wrong, before they reach the model or pollute the logs.

**Why the model and preprocessing are one Pipeline.** Training treats 0 as missing in five columns
and fills it with the training median. If the API re-implemented that by hand, any difference
(a different median, a forgotten column) would silently change predictions: this is
training/serving skew. Saving the imputer and the Random Forest as one `Pipeline` means the API
runs exactly the fitted steps, including the medians learned from the training split only.

**Why log every prediction.** Without logs you cannot answer "how many requests, how fast, what
did we predict?", you cannot audit a decision later, and you have no live data to check for drift.
Each row stores the inputs, output, probability, latency and model version. Logging is
best-effort: if the database is down the API still returns the prediction and logs a warning.

**What MLflow gives us.** Every run records its hyperparameters, metrics, the model artifact and
the drift baseline, so runs can be compared side by side in the UI and any model can be traced
back to the exact settings that produced it. Just saving a `.joblib` file loses that history.
`train.py` asks MLflow for the run with the best test F1 (ROC-AUC breaks ties) and only that run
is written to `models/model.joblib`.

**Why load the model at startup.** Unpickling a 500-tree forest takes far longer than a
prediction. Loading it once in the FastAPI lifespan keeps per-request latency at the model call
itself (~30 ms here), and a missing model fails the startup loudly instead of failing every request.
The forest is also switched to a single thread for serving: for one row, starting worker threads
cost more than they saved (latency dropped from ~150 ms to ~30 ms).

**Why the database retry at startup.** In Docker Compose the API container can start before
PostgreSQL accepts connections. The API retries for up to 30 seconds (`DB_CONNECT_TIMEOUT_S`) and
Compose also waits for the database healthcheck.

**How drift detection works.** `train.py` saves, for each feature, the training mean, standard
deviation, min, max and a 500-value sample (after cleaning). The monitor takes the last 100 logged
requests (`DRIFT_WINDOW`), applies the same zero-to-median cleaning, and per feature computes:

1. **Mean shift:** `|live_mean - baseline_mean| / baseline_std`. Flagged above 2.0
   (`DRIFT_THRESHOLD`). Easy to explain, but it only notices a change in the average: a feature
   whose spread doubles while its mean stays put scores 0.
2. **Kolmogorov-Smirnov test:** compares the whole live distribution with the training sample and
   returns a p-value, the probability of seeing a difference this large if both came from the same
   distribution. Flagged below 0.01 (`DRIFT_KS_P_VALUE`).

A feature is drifted if either check fires; 0 drifted features is `ok`, 1-2 is `warning`, 3+ is
`retrain_recommended`. The demo shows why both are reported: BMI and Age moved by 1.7 standard
deviations, under the mean-shift threshold, but the KS test catches them with p < 0.0001.

**Limits of the drift monitor (be honest about these):**
- It detects **input** shift, not accuracy loss. A shifted population might still be predicted
  well, and accuracy can fall with no input shift (the relationship between inputs and outcome
  changes, "concept drift").
- There are **no ground-truth labels** in this system, so real accuracy is never measured.
- Mean shift is crude (blind to spread or shape changes); KS is better but tests features one at
  a time and misses changes in how features relate to each other.
- **Small samples are noisy.** That is why fewer than 30 requests gives `insufficient_data`.
  Also, eight tests at p < 0.01 each means roughly a 5-8% chance that perfectly normal traffic
  flags one feature (`warning`). A Bonferroni correction (0.01 / 8) would reduce that.
- KS assumes continuous values; integer features like Age and Pregnancies have ties, which makes
  the test somewhat conservative.

## What I would improve next

- **Authentication:** API keys or OAuth2 on `/predict`, and rate limiting.
- **Prometheus and Grafana:** expose request counts and latency histograms on `/metrics` and
  build dashboards and alerts instead of a JSON summary computed from the database.
- **MLflow Model Registry:** register models with versions and stages (Staging/Production) and have
  the API load "the Production model" instead of a file baked into the image.
- **Automated retraining:** a scheduled job that runs the drift check and, when it says
  `retrain_recommended` and fresh labelled data exists, retrains and proposes the new model.
- **Collect ground truth:** store real outcomes when they become known, to measure live accuracy.
- **CI/CD with GitHub Actions:** run `pytest` and build the image on every push; deploy on merge.
- **Cloud deployment:** managed PostgreSQL, the API on a container service (Cloud Run, ECS, Azure
  Container Apps), MLflow artifacts in object storage, and several API replicas behind a load
  balancer.
