"""Send realistic (or deliberately shifted) patient requests to the API.

Examples:
    python scripts/simulate_traffic.py --n 100 --mode normal
    python scripts/simulate_traffic.py --n 100 --mode drifted --url http://localhost:8000
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "diabetes.csv"
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
INTEGER_FEATURES = ["Pregnancies", "Age"]
# Same limits as the API's Pydantic schema, so simulated requests are always valid.
UPPER_LIMITS = {
    "Pregnancies": 20, "Glucose": 300, "BloodPressure": 200, "SkinThickness": 100,
    "Insulin": 900, "BMI": 80, "DiabetesPedigreeFunction": 3, "Age": 120,
}
# "drifted" mode: a sicker, older, heavier patient population than the training data.
DRIFT_SHIFT = {"Glucose": 60, "BMI": 12, "Age": 20}


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Simulate traffic against the prediction API.")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    parser.add_argument("--mode", choices=["normal", "drifted"], default="normal")
    parser.add_argument("--url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--seed", type=int, default=None, help="Random seed for repeatable runs")
    return parser.parse_args()


def make_rows(n: int, mode: str, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n real rows, add small noise, and shift them if mode is 'drifted'."""
    data = pd.read_csv(DATA_PATH)[FEATURES]
    rows = data.sample(n=n, replace=True, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)

    # Small noise (2% of each column's spread) so requests are not exact copies of the dataset.
    # Zeros mean "not measured" and are kept as zeros.
    for column in FEATURES:
        if column in INTEGER_FEATURES:
            continue
        noise = rng.normal(0, 0.02 * data[column].std(), size=n)
        rows[column] = np.where(rows[column] == 0, 0, rows[column] + noise)

    if mode == "drifted":
        for column, shift in DRIFT_SHIFT.items():
            rows[column] = rows[column] + shift

    for column in FEATURES:
        rows[column] = rows[column].clip(lower=0, upper=UPPER_LIMITS[column])
    rows[INTEGER_FEATURES] = rows[INTEGER_FEATURES].round().astype(int)
    rows["Age"] = rows["Age"].clip(lower=1)
    return rows


def main() -> None:
    """Send the requests one by one and print a summary."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = make_rows(args.n, args.mode, rng)
    endpoint = f"{args.url.rstrip('/')}/predict"

    successes, latencies, high_risk = 0, [], 0
    started = time.perf_counter()
    with requests.Session() as session:
        for payload in rows.to_dict(orient="records"):
            try:
                response = session.post(endpoint, json=payload, timeout=10)
                response.raise_for_status()
            except requests.RequestException as exc:
                print(f"Request failed: {exc}")
                continue
            body = response.json()
            successes += 1
            latencies.append(body["latency_ms"])
            high_risk += body["prediction"]
    elapsed = time.perf_counter() - started

    print(f"Mode: {args.mode} | sent: {args.n} | succeeded: {successes} | failed: {args.n - successes}")
    if latencies:
        print(f"Average model latency: {np.mean(latencies):.2f} ms | "
              f"high-risk predictions: {high_risk}/{successes} | wall time: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
