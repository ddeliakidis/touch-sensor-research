#!/usr/bin/env python3
"""
position_predictor.py
======================
Train, save, and use a calibrated position predictor.

Given a raw 3-channel sensor reading (pa0, pa5, pa6), predict the physical
(x, y) position in mm.

Usage
-----
  # Train on the calibration CSV and save the model
  python position_predictor.py train

  # Predict from one raw reading on the command line
  python position_predictor.py predict 510 521 598

  # Use as a library
  from position_predictor import load_predictor
  predict_xy = load_predictor()
  x, y = predict_xy(510, 521, 598)

Outputs
-------
  models/position_predictor.joblib   trained model + feature spec
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH       = Path("sensor_position_calibration_20260419_191648.csv")
MODEL_PATH      = Path("models/position_predictor.joblib")
EDGE_RESID_NSIG = 4.0      # within-point outlier threshold (matches pipeline)

# Order matters: this is the column order the model expects at predict time.
FEATURE_COLS = [
    "raw_pa0", "raw_pa5", "raw_pa6",
    "ratio_pa5_pa0", "ratio_pa6_pa0", "ratio_pa5_pa6",
    "poly_pa5sq", "poly_pa0pa5", "poly_pa0pa6",
]
TARGET_COLS = ["cmd_x_mm", "cmd_y_mm"]


# ──────────────────────────────────────────────────────────────────────────────
# Feature engineering — same definitions as the pipeline's "combined" set
# ──────────────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ratio_pa5_pa0"] = df["raw_pa5"] / df["raw_pa0"]
    df["ratio_pa6_pa0"] = df["raw_pa6"] / df["raw_pa0"]
    df["ratio_pa5_pa6"] = df["raw_pa5"] / df["raw_pa6"]
    df["poly_pa5sq"]    = df["raw_pa5"] ** 2
    df["poly_pa0pa5"]   = df["raw_pa0"] * df["raw_pa5"]
    df["poly_pa0pa6"]   = df["raw_pa0"] * df["raw_pa6"]
    return df


def features_from_raw(pa0: float, pa5: float, pa6: float) -> np.ndarray:
    """Build a single-row feature vector in the order FEATURE_COLS expects."""
    return np.array([[
        pa0, pa5, pa6,
        pa5 / pa0, pa6 / pa0, pa5 / pa6,
        pa5 * pa5, pa0 * pa5, pa0 * pa6,
    ]])


def add_edge_failure_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Within-point MAD-based outlier flag (matches sensor_calibration_pipeline)."""
    df = df.copy()
    fail = pd.Series(False, index=df.index)
    for c in ["raw_pa0", "raw_pa5", "raw_pa6"]:
        group_med = df.groupby(["cmd_x_mm", "cmd_y_mm"])[c].transform("median")
        resid     = df[c] - group_med
        scale     = 1.4826 * np.median(np.abs(resid - np.median(resid)))
        if scale > 1e-9:
            fail |= resid.abs() > EDGE_RESID_NSIG * scale
    df["edge_fail"] = fail
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────────────────────
def train(data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH) -> None:
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)

    df = add_features(df)
    df = add_edge_failure_flag(df)

    n_total = len(df)
    clean = df[~df["edge_fail"]].dropna(subset=FEATURE_COLS + TARGET_COLS)
    print(f"  {n_total} rows  ->  {len(clean)} clean rows after edge_fail filter")

    X = clean[FEATURE_COLS].to_numpy()
    y = clean[TARGET_COLS].to_numpy()

    # RandomForest was the best performer on the model leaderboard
    # (XY-MAE = 0.87 mm, vs 0.97 mm for GradientBoosting).
    # Train on ALL clean data — no held-out split, since this is the deployed model.
    model = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1,
        random_state=0,
    )
    print("Training RandomForestRegressor ...")
    model.fit(X, y)

    # Quick sanity-check on the training data (in-sample MAE — not generalization).
    pred = model.predict(X)
    in_sample_mae = np.abs(pred - y).mean(axis=0)
    print(f"  In-sample MAE   x={in_sample_mae[0]:.3f} mm   y={in_sample_mae[1]:.3f} mm")
    print("  (For honest generalization MAE see datasets/model_evaluation.csv:"
          " RF + combined -> ~0.87 mm)")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "feature_cols": FEATURE_COLS,
               "target_cols": TARGET_COLS, "n_train": len(clean)}
    joblib.dump(payload, model_path)
    print(f"Saved: {model_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Predict (library + CLI)
# ──────────────────────────────────────────────────────────────────────────────
def load_predictor(model_path: Path = MODEL_PATH):
    """
    Load the saved model and return a callable: predict_xy(pa0, pa5, pa6) -> (x, y).
    """
    payload = joblib.load(model_path)
    model = payload["model"]

    def predict_xy(pa0: float, pa5: float, pa6: float) -> tuple[float, float]:
        feat = features_from_raw(pa0, pa5, pa6)
        x, y = model.predict(feat)[0]
        return float(x), float(y)

    return predict_xy


def predict_cli(pa0: float, pa5: float, pa6: float,
                model_path: Path = MODEL_PATH) -> None:
    if not model_path.exists():
        print(f"No model at {model_path}.  Run:  python {sys.argv[0]} train",
              file=sys.stderr)
        sys.exit(1)
    predict_xy = load_predictor(model_path)
    x, y = predict_xy(pa0, pa5, pa6)
    print(f"raw=(pa0={pa0}, pa5={pa5}, pa6={pa6})  ->  x={x:.2f} mm  y={y:.2f} mm")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train and save the model")
    p_train.add_argument("--data",  type=Path, default=DATA_PATH)
    p_train.add_argument("--model", type=Path, default=MODEL_PATH)

    p_pred = sub.add_parser("predict", help="predict (x, y) from raw pa0 pa5 pa6")
    p_pred.add_argument("pa0", type=float)
    p_pred.add_argument("pa5", type=float)
    p_pred.add_argument("pa6", type=float)
    p_pred.add_argument("--model", type=Path, default=MODEL_PATH)

    args = p.parse_args()
    if args.cmd == "train":
        train(args.data, args.model)
    elif args.cmd == "predict":
        predict_cli(args.pa0, args.pa5, args.pa6, args.model)


if __name__ == "__main__":
    main()
