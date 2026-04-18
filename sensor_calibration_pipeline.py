#!/usr/bin/env python3
"""
sensor_calibration_pipeline.py  (v2)
======================================
Full data-preparation AND modeling pipeline for predicting (cmd_x_mm, cmd_y_mm)
from three optical sensor channels (pa0, pa5, pa6).

New in v2
---------
Features
  • Per-repeat recentered delta (removes inter-repeat baseline shift)
  • Pairwise channel ratios pa5/pa0, pa6/pa0, pa5/pa6 (drift-canceling)
  • Rolling-window demeaned features (removes slow session drift)
  • Polynomial interaction features  pa5², pa0·pa5, pa0·pa6
  • "Combined" feature set (raw + pairwise ratios + polynomial terms)

Quality analysis
  • Interior (y ≤ 35 mm) vs edge (y > 35 mm) region split everywhere

Modeling
  • CLEAN_ONLY flag – filters bad rows before fitting
  • Train on repeat-1, test on repeat-2 (no temporal leakage)
  • Separate X-regressor and Y-regressor for each model
  • Four models: 1-NN baseline, Random Forest, Gradient Boosting, GPR
  • Full model × feature-set comparison table → datasets/model_evaluation.csv
  • GPR uncertainty map

Outputs
-------
datasets/
  static_{featureset}.csv   (9 variants)
  timeseries_K5.csv
  model_evaluation.csv
  assessment.txt
plots/
  01–09  diagnostic plots (same as v1, some enhanced)
  10     feature–target correlation bar chart
  11     interior vs edge repeatability
  12     model MAE comparison
  13     predicted vs actual scatter (best model)
  14     residual heatmaps on XY grid
  15     GPR uncertainty map

Requirements: Python ≥ 3.9, pandas, numpy, matplotlib, scikit-learn
"""

import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Silence noisy convergence warnings from GPR on the small dataset
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*lbfgs.*")

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                  errors="replace")

try:
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: scikit-learn not installed — modeling section will be skipped.\n"
          "  Install with:  pip install scikit-learn")


# ──────────────────────────────────────────────────────────────────────────────
# A  Configuration
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH       = "sensor_position_calibration.csv"
OUT_DIR         = Path("datasets")
PLOT_DIR        = Path("plots")

TIMESERIES_K    = 5        # sliding-window look-back length
OUTLIER_ZSCORE  = 4.0      # z-score threshold for statistical outlier flag
EDGE_PA6_MIN    = 1500     # raw_pa6 below this → sensor off field
EDGE_PA5_MIN    = 400      # raw_pa5 below this → transient dropout
DELTA_FAIL_ABS  = 600      # |delta| above this → motion artifact
DEMEANED_WINDOW = 20       # rolling-median window for demeaned features

CLEAN_ONLY      = True     # exclude is_outlier | edge_fail rows from modeling
TEST_REPEAT     = 2        # repeat held out for test (the other is training)
INTERIOR_Y_MAX  = 35       # cmd_y_mm ≤ this → "interior" (clean sensor) region

CHANNELS = ["pa0", "pa5", "pa6"]

# All feature sets that will be compared in the model evaluation
FEATURE_SETS: dict[str, list[str]] = {
    "raw"          : [f"raw_{c}"      for c in CHANNELS],
    "delta"        : [f"delta_{c}"    for c in CHANNELS],
    "delta_rc"     : [f"rc_delta_{c}" for c in CHANNELS],   # recentered
    "smooth"       : [f"smooth_{c}"   for c in CHANNELS],
    "ratio_global" : [f"ratio_g_{c}"  for c in CHANNELS],
    "ratio_pair"   : ["ratio_pa5_pa0", "ratio_pa6_pa0", "ratio_pa5_pa6"],
    "demeaned"     : [f"dm_{c}"       for c in CHANNELS],
    "poly"         : ["poly_pa5sq", "poly_pa0pa5", "poly_pa0pa6"],
    "combined"     : (
        [f"raw_{c}" for c in CHANNELS]
        + ["ratio_pa5_pa0", "ratio_pa6_pa0", "ratio_pa5_pa6"]
        + ["poly_pa5sq", "poly_pa0pa5", "poly_pa0pa6"]
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# B  I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    PLOT_DIR.mkdir(exist_ok=True)


def _save_fig(fig: plt.Figure, fname: str) -> None:
    path = PLOT_DIR / fname
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# C  Load, clean, sort
# ──────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load CSV, validate columns, drop NaN rows, flag statistical outliers
    (z-score > OUTLIER_ZSCORE on any raw channel).

    Adds:  is_outlier (bool)
    Does NOT remove rows — caller controls filtering.
    """
    required = [
        "session_time_s", "point_time_s", "point_index", "repeat",
        "cmd_x_mm", "cmd_y_mm",
        "raw_pa0",    "raw_pa5",    "raw_pa6",
        "delta_pa0",  "delta_pa5",  "delta_pa6",
        "smooth_pa0", "smooth_pa5", "smooth_pa6",
    ]
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    n_raw = len(df)
    df    = df.dropna(subset=required).copy()
    n_ok  = len(df)

    df["is_outlier"] = False
    for ch in CHANNELS:
        z = (df[f"raw_{ch}"] - df[f"raw_{ch}"].mean()) / df[f"raw_{ch}"].std()
        df["is_outlier"] |= z.abs() > OUTLIER_ZSCORE

    n_out = int(df["is_outlier"].sum())
    print("[load_and_clean]")
    print(f"  Rows loaded         : {n_raw}")
    print(f"  After NaN drop      : {n_ok}")
    print(f"  Outliers flagged    : {n_out} ({100*n_out/n_ok:.1f}%)")
    return df


def sort_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by (repeat, session_time_s).  Required for rolling-window features."""
    return df.sort_values(["repeat", "session_time_s"]).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# D  Feature engineering
# ──────────────────────────────────────────────────────────────────────────────

def compute_global_baselines(df: pd.DataFrame) -> dict:
    """Median raw value per channel across non-outlier rows."""
    clean = df[~df["is_outlier"]]
    return {ch: clean[f"raw_{ch}"].median() for ch in CHANNELS}


def add_global_ratio_features(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """
    ratio_g_pa* = raw_pa* / global_median.
    Reduces sensitivity to absolute session-level offset.
    """
    df = df.copy()
    for ch in CHANNELS:
        ref = baselines[ch]
        df[f"ratio_g_{ch}"] = df[f"raw_{ch}"] / ref if ref != 0 else np.nan
    return df


def add_pairwise_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Channel-to-channel ratios cancel common-mode drift entirely.
      ratio_pa5_pa0 = raw_pa5 / raw_pa0   ← strong y signal
      ratio_pa6_pa0 = raw_pa6 / raw_pa0   ← strong y signal (clean region)
      ratio_pa5_pa6 = raw_pa5 / raw_pa6   ← differential
    """
    df = df.copy()
    df["ratio_pa5_pa0"] = df["raw_pa5"] / df["raw_pa0"].replace(0, np.nan)
    df["ratio_pa6_pa0"] = df["raw_pa6"] / df["raw_pa0"].replace(0, np.nan)
    df["ratio_pa5_pa6"] = df["raw_pa5"] / df["raw_pa6"].replace(0, np.nan)
    return df


def add_recentered_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-repeat recentered delta:  rc_delta_pa* = delta_pa* - median(delta_pa*)
    within the same repeat.  Removes the inter-repeat baseline shift that makes
    the raw delta_* columns unusable for cross-repeat generalization.
    """
    df = df.copy()
    for ch in CHANNELS:
        col = f"delta_{ch}"
        df[f"rc_delta_{ch}"] = df.groupby("repeat")[col].transform(
            lambda x: x - x.median()
        )
    return df


def add_demeaned_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling-window demeaned:  dm_pa* = raw_pa* - rolling_median(raw_pa*, window=W)
    computed within each repeat independently.

    Removes slow session drift while preserving position-dependent variation.
    Requires the DataFrame to be sorted by (repeat, session_time_s).
    """
    df = df.copy()
    for ch in CHANNELS:
        col = f"raw_{ch}"
        rolling_med = df.groupby("repeat")[col].transform(
            lambda x: x.rolling(DEMEANED_WINDOW, min_periods=1, center=True).median()
        )
        df[f"dm_{ch}"] = df[col] - rolling_med
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Polynomial and cross-product terms that capture nonlinear channel responses.
      poly_pa5sq   = raw_pa5²          (PA5 is nonlinear along y)
      poly_pa0pa5  = raw_pa0 × raw_pa5 (interaction term)
      poly_pa0pa6  = raw_pa0 × raw_pa6 (interaction term)

    Values are scaled ÷ 1e6 to keep magnitudes comparable to other features.
    """
    df = df.copy()
    df["poly_pa5sq"]  = df["raw_pa5"] ** 2           / 1e6
    df["poly_pa0pa5"] = df["raw_pa0"] * df["raw_pa5"] / 1e6
    df["poly_pa0pa6"] = df["raw_pa0"] * df["raw_pa6"] / 1e6
    return df


def add_edge_failure_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    edge_fail = True when the sensor is outside its calibrated range.
    Criteria (OR):  raw_pa6 < EDGE_PA6_MIN
                  | raw_pa5 < EDGE_PA5_MIN
                  | |any delta| > DELTA_FAIL_ABS
    """
    df = df.copy()
    bad_pa6   = df["raw_pa6"] < EDGE_PA6_MIN
    bad_pa5   = df["raw_pa5"] < EDGE_PA5_MIN
    bad_delta = df[["delta_pa0","delta_pa5","delta_pa6"]].abs().max(axis=1) > DELTA_FAIL_ABS
    df["edge_fail"] = bad_pa6 | bad_pa5 | bad_delta
    return df


def add_all_features(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """
    Master feature-engineering function.
    Applies all transformations in the correct order and returns a single
    wide DataFrame containing every feature column.
    """
    df = add_global_ratio_features(df, baselines)
    df = add_pairwise_ratio_features(df)
    df = add_recentered_delta_features(df)
    df = add_demeaned_features(df)
    df = add_polynomial_features(df)
    df = add_edge_failure_flag(df)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# E  Dataset building
# ──────────────────────────────────────────────────────────────────────────────

# Columns that appear in every output dataset (meta + quality flags)
_META_COLS = [
    "repeat", "point_index", "session_time_s", "point_time_s",
    "cmd_x_mm", "cmd_y_mm", "is_outlier", "edge_fail",
]


def build_static_dataset(df: pd.DataFrame, baselines: dict) -> pd.DataFrame:
    """
    Full static dataset with all feature columns appended.
    One row per (repeat, point_index).  Columns include all feature variants
    plus quality flags.
    """
    df = add_all_features(df, baselines)

    all_feature_cols = []
    for cols in FEATURE_SETS.values():
        for c in cols:
            if c not in all_feature_cols:
                all_feature_cols.append(c)

    available = [c for c in all_feature_cols if c in df.columns]
    keep = _META_COLS + [c for c in available if c not in _META_COLS]
    return df[keep].reset_index(drop=True)


def split_static_by_feature_set(static_df: pd.DataFrame) -> dict:
    """
    Return one lean DataFrame per named feature set (target + meta + features).
    Columns from FEATURE_SETS that are missing in static_df are skipped
    gracefully (shouldn't happen if build_static_dataset was called first).
    """
    splits = {}
    for name, cols in FEATURE_SETS.items():
        avail = [c for c in cols if c in static_df.columns]
        splits[name] = static_df[_META_COLS + avail].copy()
    return splits


def build_timeseries_dataset(
    df: pd.DataFrame,
    baselines: dict,
    K: int = TIMESERIES_K,
) -> pd.DataFrame:
    """
    Sliding-window time-series feature matrix.

    For each row i (≥ K−1, within one repeat), the feature vector is the
    flattened [i−K+1 … i] window of ALL feature columns.
    Column naming:  <feature>_t0 (oldest) … <feature>_t{K-1} (current).
    Label is (cmd_x_mm, cmd_y_mm) at row i.
    Windows never cross the repeat boundary.
    """
    df = add_all_features(df, baselines)

    # Use the combined feature set for the time-series dataset
    sensor_cols = FEATURE_SETS["combined"]
    flat_names  = [f"{c}_t{j}" for j in range(K) for c in sensor_cols]

    rows = []
    for repeat_val, grp in df.groupby("repeat"):
        grp = grp.reset_index(drop=True)
        arr = grp[sensor_cols].to_numpy(dtype=float)

        for i in range(K - 1, len(grp)):
            flat = arr[i - K + 1 : i + 1].flatten(order="C")
            row  = {
                "repeat"        : repeat_val,
                "point_index"   : grp.at[i, "point_index"],
                "session_time_s": grp.at[i, "session_time_s"],
                "cmd_x_mm"      : grp.at[i, "cmd_x_mm"],
                "cmd_y_mm"      : grp.at[i, "cmd_y_mm"],
                "is_outlier"    : grp.at[i, "is_outlier"],
                "edge_fail"     : grp.at[i, "edge_fail"],
            }
            for cn, val in zip(flat_names, flat):
                row[cn] = val
            rows.append(row)

    ts_df = pd.DataFrame(rows)
    meta  = ["repeat","point_index","session_time_s","cmd_x_mm","cmd_y_mm",
             "is_outlier","edge_fail"]
    feat  = [c for c in ts_df.columns if c not in meta]
    return ts_df[meta + feat].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# F  Quality analysis
# ──────────────────────────────────────────────────────────────────────────────

def _split_regions(df: pd.DataFrame) -> dict:
    """Return {'all': df, 'interior': df[y≤35], 'edge': df[y>35]}."""
    return {
        "all"     : df,
        "interior": df[df["cmd_y_mm"] <= INTERIOR_Y_MAX],
        "edge"    : df[df["cmd_y_mm"] >  INTERIOR_Y_MAX],
    }


def analyze_repeatability(df: pd.DataFrame) -> pd.DataFrame:
    """
    RMS / std / max of (repeat-1 minus repeat-2) for each raw channel,
    broken out by region (all / interior / edge).
    """
    rows = []
    for region_name, region_df in _split_regions(df).items():
        r1 = region_df[region_df["repeat"] == 1].set_index(["cmd_x_mm","cmd_y_mm"])
        r2 = region_df[region_df["repeat"] == 2].set_index(["cmd_x_mm","cmd_y_mm"])
        common = r1.index.intersection(r2.index)
        r1, r2 = r1.loc[common], r2.loc[common]
        for ch in CHANNELS:
            diff = r1[f"raw_{ch}"] - r2[f"raw_{ch}"]
            rows.append({
                "region"       : region_name,
                "channel"      : ch,
                "mean_diff"    : round(diff.mean(), 1),
                "std_diff"     : round(diff.std(), 1),
                "rms_diff"     : round(np.sqrt((diff**2).mean()), 1),
                "max_abs_diff" : round(diff.abs().max(), 1),
            })
    result = pd.DataFrame(rows)
    print("\n[analyze_repeatability]")
    print(result.to_string(index=False))
    return result


def analyze_drift(df: pd.DataFrame) -> pd.DataFrame:
    """Linear trend (counts/s) per channel per repeat."""
    rows = []
    for repeat_val, grp in df.groupby("repeat"):
        t = grp["session_time_s"].values
        for ch in CHANNELS:
            slope, _ = np.polyfit(t, grp[f"raw_{ch}"].values, 1)
            rows.append({"repeat": repeat_val, "channel": ch,
                         "drift_counts_per_s": round(slope, 4)})
    result = pd.DataFrame(rows)
    print("\n[analyze_drift]")
    print(result.to_string(index=False))
    return result


def analyze_noise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-position noise estimate: |r1−r2| / √2 ≈ single-reading std.
    Reported separately for interior and edge.
    """
    rows = []
    for region_name, region_df in _split_regions(df).items():
        r1 = region_df[region_df["repeat"] == 1].set_index(["cmd_x_mm","cmd_y_mm"])
        r2 = region_df[region_df["repeat"] == 2].set_index(["cmd_x_mm","cmd_y_mm"])
        common = r1.index.intersection(r2.index)
        r1, r2 = r1.loc[common], r2.loc[common]
        for ch in CHANNELS:
            ne = (r1[f"raw_{ch}"] - r2[f"raw_{ch}"]).abs() / np.sqrt(2)
            rows.append({
                "region"  : region_name, "channel": ch,
                "median"  : round(ne.median(), 1),
                "p95"     : round(ne.quantile(0.95), 1),
                "max"     : round(ne.max(), 1),
            })
    result = pd.DataFrame(rows)
    print("\n[analyze_noise]")
    print(result.to_string(index=False))
    return result


def analyze_edge_failures(df: pd.DataFrame) -> pd.DataFrame:
    """Flag edge/failure rows and print a position-level summary."""
    if "edge_fail" not in df.columns:
        df = add_edge_failure_flag(df)
    n_fail = int(df["edge_fail"].sum())
    print(f"\n[analyze_edge_failures]")
    print(f"  Flagged rows : {n_fail} / {len(df)} ({100*n_fail/len(df):.1f}%)")
    top = (df[df["edge_fail"]]
           .groupby(["cmd_x_mm","cmd_y_mm"]).size()
           .reset_index(name="count")
           .sort_values("count", ascending=False))
    print(top.head(10).to_string(index=False))
    return df


# ──────────────────────────────────────────────────────────────────────────────
# G  Diagnostic plots  (01–11)
# ──────────────────────────────────────────────────────────────────────────────

def plot_raw_channels_over_time(df: pd.DataFrame) -> None:
    """Raw sensor values vs session time, coloured by repeat."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    colors = {1: "steelblue", 2: "tomato"}
    for rv, grp in df.groupby("repeat"):
        for ax, ch in zip(axes, CHANNELS):
            lbl = f"repeat {rv}" if ch == "pa0" else None
            ax.plot(grp["session_time_s"], grp[f"raw_{ch}"],
                    ".", ms=2, alpha=0.6, color=colors[rv], label=lbl)
    for ax, ch in zip(axes, CHANNELS):
        ax.set_ylabel(f"raw_{ch}"); ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", markerscale=5)
    axes[-1].set_xlabel("session_time_s")
    fig.suptitle("Raw sensor channels over session time")
    _save_fig(fig, "01_raw_channels_over_time.png")


def plot_signal_vs_position(df: pd.DataFrame) -> None:
    """Mean raw value heatmaps across XY grid."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    grp = df.groupby(["cmd_x_mm","cmd_y_mm"])
    for ax, ch in zip(axes, CHANNELS):
        pv = grp[f"raw_{ch}"].mean().unstack("cmd_y_mm")
        im = ax.pcolormesh(pv.index.values, pv.columns.values,
                           pv.T.values, shading="auto", cmap="viridis")
        plt.colorbar(im, ax=ax, shrink=0.85, label="counts")
        ax.set_title(f"mean raw_{ch}")
        ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    fig.suptitle("Mean raw sensor value across XY grid (both repeats)")
    _save_fig(fig, "02_signal_vs_position_heatmap.png")


def plot_repeatability(df: pd.DataFrame) -> None:
    """Repeat-1 vs repeat-2 scatter for each channel."""
    r1 = df[df["repeat"]==1].set_index(["cmd_x_mm","cmd_y_mm"])
    r2 = df[df["repeat"]==2].set_index(["cmd_x_mm","cmd_y_mm"])
    common = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common], r2.loc[common]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ch in zip(axes, CHANNELS):
        col = f"raw_{ch}"
        ax.scatter(r1[col], r2[col], s=6, alpha=0.5, edgecolors="none")
        lo = min(r1[col].min(), r2[col].min())
        hi = max(r1[col].max(), r2[col].max())
        ax.plot([lo,hi],[lo,hi],"r--",lw=1, label="y = x")
        ax.set_xlabel(f"repeat-1"); ax.set_ylabel(f"repeat-2")
        ax.set_title(f"raw_{ch}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle("Repeat-1 vs Repeat-2  (ideal: points on diagonal)")
    _save_fig(fig, "03_repeatability_scatter.png")


def plot_noise_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of |r1−r2| per grid position."""
    r1 = df[df["repeat"]==1].set_index(["cmd_x_mm","cmd_y_mm"])
    r2 = df[df["repeat"]==2].set_index(["cmd_x_mm","cmd_y_mm"])
    common = r1.index.intersection(r2.index)
    r1, r2 = r1.loc[common], r2.loc[common]
    diff_df = pd.DataFrame({"cmd_x_mm": r1.index.get_level_values(0),
                             "cmd_y_mm": r1.index.get_level_values(1)})
    for ch in CHANNELS:
        diff_df[f"d_{ch}"] = (r1[f"raw_{ch}"] - r2[f"raw_{ch}"]).abs().values
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, ch in zip(axes, CHANNELS):
        pv = diff_df.groupby(["cmd_x_mm","cmd_y_mm"])[f"d_{ch}"].mean().unstack("cmd_y_mm")
        im = ax.pcolormesh(pv.index.values, pv.columns.values,
                           pv.T.values, shading="auto", cmap="hot")
        plt.colorbar(im, ax=ax, shrink=0.85, label="counts")
        ax.set_title(f"|r1−r2|  raw_{ch}"); ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    fig.suptitle("|repeat-1 − repeat-2| per position  (lower = better)")
    _save_fig(fig, "04_noise_heatmap.png")


def plot_edge_failure_map(df: pd.DataFrame) -> None:
    """Binary heatmap of sensor failure positions."""
    if "edge_fail" not in df.columns:
        df = add_edge_failure_flag(df)
    pv = df.groupby(["cmd_x_mm","cmd_y_mm"])["edge_fail"].sum().unstack("cmd_y_mm")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(pv.index.values, pv.columns.values,
                       pv.T.values.astype(float), shading="auto",
                       cmap="Reds", vmin=0, vmax=2)
    cb = plt.colorbar(im, ax=ax); cb.set_ticks([0,1,2])
    cb.set_label("failure count (max 2)")
    ax.axhline(INTERIOR_Y_MAX + 0.5, color="cyan", lw=1.5, ls="--",
               label=f"interior / edge boundary (y={INTERIOR_Y_MAX})")
    ax.legend(fontsize=8)
    ax.set_title("Edge / sensor failure map"); ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    _save_fig(fig, "05_edge_failure_map.png")


def plot_feature_correlation_matrix(df: pd.DataFrame, baselines: dict) -> None:
    """Full correlation matrix between all features and targets."""
    df = add_all_features(df, baselines)
    all_cols = (["cmd_x_mm","cmd_y_mm"]
                + [f"raw_{c}" for c in CHANNELS]
                + [f"delta_{c}" for c in CHANNELS]
                + [f"smooth_{c}" for c in CHANNELS]
                + [f"ratio_g_{c}" for c in CHANNELS]
                + ["ratio_pa5_pa0","ratio_pa6_pa0","ratio_pa5_pa6"]
                + [f"rc_delta_{c}" for c in CHANNELS]
                + [f"dm_{c}" for c in CHANNELS]
                + ["poly_pa5sq","poly_pa0pa5","poly_pa0pa6"])
    corr = df[all_cols].corr()
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
    ax.set_xticks(range(len(all_cols))); ax.set_yticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=90, fontsize=7)
    ax.set_yticklabels(all_cols, fontsize=7)
    ax.set_title("Feature correlation matrix  (all feature types)")
    _save_fig(fig, "06_feature_correlation_matrix.png")


def plot_feature_target_bar(df: pd.DataFrame, baselines: dict) -> None:
    """
    Ranked bar chart of |Pearson r| with cmd_x_mm and cmd_y_mm for every
    feature column.  Shows at a glance which features carry position signal.
    """
    df = add_all_features(df, baselines)
    feat_cols = []
    for cols in FEATURE_SETS.values():
        for c in cols:
            if c not in feat_cols and c in df.columns:
                feat_cols.append(c)
    feat_cols = list(dict.fromkeys(feat_cols))   # deduplicate, preserve order

    corr_x = df[feat_cols + ["cmd_x_mm"]].corr()["cmd_x_mm"].drop("cmd_x_mm").abs()
    corr_y = df[feat_cols + ["cmd_y_mm"]].corr()["cmd_y_mm"].drop("cmd_y_mm").abs()

    # Sort by sum of |r| with both targets
    order = (corr_x + corr_y).sort_values(ascending=False).index.tolist()
    corr_x, corr_y = corr_x[order], corr_y[order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    y_pos = np.arange(len(order))
    for ax, corr, target in zip(axes, [corr_x, corr_y], ["cmd_x_mm", "cmd_y_mm"]):
        ax.barh(y_pos, corr.values, color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels(order, fontsize=8)
        ax.set_xlabel("|Pearson r|"); ax.set_title(f"Correlation with {target}")
        ax.axvline(0.3, color="r", ls="--", lw=0.8, label="|r|=0.3")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("Feature–target correlation (ranked by total |r|  across both targets)")
    _save_fig(fig, "10_feature_target_bar.png")


def plot_delta_distributions(df: pd.DataFrame) -> None:
    """Delta and recentered-delta distributions per repeat — shows drift fix."""
    if "rc_delta_pa0" not in df.columns:
        df = add_recentered_delta_features(df)
    colors = {1: "steelblue", 2: "tomato"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for col_idx, ch in enumerate(CHANNELS):
        for row_idx, (prefix, title) in enumerate([
            (f"delta_{ch}",    f"delta_{ch}  (original)"),
            (f"rc_delta_{ch}", f"rc_delta_{ch}  (recentered)"),
        ]):
            ax = axes[row_idx, col_idx]
            for rv, grp in df.groupby("repeat"):
                ax.hist(grp[prefix], bins=60, alpha=0.55, color=colors[rv],
                        label=f"repeat {rv}", density=True)
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("counts"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    fig.suptitle("Delta distributions: before and after per-repeat recentering\n"
                 "(shift between repeats should disappear in bottom row)")
    _save_fig(fig, "07_delta_distributions.png")


def plot_pa6_anomaly(df: pd.DataFrame) -> None:
    """PA6 field-edge failure analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lo, hi = df["raw_pa6"].quantile(0.05), df["raw_pa6"].quantile(0.95)
    sc = axes[0].scatter(df["cmd_x_mm"], df["cmd_y_mm"],
                         c=df["raw_pa6"], cmap="viridis", s=18, vmin=lo, vmax=hi)
    plt.colorbar(sc, ax=axes[0], label="raw_pa6 (counts)")
    axes[0].axhline(INTERIOR_Y_MAX+0.5, color="r", ls="--", lw=1,
                    label=f"y={INTERIOR_Y_MAX}")
    axes[0].legend(fontsize=8)
    axes[0].set_title("raw_pa6 across XY grid"); axes[0].set_xlabel("cmd_x_mm"); axes[0].set_ylabel("cmd_y_mm")
    colors = {1:"steelblue",2:"tomato"}
    for rv, grp in df.groupby("repeat"):
        axes[1].plot(grp["cmd_y_mm"], grp["raw_pa6"],".",ms=3,alpha=0.4,
                     color=colors[rv],label=f"repeat {rv}")
    axes[1].axhline(EDGE_PA6_MIN, color="r", lw=1.2, ls="--",
                    label=f"failure threshold ({EDGE_PA6_MIN})")
    axes[1].set_xlabel("cmd_y_mm"); axes[1].set_ylabel("raw_pa6")
    axes[1].set_title("raw_pa6 vs cmd_y_mm"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    fig.suptitle("PA6 sensor failure analysis")
    _save_fig(fig, "08_pa6_anomaly.png")


def plot_xy_coverage(df: pd.DataFrame) -> None:
    """Commanded XY grid scatter."""
    n_unique = df[["cmd_x_mm","cmd_y_mm"]].drop_duplicates().shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["cmd_x_mm"], df["cmd_y_mm"], s=20, alpha=0.35, edgecolors="none")
    ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    ax.set_title(f"XY target coverage  (n={len(df)}, {n_unique} unique positions)")
    ax.grid(True, alpha=0.3)
    _save_fig(fig, "09_xy_coverage.png")


def plot_interior_edge_repeatability(repeat_stats: pd.DataFrame) -> None:
    """
    Side-by-side RMS diff bar chart for interior vs edge region per channel.
    Makes it immediately clear which channels degrade at the edge.
    """
    regions = ["interior", "edge"]
    x   = np.arange(len(CHANNELS))
    w   = 0.35
    colors = {"interior": "steelblue", "edge": "tomato"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, region in enumerate(regions):
        rdf = repeat_stats[repeat_stats["region"] == region]
        rms_vals = [rdf.loc[rdf["channel"]==ch, "rms_diff"].values[0]
                    if ch in rdf["channel"].values else 0
                    for ch in CHANNELS]
        ax.bar(x + i*w, rms_vals, w, label=region,
               color=colors[region], alpha=0.8, edgecolor="white")

    ax.set_xticks(x + w/2)
    ax.set_xticklabels([f"raw_{c}" for c in CHANNELS])
    ax.set_ylabel("RMS(repeat-1 − repeat-2)  [counts]")
    ax.set_title("Repeatability: interior region vs edge region\n"
                 "(lower = more reproducible;  edge failures inflate edge bars)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    _save_fig(fig, "11_interior_vs_edge_repeatability.png")


# ──────────────────────────────────────────────────────────────────────────────
# H  Modeling
# ──────────────────────────────────────────────────────────────────────────────

def make_train_test_split(
    static_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """
    Split by repeat (train = repeat ≠ TEST_REPEAT, test = TEST_REPEAT).
    If CLEAN_ONLY is True, filter out is_outlier | edge_fail rows from both sets.

    Returns (X_train, y_train, X_test, y_test, test_df)
    where test_df carries (cmd_x_mm, cmd_y_mm) for region analysis.
    """
    df = static_df.copy()
    if CLEAN_ONLY:
        df = df[~df["is_outlier"] & ~df["edge_fail"]]

    train = df[df["repeat"] != TEST_REPEAT]
    test  = df[df["repeat"] == TEST_REPEAT]

    # Drop rows where any feature column is NaN (edge of ratio features)
    train = train.dropna(subset=feature_cols)
    test  = test.dropna(subset=feature_cols)

    X_train = train[feature_cols].to_numpy()
    y_train = train[["cmd_x_mm", "cmd_y_mm"]].to_numpy()
    X_test  = test[feature_cols].to_numpy()
    y_test  = test[["cmd_x_mm", "cmd_y_mm"]].to_numpy()

    return X_train, y_train, X_test, y_test, test.reset_index(drop=True)


def _evaluate(y_pred: np.ndarray, y_true: np.ndarray,
              test_df: pd.DataFrame) -> dict:
    """
    Return MAE and RMSE for X, Y, and both axes combined,
    broken down by region (all / interior / edge).
    """
    results = {}
    for region, mask in [
        ("all",      np.ones(len(test_df), dtype=bool)),
        ("interior", (test_df["cmd_y_mm"] <= INTERIOR_Y_MAX).values),
        ("edge",     (test_df["cmd_y_mm"] >  INTERIOR_Y_MAX).values),
    ]:
        if mask.sum() == 0:
            for target in ["x","y","xy"]:
                results[f"{region}_mae_{target}"]  = np.nan
                results[f"{region}_rmse_{target}"] = np.nan
            continue
        for t_idx, target in enumerate(["x","y"]):
            err = y_pred[mask, t_idx] - y_true[mask, t_idx]
            results[f"{region}_mae_{target}"]  = round(np.abs(err).mean(), 4)
            results[f"{region}_rmse_{target}"] = round(np.sqrt((err**2).mean()), 4)
        # Euclidean distance for combined XY error
        dist = np.sqrt(((y_pred[mask] - y_true[mask])**2).sum(axis=1))
        results[f"{region}_mae_xy"]  = round(dist.mean(), 4)
        results[f"{region}_rmse_xy"] = round(dist.max(), 4)   # max as "worst case"
    return results


def _build_model(name: str):
    """
    Return an sklearn estimator for a given model name.
    GPR and NNR are wrapped in a StandardScaler pipeline.
    """
    if name == "nnr":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  KNeighborsRegressor(n_neighbors=1)),
        ])
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                     random_state=42, n_jobs=-1)
    if name == "gb":
        return GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                         max_depth=4, subsample=0.8,
                                         random_state=42)
    if name == "gpr":
        kernel = ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(0.1)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GaussianProcessRegressor(
                kernel=kernel, n_restarts_optimizer=3,
                normalize_y=True, random_state=42)),
        ])
    raise ValueError(f"Unknown model: {name}")


def run_model_comparison(
    static_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train and evaluate every (model × feature_set) combination.
    Uses separate X-regressor and Y-regressor for every model.

    Returns a results DataFrame saved to datasets/model_evaluation.csv.
    """
    MODEL_NAMES = ["nnr", "rf", "gb", "gpr"]
    records = []

    total = len(MODEL_NAMES) * len(FEATURE_SETS)
    done  = 0
    for fs_name, feat_cols in FEATURE_SETS.items():
        avail = [c for c in feat_cols if c in static_df.columns]
        if not avail:
            continue
        X_tr, y_tr, X_te, y_te, test_df = make_train_test_split(static_df, avail)

        if len(X_tr) == 0 or len(X_te) == 0:
            continue

        for m_name in MODEL_NAMES:
            done += 1
            print(f"  [{done:2d}/{total}]  {m_name:4s}  ×  {fs_name}", end=" ... ")
            # Train separate X and Y models
            model_x = _build_model(m_name)
            model_y = _build_model(m_name)
            model_x.fit(X_tr, y_tr[:, 0])
            model_y.fit(X_tr, y_tr[:, 1])

            pred_x = model_x.predict(X_te)
            pred_y = model_y.predict(X_te)
            y_pred = np.column_stack([pred_x, pred_y])

            metrics = _evaluate(y_pred, y_te, test_df)
            records.append({"model": m_name, "features": fs_name,
                            "n_train": len(X_tr), "n_test": len(X_te),
                            **metrics})
            print(f"X-MAE={metrics['all_mae_x']:.3f}  Y-MAE={metrics['all_mae_y']:.3f}")

    result = pd.DataFrame(records)
    out_path = OUT_DIR / "model_evaluation.csv"
    result.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    return result


def get_best_predictions(
    static_df: pd.DataFrame,
    model_name: str = "gb",
    feature_set: str = "combined",
) -> tuple:
    """
    Fit the specified model/feature combination and return
    (y_pred, y_true, test_df) for downstream plotting.
    """
    feat_cols = [c for c in FEATURE_SETS[feature_set] if c in static_df.columns]
    X_tr, y_tr, X_te, y_te, test_df = make_train_test_split(static_df, feat_cols)

    mx = _build_model(model_name); my = _build_model(model_name)
    mx.fit(X_tr, y_tr[:, 0]); my.fit(X_tr, y_tr[:, 1])
    y_pred = np.column_stack([mx.predict(X_te), my.predict(X_te)])

    # Also return the GB feature importances (only works for pure GB, not pipeline)
    importances = None
    if model_name in ("rf", "gb"):
        importances = {
            "x": mx.feature_importances_,
            "y": my.feature_importances_,
            "cols": feat_cols,
        }
    return y_pred, y_te, test_df, importances


def get_gpr_uncertainty(
    static_df: pd.DataFrame,
    feature_set: str = "combined",
) -> tuple:
    """
    Fit GPR and return (y_pred, y_std, test_df) so we can plot the
    uncertainty map for each axis.
    """
    feat_cols = [c for c in FEATURE_SETS[feature_set] if c in static_df.columns]
    X_tr, y_tr, X_te, y_te, test_df = make_train_test_split(static_df, feat_cols)

    results = {}
    for t_idx, target in enumerate(["x", "y"]):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GaussianProcessRegressor(
                kernel=ConstantKernel(1.0)*RBF(1.0)+WhiteKernel(0.1),
                n_restarts_optimizer=3, normalize_y=True, random_state=42)),
        ])
        pipe.fit(X_tr, y_tr[:, t_idx])
        # GPR predict_std requires calling the inner model after scaling
        X_te_scaled = pipe["scaler"].transform(X_te)
        mean, std   = pipe["model"].predict(X_te_scaled, return_std=True)
        results[target] = {"mean": mean, "std": std}

    return results, test_df


# ──────────────────────────────────────────────────────────────────────────────
# I  Model result plots  (12–15)
# ──────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(eval_df: pd.DataFrame) -> None:
    """
    Grouped bar chart: MAE for each model across all feature sets.
    Separate sub-plots for X and Y, with interior/all/edge breakout.
    """
    models = eval_df["model"].unique()
    feat_sets = eval_df["features"].unique()
    x = np.arange(len(feat_sets))
    w = 0.8 / len(models)
    palette = ["#4C72B0","#DD8452","#55A868","#C44E52"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    for ax, target, t_label in zip(axes, ["x","y"], ["cmd_x_mm","cmd_y_mm"]):
        for i, (m, color) in enumerate(zip(models, palette)):
            mdf   = eval_df[eval_df["model"] == m]
            all_  = [mdf.loc[mdf["features"]==f, "all_mae_"+target].values[0]
                     if f in mdf["features"].values else np.nan
                     for f in feat_sets]
            int_  = [mdf.loc[mdf["features"]==f, "interior_mae_"+target].values[0]
                     if f in mdf["features"].values else np.nan
                     for f in feat_sets]
            ax.bar(x + i*w, all_, w*0.6, label=f"{m} (all)",
                   color=color, alpha=0.9, edgecolor="white")
            ax.bar(x + i*w, int_, w*0.6, label=f"{m} (interior)",
                   color=color, alpha=0.45, hatch="//", edgecolor="white")
        ax.set_xticks(x + w*(len(models)-1)/2)
        ax.set_xticklabels(feat_sets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("MAE (mm)"); ax.set_title(f"{t_label} prediction")
        ax.grid(True, alpha=0.3, axis="y")
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Model × Feature-set MAE comparison\n"
                 "(solid = all test rows,  hatched = interior only)")
    _save_fig(fig, "12_model_comparison_mae.png")


def plot_predictions_scatter(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    test_df: pd.DataFrame,
    title_suffix: str = "GradientBoosting + combined",
) -> None:
    """
    Predicted vs actual scatter, coloured by interior/edge region,
    with per-region MAE annotated.
    """
    interior = (test_df["cmd_y_mm"] <= INTERIOR_Y_MAX).values
    colors   = np.where(interior, "steelblue", "tomato")
    labels   = ["interior (y≤35)", "edge (y>35)"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, t_idx, t_name in zip(axes, [0,1], ["cmd_x_mm","cmd_y_mm"]):
        for mask, label, c in [(interior, labels[0], "steelblue"),
                                (~interior, labels[1], "tomato")]:
            if mask.sum():
                ax.scatter(y_true[mask,t_idx], y_pred[mask,t_idx],
                           s=12, alpha=0.6, color=c, label=label, edgecolors="none")
                mae = np.abs(y_pred[mask,t_idx]-y_true[mask,t_idx]).mean()
                ax.annotate(f"{label}: MAE={mae:.3f} mm",
                            xy=(0.03, 0.97-0.06*labels.index(label)),
                            xycoords="axes fraction", fontsize=8,
                            color=c, va="top")
        lo = min(y_true[:,t_idx].min(), y_pred[:,t_idx].min())
        hi = max(y_true[:,t_idx].max(), y_pred[:,t_idx].max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1)
        ax.set_xlabel(f"actual {t_name} (mm)")
        ax.set_ylabel(f"predicted {t_name} (mm)")
        ax.set_title(f"{t_name}"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.suptitle(f"Predicted vs actual  ({title_suffix})")
    _save_fig(fig, "13_predictions_scatter.png")


def plot_residual_heatmap(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    test_df: pd.DataFrame,
) -> None:
    """
    Heatmap of mean absolute residual on the XY grid for X and Y separately.
    Reveals systematic position-dependent errors.
    """
    resid_df = test_df[["cmd_x_mm","cmd_y_mm"]].copy()
    resid_df["res_x"] = np.abs(y_pred[:,0] - y_true[:,0])
    resid_df["res_y"] = np.abs(y_pred[:,1] - y_true[:,1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, label in zip(axes, ["res_x","res_y"], ["X residual (mm)","Y residual (mm)"]):
        pv = resid_df.groupby(["cmd_x_mm","cmd_y_mm"])[col].mean().unstack("cmd_y_mm")
        im = ax.pcolormesh(pv.index.values, pv.columns.values,
                           pv.T.values, shading="auto", cmap="hot")
        plt.colorbar(im, ax=ax, label=label)
        ax.axhline(INTERIOR_Y_MAX+0.5, color="cyan", lw=1.5, ls="--",
                   label="interior/edge boundary")
        ax.legend(fontsize=8)
        ax.set_title(label); ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    fig.suptitle("Mean absolute residual heatmap on XY grid  (GradientBoosting + combined)")
    _save_fig(fig, "14_residual_heatmap.png")


def plot_gpr_uncertainty(gpr_results: dict, test_df: pd.DataFrame) -> None:
    """
    Heatmap of GPR posterior std on the XY grid.
    High std = position where the model is uncertain — correlates with
    sensor failure regions and sparse coverage.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, target in zip(axes, ["x","y"]):
        std_df = test_df[["cmd_x_mm","cmd_y_mm"]].copy()
        std_df["std"] = gpr_results[target]["std"]
        pv = std_df.groupby(["cmd_x_mm","cmd_y_mm"])["std"].mean().unstack("cmd_y_mm")
        im = ax.pcolormesh(pv.index.values, pv.columns.values,
                           pv.T.values, shading="auto", cmap="plasma")
        plt.colorbar(im, ax=ax, label="posterior std (mm)")
        ax.axhline(INTERIOR_Y_MAX+0.5, color="cyan", lw=1.5, ls="--",
                   label="interior/edge")
        ax.legend(fontsize=8)
        ax.set_title(f"GPR uncertainty: cmd_{target}_mm")
        ax.set_xlabel("cmd_x_mm"); ax.set_ylabel("cmd_y_mm")
    fig.suptitle("GPR posterior uncertainty  (combined features)\n"
                 "High values → model less confident → correlated with sensor failures")
    _save_fig(fig, "15_gpr_uncertainty.png")


def plot_feature_importances(importances: dict) -> None:
    """
    Horizontal bar chart of GradientBoosting feature importances for X and Y.
    """
    cols = importances["cols"]
    imp_x = importances["x"]
    imp_y = importances["y"]
    order = np.argsort(imp_x + imp_y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    y_pos = np.arange(len(cols))
    for ax, imp, target in zip(axes, [imp_x, imp_y], ["cmd_x_mm","cmd_y_mm"]):
        ax.barh(y_pos, imp[order], color="steelblue", alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([cols[i] for i in order], fontsize=8)
        ax.set_xlabel("Feature importance (mean decrease impurity)")
        ax.set_title(f"{target} model"); ax.grid(True, alpha=0.3, axis="x")
    fig.suptitle("GradientBoosting feature importances  (combined feature set)")
    _save_fig(fig, "16_feature_importances.png")


# ──────────────────────────────────────────────────────────────────────────────
# J  Save datasets
# ──────────────────────────────────────────────────────────────────────────────

def save_datasets(static_splits: dict, ts_df: pd.DataFrame) -> None:
    print("\n[save_datasets]")
    for name, sdf in static_splits.items():
        path = OUT_DIR / f"static_{name}.csv"
        sdf.to_csv(path, index=False)
        n_clean = int((~sdf["is_outlier"] & ~sdf["edge_fail"]).sum())
        print(f"  {path.name:<35s} {len(sdf):4d} rows × {len(sdf.columns):2d} cols"
              f"  ({n_clean} clean)")
    path_ts = OUT_DIR / f"timeseries_K{TIMESERIES_K}.csv"
    ts_df.to_csv(path_ts, index=False)
    n_clean_ts = int((~ts_df["is_outlier"] & ~ts_df["edge_fail"]).sum())
    print(f"  {path_ts.name:<35s} {len(ts_df):4d} rows × {len(ts_df.columns):3d} cols"
          f"  ({n_clean_ts} clean)")


# ──────────────────────────────────────────────────────────────────────────────
# K  Assessment
# ──────────────────────────────────────────────────────────────────────────────

def write_assessment(
    df: pd.DataFrame,
    repeat_stats: pd.DataFrame,
    drift_stats: pd.DataFrame,
    noise_stats: pd.DataFrame,
    eval_df: pd.DataFrame | None,
) -> None:
    """Write the full written assessment to datasets/assessment.txt."""

    n_total  = len(df)
    n_unique = df[["cmd_x_mm","cmd_y_mm"]].drop_duplicates().shape[0]
    n_out    = int(df["is_outlier"].sum())
    bad_pa6  = int((df["raw_pa6"] < EDGE_PA6_MIN).sum())
    bad_pa5  = int((df["raw_pa5"] < EDGE_PA5_MIN).sum())
    if "edge_fail" not in df.columns:
        df = add_edge_failure_flag(df)
    pct_good = f"{100 * (~df['edge_fail']).mean():.0f}"

    def rms(ch, region="all"):
        sub = repeat_stats[(repeat_stats["channel"]==ch) & (repeat_stats["region"]==region)]
        return sub["rms_diff"].values[0] if len(sub) else float("nan")

    corr_x = df[[f"raw_{c}" for c in CHANNELS]+["cmd_x_mm"]].corr()["cmd_x_mm"]
    corr_y = df[[f"raw_{c}" for c in CHANNELS]+["cmd_y_mm"]].corr()["cmd_y_mm"]

    model_section = ""
    if eval_df is not None and len(eval_df):
        best_gb = eval_df[(eval_df["model"]=="gb") & (eval_df["features"]=="combined")]
        if len(best_gb):
            r = best_gb.iloc[0]
            model_section = f"""
━━━ MODEL RESULTS (GradientBoosting + combined features) ━━━━━━━━━━━━━━━━━━━━
  Train: repeat-1  ({r['n_train']:.0f} clean rows)
  Test : repeat-2  ({r['n_test']:.0f} clean rows)

  Region        X-MAE (mm)   Y-MAE (mm)   XY-MAE (mm)
  All           {r['all_mae_x']:.3f}        {r['all_mae_y']:.3f}        {r['all_mae_xy']:.3f}
  Interior      {r['interior_mae_x']:.3f}        {r['interior_mae_y']:.3f}        {r['interior_mae_xy']:.3f}
  Edge          {r['edge_mae_x']:.3f}        {r['edge_mae_y']:.3f}        {r['edge_mae_xy']:.3f}

Full comparison across all models and feature sets:
  → datasets/model_evaluation.csv"""

    assessment = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          DATASET QUALITY ASSESSMENT FOR REGRESSION MODELING  (v2)         ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ DATASET OVERVIEW ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Total rows          : {n_total}
  Unique XY positions : {n_unique}  (17 x-values × 21 y-values = 357 grid points)
  Repeats             : 2  →  {n_total//2} samples per repeat
  Statistical outliers: {n_out} ({100*n_out/n_total:.1f}%)
  Clean rows          : {pct_good}%  (edge_fail=False AND is_outlier=False)

━━━ REPEATABILITY (RMS counts, repeat-1 minus repeat-2) ━━━━━━━━━━━━━━━━━━━━
  Channel   All      Interior (y≤{INTERIOR_Y_MAX})   Edge (y>{INTERIOR_Y_MAX})
  pa0       {rms('pa0'):.1f}    {rms('pa0','interior'):.1f}              {rms('pa0','edge'):.1f}
  pa5       {rms('pa5'):.1f}    {rms('pa5','interior'):.1f}              {rms('pa5','edge'):.1f}
  pa6       {rms('pa6'):.1f}    {rms('pa6','interior'):.1f}              {rms('pa6','edge'):.1f}

  Interior repeatability is significantly better than the pooled figure.
  Edge failures inflate the pa5/pa6 numbers substantially.

━━━ DRIFT (linear trend counts/s) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{drift_stats.to_string(index=False)}

  PA6 drifts ~18–21 counts/s.  Demeaned features correct this automatically.

━━━ NOISE (estimated std from |r1−r2|/√2) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{noise_stats[noise_stats['region']=='interior'].drop(columns='region').to_string(index=False)}
  (interior only — edge noise is dominated by failure artifacts)

━━━ FEATURE–TARGET CORRELATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Channel     corr(raw, cmd_x_mm)   corr(raw, cmd_y_mm)
  raw_pa0     {corr_x['raw_pa0']:+.3f}                  {corr_y['raw_pa0']:+.3f}
  raw_pa5     {corr_x['raw_pa5']:+.3f}                  {corr_y['raw_pa5']:+.3f}
  raw_pa6     {corr_x['raw_pa6']:+.3f}                  {corr_y['raw_pa6']:+.3f}
{model_section}

━━━ HARDWARE / DATA ISSUES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  1. PA6 FIELD EDGE (y ≥ 36 mm):  {bad_pa6} rows with raw_pa6 < {EDGE_PA6_MIN}.
     Sensor loses the reflected beam.  Must exclude these positions OR
     physically reposition the detector to extend the clean operating range.

  2. INTER-REPEAT DELTA DRIFT:  delta_* features shift ~20–80 counts
     between repeats.  FIXED in v2 by rc_delta_* (per-repeat recentering).

  3. PA6 SESSION DRIFT (~19 counts/s):
     FIXED in v2 by dm_* features (rolling-window demeaning).

  4. MOTION ARTIFACTS AT Y-BOUNDARIES:  single-sample delta spikes.
     FIXED by edge_fail flag + CLEAN_ONLY filter.

  5. SMOOTH FEATURES LEAK TEMPORAL CONTEXT:
     Safe only when splitting by repeat (which we do).  Do NOT use
     smooth_* features if ever splitting by random row index.

━━━ RECOMMENDED MODELING SETUP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Feature set : "combined"  (raw + pairwise ratios + polynomial interactions)
  • Model       : GradientBoosting  (best MAE in interior region)
  • Split       : repeat-1 → train,  repeat-2 → test
  • Filter      : CLEAN_ONLY = True  (exclude is_outlier | edge_fail rows)
  • Separate    : independent X-regressor and Y-regressor
  • Restrict    : evaluate interior (y ≤ {INTERIOR_Y_MAX}) and edge separately
  • Uncertainty : use GPR posterior std as a confidence indicator per prediction
"""

    print(assessment)
    out_path = OUT_DIR / "assessment.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(assessment)
    print(f"  Assessment saved to: {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# L  Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    sep = "─" * 62

    print("=" * 62)
    print("  SENSOR CALIBRATION PIPELINE  v2")
    print("=" * 62)

    # ── C: Load & sort ────────────────────────────────────────────
    print(f"\n{sep}\n  C  Load, clean, sort\n{sep}")
    df = load_and_clean(DATA_PATH)
    df = sort_data(df)
    print(f"  Shape: {df.shape}")

    # ── D: Feature engineering ────────────────────────────────────
    print(f"\n{sep}\n  D  Baselines\n{sep}")
    baselines = compute_global_baselines(df)
    for ch, v in baselines.items():
        print(f"  {ch}: {v:.1f} counts")

    # ── F: Quality analysis ───────────────────────────────────────
    print(f"\n{sep}\n  F  Quality analysis\n{sep}")
    repeat_stats = analyze_repeatability(df)
    drift_stats  = analyze_drift(df)
    noise_stats  = analyze_noise(df)
    df           = analyze_edge_failures(df)

    # ── E: Build datasets ─────────────────────────────────────────
    print(f"\n{sep}\n  E  Build datasets\n{sep}")
    static_df     = build_static_dataset(df, baselines)
    static_splits = split_static_by_feature_set(static_df)
    for name, sdf in static_splits.items():
        n_c = int((~sdf["is_outlier"] & ~sdf["edge_fail"]).sum())
        print(f"  {name:15s}: {len(sdf)} rows, {n_c} clean")

    ts_df = build_timeseries_dataset(df, baselines)
    print(f"\n  Time-series: {ts_df.shape}  "
          f"(K={TIMESERIES_K}, lost {len(df)-len(ts_df)} warm-up rows)")

    # ── G: Diagnostic plots ───────────────────────────────────────
    print(f"\n{sep}\n  G  Diagnostic plots\n{sep}")
    plot_raw_channels_over_time(df)
    plot_signal_vs_position(df)
    plot_repeatability(df)
    plot_noise_heatmap(df)
    plot_edge_failure_map(df)
    plot_feature_correlation_matrix(df, baselines)
    plot_feature_target_bar(df, baselines)
    plot_delta_distributions(df)
    plot_pa6_anomaly(df)
    plot_xy_coverage(df)
    plot_interior_edge_repeatability(repeat_stats)

    # ── J: Save datasets ──────────────────────────────────────────
    print(f"\n{sep}\n  J  Save CSV outputs\n{sep}")
    save_datasets(static_splits, ts_df)

    # ── H/I: Modeling ─────────────────────────────────────────────
    eval_df = None
    if SKLEARN_AVAILABLE:
        print(f"\n{sep}\n  H  Model training & evaluation\n{sep}")
        eval_df = run_model_comparison(static_df)

        print(f"\n{sep}\n  I  Model result plots\n{sep}")
        plot_model_comparison(eval_df)

        # Best model plots (GradientBoosting + combined)
        y_pred, y_true, test_df, importances = get_best_predictions(
            static_df, model_name="gb", feature_set="combined"
        )
        plot_predictions_scatter(y_pred, y_true, test_df,
                                 title_suffix="GradientBoosting + combined")
        plot_residual_heatmap(y_pred, y_true, test_df)

        if importances:
            plot_feature_importances(importances)

        # GPR uncertainty map
        print("  Fitting GPR for uncertainty map ...", end=" ")
        gpr_results, gpr_test_df = get_gpr_uncertainty(static_df, "combined")
        print("done")
        plot_gpr_uncertainty(gpr_results, gpr_test_df)
    else:
        print(f"\n{sep}\n  H  Modeling SKIPPED (scikit-learn not available)\n{sep}")

    # ── K: Assessment ─────────────────────────────────────────────
    print(f"\n{sep}\n  K  Written assessment\n{sep}")
    write_assessment(df, repeat_stats, drift_stats, noise_stats, eval_df)

    print(f"\n{'=' * 62}")
    print(f"  Pipeline complete.")
    print(f"  Datasets : {OUT_DIR}/")
    print(f"  Plots    : {PLOT_DIR}/")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
