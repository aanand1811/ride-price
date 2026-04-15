"""
Dynamic Pricing ML Model - Training Pipeline
Dataset: Uber & Lyft Dataset (Boston, MA) from Kaggle
https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma

If you don't have the dataset yet, run with --synthetic to generate synthetic data for testing.
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed — using sklearn GradientBoostingRegressor as drop-in.")

# Optional ClearML tracking - gracefully skipped if not configured
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False


# ── Feature Engineering ────────────────────────────────────────────────────────

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw ride-hailing data.
    Works with the Uber/Lyft Kaggle dataset columns:
      datetime, source, destination, cab_type, name,
      price, distance, surge_multiplier, hour, day,
      month, timezone, windSpeed, humidity, visibility,
      apparentTemperature, temperature, precipIntensity, ...
    """
    df = df.copy()

    # ── Time features ──
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], unit="s", errors="coerce")
        df["hour"]        = df["datetime"].dt.hour
        df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Mon … 6=Sun
        df["month"]       = df["datetime"].dt.month
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    elif "hour" in df.columns:
        df["day_of_week"] = df.get("day", pd.Series([2] * len(df)))
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

    # ── Rush hour flag (7-9 AM and 5-7 PM) ──
    df["is_rush_hour"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0
    )

    # ── Night surcharge flag ──
    df["is_night"] = df["hour"].apply(lambda h: 1 if h >= 22 or h <= 5 else 0)

    # ── Cab type encoding ──
    if "cab_type" in df.columns:
        df["is_uber"] = (df["cab_type"].str.lower() == "uber").astype(int)
    else:
        df["is_uber"] = 0

    # ── Service tier encoding ──
    if "name" in df.columns:
        tier_map = {
            "UberX": 1, "UberPool": 0, "UberXL": 2,
            "Black": 3, "Black SUV": 4, "WAV": 2,
            "Lyft": 1, "Shared": 0, "Lyft XL": 2,
            "Lux": 3, "Lux Black": 4, "Lux Black XL": 5
        }
        df["service_tier"] = df["name"].map(tier_map).fillna(1)
    else:
        df["service_tier"] = 1

    # ── Weather features (fill missing with neutral values) ──
    weather_defaults = {
        "temperature": 60,
        "apparentTemperature": 60,
        "humidity": 0.5,
        "windSpeed": 5,
        "visibility": 10,
        "precipIntensity": 0,
        "cloudCover": 0.3,
    }
    for col, default in weather_defaults.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # ── Bad weather flag ──
    df["bad_weather"] = (
        (df["precipIntensity"] > 0.1) |
        (df["windSpeed"] > 20) |
        (df["visibility"] < 5)
    ).astype(int)

    # ── Distance (fill if missing) ──
    if "distance" not in df.columns:
        df["distance"] = 2.5
    else:
        df["distance"] = df["distance"].fillna(df["distance"].median())

    return df


FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_rush_hour", "is_night",
    "is_uber", "service_tier", "distance",
    "temperature", "apparentTemperature", "humidity",
    "windSpeed", "visibility", "precipIntensity", "cloudCover",
    "bad_weather"
]


# ── Synthetic Dataset (fallback for testing) ────────────────────────────────

def generate_synthetic_data(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic ride-hailing dataset that mirrors the
    feature distributions of the Uber/Lyft Boston dataset.
    Used for development/testing when the real dataset is not available.
    """
    rng = np.random.default_rng(seed)

    hour        = rng.integers(0, 24, n)
    day_of_week = rng.integers(0, 7, n)
    month       = rng.integers(1, 13, n)
    is_weekend  = (day_of_week >= 5).astype(int)
    is_rush_hour = np.where((hour >= 7) & (hour <= 9) | (hour >= 17) & (hour <= 19), 1, 0)
    is_night    = np.where((hour >= 22) | (hour <= 5), 1, 0)
    is_uber     = rng.integers(0, 2, n)
    service_tier = rng.choice([0, 1, 2, 3, 4], n, p=[0.1, 0.45, 0.25, 0.15, 0.05])
    distance    = rng.exponential(2.5, n).clip(0.3, 20)
    temperature         = rng.normal(55, 15, n).clip(10, 95)
    apparent_temperature = temperature + rng.normal(0, 3, n)
    humidity    = rng.beta(5, 5, n)
    wind_speed  = rng.exponential(8, n).clip(0, 50)
    visibility  = rng.beta(8, 2, n) * 10
    precip      = rng.exponential(0.05, n).clip(0, 1)
    cloud_cover = rng.beta(2, 2, n)
    bad_weather = ((precip > 0.1) | (wind_speed > 20) | (visibility < 5)).astype(int)

    # Realistic price formula
    base_price = (
        2.50
        + service_tier * 4.0
        + distance * 1.80
        + is_rush_hour * 3.5
        + is_night * 2.0
        + bad_weather * 2.5
        + is_weekend * 1.5
        + rng.normal(0, 2, n)          # noise
    ).clip(3, 100)

    df = pd.DataFrame({
        "hour": hour, "day_of_week": day_of_week, "month": month,
        "is_weekend": is_weekend, "is_rush_hour": is_rush_hour,
        "is_night": is_night, "is_uber": is_uber,
        "service_tier": service_tier, "distance": distance,
        "temperature": temperature, "apparentTemperature": apparent_temperature,
        "humidity": humidity, "windSpeed": wind_speed,
        "visibility": visibility, "precipIntensity": precip,
        "cloudCover": cloud_cover, "bad_weather": bad_weather,
        "price": base_price
    })
    return df


# ── Model Training ─────────────────────────────────────────────────────────────

def train(data_path: str = None, synthetic: bool = False, output_dir: str = "models"):
    os.makedirs(output_dir, exist_ok=True)

    # ── ClearML task ──
    task = None
    if CLEARML_AVAILABLE:
        try:
            task = Task.init(
                project_name="RidePricingML",
                task_name="XGBoost Dynamic Pricing"
            )
        except Exception:
            pass

    # ── Load data ──
    if synthetic or data_path is None:
        print("[INFO] Using synthetic dataset …")
        df = generate_synthetic_data()
    else:
        print(f"[INFO] Loading dataset from {data_path} …")
        df = pd.read_csv(data_path)
        df = extract_features(df)
        df = df.dropna(subset=["price"])

    X = df[FEATURE_COLS]
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Scale features ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    results = {}

    # ── Baseline: Linear Regression ──
    print("[INFO] Training Linear Regression baseline …")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    results["LinearRegression"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, lr_preds))),
        "mae":  float(mean_absolute_error(y_test, lr_preds)),
        "r2":   float(r2_score(y_test, lr_preds))
    }
    print(f"  LR  → RMSE={results['LinearRegression']['rmse']:.3f}  "
          f"MAE={results['LinearRegression']['mae']:.3f}  "
          f"R²={results['LinearRegression']['r2']:.3f}")

    # ── XGBoost (or sklearn GradientBoosting fallback) ──
    xgb_params = {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "random_state": 42,
    }
    if XGBOOST_AVAILABLE:
        print("[INFO] Training XGBoost …")
        xgb_params.update({"colsample_bytree": 0.8, "min_child_weight": 3,
                            "reg_alpha": 0.1, "reg_lambda": 1.0, "n_jobs": -1})
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        print("[INFO] Training GradientBoostingRegressor (xgboost fallback) …")
        sklearn_params = {k: v for k, v in xgb_params.items()
                          if k in ["n_estimators", "max_depth", "learning_rate",
                                   "subsample", "random_state"]}
        xgb_model = GradientBoostingRegressor(**sklearn_params)
        xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    results["XGBoost"] = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, xgb_preds))),
        "mae":  float(mean_absolute_error(y_test, xgb_preds)),
        "r2":   float(r2_score(y_test, xgb_preds))
    }
    print(f"  XGB → RMSE={results['XGBoost']['rmse']:.3f}  "
          f"MAE={results['XGBoost']['mae']:.3f}  "
          f"R²={results['XGBoost']['r2']:.3f}")

    # ── Log to ClearML ──
    if task:
        logger = task.get_logger()
        for model_name, metrics in results.items():
            for metric, val in metrics.items():
                logger.report_scalar(title=metric.upper(), series=model_name, value=val, iteration=0)
        task.connect(xgb_params)

    # ── Save artifacts ──
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "feature_cols.pkl"), "wb") as f:
        pickle.dump(FEATURE_COLS, f)

    if XGBOOST_AVAILABLE:
        xgb_model.save_model(os.path.join(output_dir, "xgb_model.json"))
    else:
        with open(os.path.join(output_dir, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
    print(f"[INFO] Model artifacts saved to ./{output_dir}/")

    if task:
        task.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train dynamic pricing model")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV dataset")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--output", type=str, default="models", help="Output directory")
    args = parser.parse_args()

    train(data_path=args.data, synthetic=args.synthetic, output_dir=args.output)
