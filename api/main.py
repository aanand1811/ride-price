"""
Dynamic Pricing Inference API
Run locally:  uvicorn api.main:app --reload --port 8000
Docs:         http://localhost:8000/docs
"""

import os
import pickle
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  Globals loaded at startup 
MODEL_DIR = os.getenv("MODEL_DIR", "models")
xgb_model = None
scaler = None
feature_cols = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global xgb_model, scaler, feature_cols
    logger.info("Loading model artifacts …")
    try:
        import os, pickle
        json_path = os.path.join(MODEL_DIR, "xgb_model.json")
        pkl_path  = os.path.join(MODEL_DIR, "xgb_model.pkl")

        if os.path.exists(json_path):
            import xgboost as xgb
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(json_path)
        elif os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                xgb_model = pickle.load(f)
        else:
            raise FileNotFoundError("No model file found (xgb_model.json or xgb_model.pkl)")

        with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title="Ride Pricing ML API",
    description="Real-time dynamic pricing predictions for ride-hailing services",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#  Schemas 

class RideRequest(BaseModel):
    # Time features
    hour:         int   = Field(..., ge=0, le=23,  description="Hour of the day (0-23)")
    day_of_week:  int   = Field(..., ge=0, le=6,   description="Day of week (0=Mon, 6=Sun)")
    month:        int   = Field(..., ge=1, le=12,  description="Month (1-12)")

    # Ride features
    distance:     float = Field(..., gt=0, le=100, description="Trip distance in miles")
    is_uber:      int   = Field(0,  ge=0, le=1,   description="1=Uber, 0=Lyft")
    service_tier: int   = Field(1,  ge=0, le=5,   description="0=Pool/Shared … 5=Lux Black XL")

    # Weather features 
    temperature:          Optional[float] = Field(60.0,  description="Temperature (°F)")
    apparentTemperature:  Optional[float] = Field(60.0,  description="Feels-like temperature (°F)")
    humidity:             Optional[float] = Field(0.5,   ge=0, le=1)
    windSpeed:            Optional[float] = Field(5.0,   ge=0)
    visibility:           Optional[float] = Field(10.0,  ge=0)
    precipIntensity:      Optional[float] = Field(0.0,   ge=0)
    cloudCover:           Optional[float] = Field(0.3,   ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "hour": 8, "day_of_week": 0, "month": 3,
                "distance": 3.2, "is_uber": 1, "service_tier": 1,
                "temperature": 55.0, "apparentTemperature": 50.0,
                "humidity": 0.6, "windSpeed": 10.0,
                "visibility": 8.0, "precipIntensity": 0.05, "cloudCover": 0.4
            }
        }


class PriceResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_price: float
    surge_multiplier: float
    price_tier: str
    latency_ms: float
    model_version: str = "xgboost-v1"


#  Helper 

def build_feature_vector(req: RideRequest) -> np.ndarray:
    is_weekend   = int(req.day_of_week >= 5)
    is_rush_hour = int((7 <= req.hour <= 9) or (17 <= req.hour <= 19))
    is_night     = int(req.hour >= 22 or req.hour <= 5)
    bad_weather  = int(
        (req.precipIntensity or 0) > 0.1 or
        (req.windSpeed or 0) > 20 or
        (req.visibility or 10) < 5
    )

    feature_map = {
        "hour":                 req.hour,
        "day_of_week":          req.day_of_week,
        "month":                req.month,
        "is_weekend":           is_weekend,
        "is_rush_hour":         is_rush_hour,
        "is_night":             is_night,
        "is_uber":              req.is_uber,
        "service_tier":         req.service_tier,
        "distance":             req.distance,
        "temperature":          req.temperature,
        "apparentTemperature":  req.apparentTemperature,
        "humidity":             req.humidity,
        "windSpeed":            req.windSpeed,
        "visibility":           req.visibility,
        "precipIntensity":      req.precipIntensity,
        "cloudCover":           req.cloudCover,
        "bad_weather":          bad_weather,
    }
    return np.array([[feature_map[col] for col in feature_cols]])


def price_tier(price: float) -> str:
    if price < 10:  return "budget"
    if price < 20:  return "standard"
    if price < 35:  return "premium"
    return "luxury"


def estimate_surge(price: float, distance: float) -> float:
    base = 2.50 + distance * 1.80
    multiplier = round(price / base, 2) if base > 0 else 1.0
    return max(1.0, min(multiplier, 5.0))


# Endpoints 

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Ride Pricing ML API", "version": "1.0.0"}


@app.get("/health", tags=["Health"])
def health():
    model_loaded = xgb_model is not None
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded
    }


@app.post("/predict", response_model=PriceResponse, tags=["Prediction"])
def predict(ride: RideRequest):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    features = build_feature_vector(ride)
    price = float(xgb_model.predict(features)[0])
    latency = (time.perf_counter() - t0) * 1000

    return PriceResponse(
        predicted_price  = round(price, 2),
        surge_multiplier = estimate_surge(price, ride.distance),
        price_tier       = price_tier(price),
        latency_ms       = round(latency, 3)
    )


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(rides: list[RideRequest]):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(rides) > 100:
        raise HTTPException(status_code=400, detail="Batch size limit is 100")

    t0 = time.perf_counter()
    features = np.vstack([build_feature_vector(r) for r in rides])
    prices   = xgb_model.predict(features).tolist()
    latency  = (time.perf_counter() - t0) * 1000

    return {
        "predictions": [
            {
                "predicted_price":  round(p, 2),
                "surge_multiplier": estimate_surge(p, rides[i].distance),
                "price_tier":       price_tier(p)
            }
            for i, p in enumerate(prices)
        ],
        "count":      len(prices),
        "latency_ms": round(latency, 3)
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    return {
        "model_type":    "XGBoost Regressor",
        "features":      feature_cols,
        "feature_count": len(feature_cols) if feature_cols else 0,
        "task":          "ride price regression",
        "version":       "xgboost-v1"
    }
