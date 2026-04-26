# Containerized Dynamic Pricing ML System

A containerized machine learning system for real-time ride-hailing price prediction, featuring autoscaling via Kubernetes and load testing with Locust.

---

## Problem Statement

Most machine learning models are developed in isolation and are not directly usable in real-world systems. This project focuses on building a complete, deployable machine learning system for real-time ride pricing, addressing challenges such as scalability, latency, and reliability.

---

## Project Structure

```
ride-pricing/
├── train.py               # Data preprocessing + model training (XGBoost)
├── api/
│   └── main.py            # FastAPI inference service
├── k8s/
│   └── deployment.yaml    # Kubernetes Deployment + Service + HPA
├── tests/
│   └── locustfile.py      # Locust load testing scenarios
├── models/                # Generated: scaler.pkl, xgb_model.json
├── data/                  # Place your dataset CSV here
├── Dockerfile             # Multi-stage Docker build
├── docker-compose.yml     # Local development stack
└── requirements.txt
```

---

## Quickstart (Local — no Docker)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model

**With synthetic data (no dataset needed):**
```bash
python train.py --synthetic
```

**With the real Uber/Lyft dataset:**
```bash
# Download from: https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma
# Place rideshare_kaggle.csv in data/
python train.py --data data/rideshare_kaggle.csv
```

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Test it
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hour": 8, "day_of_week": 0, "month": 3,
    "distance": 3.2, "is_uber": 1, "service_tier": 1,
    "temperature": 55.0, "humidity": 0.6, "windSpeed": 10.0,
    "visibility": 8.0, "precipIntensity": 0.05
  }'
```

Interactive docs: http://localhost:8000/docs

---

## Docker Deployment

### Build and run
```bash
docker compose up --build
```

### Run with load testing
```bash
docker compose --profile load up
# Open Locust UI: http://localhost:8089
```

---

## Kubernetes Deployment

### Prerequisites
- A running Kubernetes cluster (Minikube, EKS, GKE, etc.)
- kubectl configured
- Metrics Server installed (for HPA)

### Install Metrics Server
```bash
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```

### Push image to registry
```bash
docker tag ride-pricing:latest <your-registry>/ride-pricing:latest
docker push <your-registry>/ride-pricing:latest
# Update image: in k8s/deployment.yaml to match
```

### Deploy
```bash
kubectl apply -f k8s/
```

### Monitor autoscaling
```bash
kubectl get hpa -n ride-pricing --watch
kubectl get pods -n ride-pricing --watch
```

> **Note:** This system is designed to be cloud-deployable (e.g., AWS EC2/EKS).  
> The current implementation has been validated on a local Kubernetes cluster.

---

## Load Testing

### Headless (CI/CD mode)
```bash
locust -f tests/locustfile.py \
       --host http://localhost:8000 \
       --headless -u 100 -r 10 --run-time 60s \
       --csv results/load_test
```

### Web UI mode
```bash
locust -f tests/locustfile.py --host http://localhost:8000
# Open: http://localhost:8089
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/health` | Liveness/readiness check |
| POST | `/predict` | Single ride price prediction |
| POST | `/predict/batch` | Batch prediction (up to 100 rides) |
| GET | `/model/info` | Model metadata and feature list |

---

## Dataset

This project is designed for the **Uber & Lyft Dataset (Boston, MA)** from Kaggle:
https://www.kaggle.com/datasets/brllrb/uber-and-lyft-dataset-boston-ma

Key columns used: `time_stamp`, `distance`, `cab_type`, `name`, `price`,
`temperature`, `humidity`, `windSpeed`, `visibility`, `precipIntensity`, `cloudCover`

A synthetic data generator is included for development without the dataset.

---

## ClearML Experiment Tracking 

```bash
pip install clearml
clearml-init   # Enter your ClearML credentials
python train.py --synthetic  # Metrics automatically logged
```

---

## Evaluation Metrics

### Model
| Metric | Description |
|--------|-------------|
| RMSE | Root Mean Squared Error on held-out test set |
| MAE | Mean Absolute Error |
| R² | Coefficient of determination |

### System (from Locust)
| Metric | Description |
|--------|-------------|
| Latency (p50/p95/p99) | Response time percentiles |
| Throughput (RPS) | Requests per second handled |
| Failure Rate | % of failed requests under load |
| Autoscaling | Pod count vs load curve |

---

## Results

### Model Performance (Kaggle Dataset)
- Linear Regression → RMSE ≈ 4.03, MAE ≈ 2.89, R² ≈ 0.81  
- XGBoost → RMSE ≈ 2.88, MAE ≈ 1.73, R² ≈ 0.90  

XGBoost significantly outperforms the linear baseline, indicating nonlinear relationships in ride pricing.

### System Performance
- Latency (p50): < 5 ms  
- Throughput: 200+ requests/sec  
- Failure rate: < 1%  
- Autoscaling: dynamic pod scaling under load
