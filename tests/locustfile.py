"""
Locust Load Testing — Ride Pricing ML API
==========================================
Run:
  locust -f tests/locustfile.py --host http://localhost:8000

Headless (CI/CD):
  locust -f tests/locustfile.py --host http://localhost:8000 \
         --headless -u 100 -r 10 --run-time 60s \
         --csv results/load_test

Scenarios
---------
  RidePricingUser  — mixed single + batch predict (90% of traffic)
  HealthCheckUser  — health/info endpoints         (10% of traffic)
"""

import random
from locust import HttpUser, TaskSet, task, between, tag


# ── Payload helpers ────────────────────────────────────────────────────────────

def random_ride():
    return {
        "hour":        random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "month":       random.randint(1, 12),
        "distance":    round(random.uniform(0.5, 15.0), 2),
        "is_uber":     random.randint(0, 1),
        "service_tier": random.choice([0, 1, 1, 1, 2, 3]),  # weighted toward tier 1
        "temperature":          round(random.uniform(20, 90), 1),
        "apparentTemperature":  round(random.uniform(20, 90), 1),
        "humidity":             round(random.uniform(0.2, 0.9), 2),
        "windSpeed":            round(random.uniform(0, 30), 1),
        "visibility":           round(random.uniform(1, 10), 1),
        "precipIntensity":      round(random.uniform(0, 0.3), 3),
        "cloudCover":           round(random.uniform(0, 1), 2),
    }


# ── Task Sets ──────────────────────────────────────────────────────────────────

class PredictTasks(TaskSet):

    @tag("predict", "single")
    @task(7)
    def predict_single(self):
        """Single ride price prediction — highest frequency."""
        with self.client.post(
            "/predict",
            json=random_ride(),
            name="/predict (single)",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "predicted_price" not in data:
                    resp.failure("Missing predicted_price in response")
                elif data["predicted_price"] <= 0:
                    resp.failure(f"Invalid price: {data['predicted_price']}")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("predict", "batch")
    @task(2)
    def predict_batch(self):
        """Batch prediction with 5-20 rides — simulates aggregator requests."""
        batch_size = random.randint(5, 20)
        payload    = [random_ride() for _ in range(batch_size)]

        with self.client.post(
            "/predict/batch",
            json=payload,
            name="/predict/batch",
            catch_response=True
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if data.get("count") != batch_size:
                    resp.failure(f"Expected {batch_size} predictions, got {data.get('count')}")
                else:
                    resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}")

    @tag("info")
    @task(1)
    def model_info(self):
        """Model info endpoint — low frequency."""
        self.client.get("/model/info", name="/model/info")


class HealthTasks(TaskSet):

    @tag("health")
    @task(1)
    def health_check(self):
        with self.client.get("/health", name="/health", catch_response=True) as resp:
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                resp.success()
            else:
                resp.failure("Health check failed")

    @tag("health")
    @task(1)
    def root(self):
        self.client.get("/", name="/")


# ── User Classes ───────────────────────────────────────────────────────────────

class RidePricingUser(HttpUser):
    """
    Primary user — simulates a ride-hailing app backend hitting the API.
    Wait between 0.5 and 2 seconds between requests (realistic API call rate).
    """
    tasks      = [PredictTasks]
    weight     = 9                        # 90% of simulated users
    wait_time  = between(0.5, 2.0)

    def on_start(self):
        # Warm-up: verify the service is alive before hammering it
        self.client.get("/health")


class HealthCheckUser(HttpUser):
    """
    Monitoring agent — simulates a Kubernetes liveness/readiness probe.
    """
    tasks     = [HealthTasks]
    weight    = 1                         # 10% of simulated users
    wait_time = between(5, 10)
