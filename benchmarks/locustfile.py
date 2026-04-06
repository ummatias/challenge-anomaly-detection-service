"""
Locust benchmark

Simulate 100 users
"""

import random
import string
from locust import HttpUser, task, between, events


def random_series_id() -> str:
    return "bench_" + "".join(random.choices(string.ascii_lowercase, k=6))


class AnomalyUser(HttpUser):
    """
    Each user owns one series, traines it on first request,
    then continuously predicts. 95% predict, 5% anomaly injection.
    """

    wait_time = between(0.05, 0.2)

    def on_start(self):
        self.series_id = random_series_id()
        payload = {
            "timestamps": list(range(100)),
            "values": [float(i) + random.gauss(0, 1) for i in range(100)],
        }
        with self.client.post(
            f"/fit/{self.series_id}",
            json=payload,
            catch_response=True,
            name="/fit/[series_id] (warmup)",
        ) as r:
            if r.status_code != 200:
                r.failure(f"Train failed: {r.status_code} {r.text}")

    @task(19)
    def predict_normal(self):
        value = random.gauss(50, 5)
        with self.client.post(
            f"/predict/{self.series_id}",
            json={"timestamp": "1000", "value": value},
            catch_response=True,
            name="/predict/[series_id]",
        ) as r:
            if r.status_code != 200:
                r.failure(f"Predict failed: {r.status_code}")

    @task(1)
    def predict_anomaly(self):
        value = random.uniform(10_000, 100_000)
        with self.client.post(
            f"/predict/{self.series_id}",
            json={"timestamp": "1001", "value": value},
            catch_response=True,
            name="/predict/[series_id] (anomaly)",
        ) as r:
            if r.status_code != 200:
                r.failure(f"Anomaly predict failed: {r.status_code}")
            elif not r.json().get("anomaly"):
                r.failure("Expected anomaly=True for extreme value")

    @task(1)
    def healthcheck(self):
        self.client.get("/healthcheck", name="/healthcheck")


@events.quitting.add_listener
def on_quit(environment, **kwargs):
    stats = environment.runner.stats.total
    print(f"\n{'='*50}")
    print(f"Benchmark Summary")
    print(f"  Requests : {stats.num_requests}")
    print(f"  Failures : {stats.num_failures}")
    print(f"  p50 (ms) : {stats.median_response_time}")
    print(f"  p95 (ms) : {stats.get_response_time_percentile(0.95)}")
    print(f"  p99 (ms) : {stats.get_response_time_percentile(0.99)}")
    print(f"  req/s    : {stats.current_rps:.1f}")
    print(f"{'='*50}\n")
