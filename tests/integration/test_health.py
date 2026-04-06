"""
Integration tests - GET /healthcheck
"""


class TestHealthCheckShape:
    def test_returns_200(self, client):
        r = client.get("/healthcheck")
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.get("/healthcheck")
        body = r.json()
        assert "series_trained" in body
        assert "inference_latency_ms" in body
        assert "training_latency_ms" in body

    def test_latency_fields_have_avg_and_p95(self, client):
        r = client.get("/healthcheck")
        body = r.json()
        assert "avg" in body["inference_latency_ms"]
        assert "p95" in body["inference_latency_ms"]
        assert "avg" in body["training_latency_ms"]
        assert "p95" in body["training_latency_ms"]


class TestHealthCheckSeriesCount:
    def test_zero_series_before_training(self, client):
        r = client.get("/healthcheck")
        assert r.json()["series_trained"] == 0

    def test_series_count_updates_after_one_train(self, client, make_train_payload):
        client.post("/fit/s1", json=make_train_payload())
        r = client.get("/healthcheck")
        assert r.json()["series_trained"] == 1

    def test_series_count_updates_after_multiple_trains(
        self, client, make_train_payload
    ):
        client.post("/fit/s1", json=make_train_payload())
        client.post("/fit/s2", json=make_train_payload())
        client.post("/fit/s3", json=make_train_payload())
        r = client.get("/healthcheck")
        assert r.json()["series_trained"] == 3

    def test_retrain_does_not_increment_series_count(self, client, make_train_payload):
        """Retraining the same series_id creates a new version, not a new series."""
        client.post("/fit/s1", json=make_train_payload())
        client.post("/fit/s1", json=make_train_payload())  # retrain
        r = client.get("/healthcheck")
        assert r.json()["series_trained"] == 1


class TestHealthCheckLatency:
    def test_latency_null_before_any_requests(self, client):
        r = client.get("/healthcheck")
        body = r.json()
        assert body["training_latency_ms"]["avg"] is None
        assert body["training_latency_ms"]["p95"] is None
        assert body["inference_latency_ms"]["avg"] is None
        assert body["inference_latency_ms"]["p95"] is None

    def test_training_latency_populated_after_fit(self, client, make_train_payload):
        client.post("/fit/s1", json=make_train_payload())
        r = client.get("/healthcheck")
        body = r.json()
        assert body["training_latency_ms"]["avg"] is not None
        assert body["training_latency_ms"]["p95"] is not None

    def test_inference_latency_null_before_predict(self, client, make_train_payload):
        client.post("/fit/s1", json=make_train_payload())
        r = client.get("/healthcheck")
        # Trained but no predict yet
        assert r.json()["inference_latency_ms"]["avg"] is None

    def test_inference_latency_populated_after_predict(
        self, client, make_train_payload
    ):
        client.post("/fit/s1", json=make_train_payload())
        client.post("/predict/s1", json={"timestamp": "0", "value": 1.0})
        r = client.get("/healthcheck")
        body = r.json()
        assert body["inference_latency_ms"]["avg"] is not None
        assert body["inference_latency_ms"]["p95"] is not None

    def test_latency_values_are_positive(self, client, make_train_payload):
        client.post("/fit/s1", json=make_train_payload())
        r = client.get("/healthcheck")
        avg = r.json()["training_latency_ms"]["avg"]
        assert avg > 0

    def test_p95_greater_or_equal_avg(self, client, make_train_payload):
        """p95 must always be >= avg by definition."""
        for i in range(10):
            client.post("/fit/s1", json=make_train_payload())
        r = client.get("/healthcheck")
        body = r.json()["training_latency_ms"]
        assert body["p95"] >= body["avg"]
