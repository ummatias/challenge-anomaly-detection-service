"""
Integration tests - POST /predict/{series_id}
"""


class TestPredictHappyPath:
    def test_returns_200_with_correct_shape(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload(n=100))
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 25.0})
        assert r.status_code == 200
        body = r.json()
        assert "anomaly" in body
        assert "model_version" in body

    def test_anomaly_is_boolean(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload(n=100))
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 25.0})
        assert isinstance(r.json()["anomaly"], bool)

    def test_model_version_is_string(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 1.0})
        assert isinstance(r.json()["model_version"], str)


class TestPredictAnomalyDetection:
    def test_normal_value_not_anomaly(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload(n=100))
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 25.0})
        assert r.json()["anomaly"] is False

    def test_extreme_high_value_is_anomaly(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload(n=100))
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 1_000_000.0})
        assert r.json()["anomaly"] is True

    def test_one_sided_limitation_extreme_low_not_flagged(self, client, make_train_payload):
        """
        Documents the known model limitation: threshold is upper-only.

        A value far below the mean is NOT be marked as anomaly
        This is a property of the spec's model, not a bug in the API.
        """
        client.post("/fit/sensor_1", json=make_train_payload(n=100))
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": -1_000_000.0})
        assert r.json()["anomaly"] is False  # known limitation, documented intentionally


class TestPredictVersioning:
    def test_defaults_to_latest_version(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 1.0})
        assert r.json()["model_version"] == "v1"

    def test_defaults_to_latest_after_retrain(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        client.post("/fit/sensor_1", json=make_train_payload())
        r = client.post("/predict/sensor_1", json={"timestamp": "0", "value": 1.0})
        assert r.json()["model_version"] == "v2"

    def test_predict_specific_version(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        client.post("/fit/sensor_1", json=make_train_payload(multiplier=100.0))
        r = client.post("/predict/sensor_1?version=v1", json={"timestamp": "0", "value": 1.0})
        assert r.json()["model_version"] == "v1"

    def test_specific_version_uses_that_models_parameters(self, client, make_train_payload):
        """
        v1 trained on small values (mean~25, threshold low)
        v2 trained on large values (mean~2500, threshold high)
        Same input value should produce different results per version.
        """
        client.post("/fit/sensor_1", json=make_train_payload(n=50))
        client.post("/fit/sensor_1", json=make_train_payload(n=50, multiplier=100.0))

        r_v1 = client.post("/predict/sensor_1?version=v1", json={"timestamp": "0", "value": 200.0})
        r_v2 = client.post("/predict/sensor_1?version=v2", json={"timestamp": "0", "value": 200.0})

        assert r_v1.json()["anomaly"] is True
        assert r_v2.json()["anomaly"] is False


class TestPredictErrors:
    def test_untrained_series_returns_404(self, client):
        r = client.post("/predict/unknown_series", json={"timestamp": "0", "value": 1.0})
        assert r.status_code == 404

    def test_invalid_version_returns_404(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        r = client.post("/predict/sensor_1?version=v99", json={"timestamp": "0", "value": 1.0})
        assert r.status_code == 404

    def test_404_detail_is_informative(self, client):
        r = client.post("/predict/ghost_series", json={"timestamp": "0", "value": 1.0})
        assert "ghost_series" in r.json()["detail"]