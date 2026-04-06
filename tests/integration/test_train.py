"""
Integration tests - POST /fit/{series_id}
"""


class TestTrainHappyPath:
    def test_returns_200_with_correct_shape(self, client, make_train_payload):
        r = client.post("/fit/sensor_1", json=make_train_payload())
        assert r.status_code == 200
        body = r.json()
        assert "series_id" in body
        assert "version" in body
        assert "points_used" in body

    def test_returns_correct_series_id(self, client, make_train_payload):
        r = client.post("/fit/sensor_temperature", json=make_train_payload())
        assert r.json()["series_id"] == "sensor_temperature"

    def test_first_train_is_v1(self, client, make_train_payload):
        r = client.post("/fit/sensor_1", json=make_train_payload())
        assert r.json()["version"] == "v1"

    def test_points_used_matches_input(self, client, make_train_payload):
        r = client.post("/fit/sensor_1", json=make_train_payload(n=50))
        assert r.json()["points_used"] == 50


class TestTrainVersioning:
    def test_retrain_increments_to_v2(self, client, make_train_payload):
        client.post("/fit/sensor_1", json=make_train_payload())
        r = client.post("/fit/sensor_1", json=make_train_payload())
        assert r.json()["version"] == "v2"

    def test_multiple_retrains_increment_correctly(self, client, make_train_payload):
        for expected in ["v1", "v2", "v3", "v4", "v5"]:
            r = client.post("/fit/sensor_1", json=make_train_payload())
            assert r.json()["version"] == expected

    def test_different_series_version_independently(self, client, make_train_payload):
        r1 = client.post("/fit/sensor_A", json=make_train_payload())
        r2 = client.post("/fit/sensor_B", json=make_train_payload())
        assert r1.json()["version"] == "v1"
        assert r2.json()["version"] == "v1"

    def test_retrain_one_series_does_not_affect_other(self, client, make_train_payload):
        client.post("/fit/sensor_A", json=make_train_payload())
        client.post(
            "/fit/sensor_A", json=make_train_payload()
        )  # retrain sensor_A to v2
        r = client.post("/fit/sensor_B", json=make_train_payload())
        assert r.json()["version"] == "v1"  # sensor_B should still be at v1


class TestTrainPreflight:
    def test_too_few_points_returns_422(self, client):
        payload = {"timestamps": list(range(5)), "values": [1.0] * 5}
        r = client.post("/fit/sensor_1", json=payload)
        assert r.status_code == 422

    def test_422_includes_rule_in_detail(self, client):
        payload = {"timestamps": list(range(5)), "values": [1.0] * 5}
        r = client.post("/fit/sensor_1", json=payload)
        body = r.json()
        assert "detail" in body
        assert "rule" in body["detail"]
        assert body["detail"]["rule"] == "minimum_points"

    def test_constant_values_returns_422(self, client, make_train_payload):
        payload = {
            "timestamps": list(range(50)),
            "values": [5.0] * 50,
        }
        r = client.post("/fit/sensor_1", json=payload)
        assert r.status_code == 422
        assert r.json()["detail"]["rule"] == "non_constant"

    def test_mismatched_lengths_returns_422(self, client):
        payload = {"timestamps": [1, 2, 3], "values": [1.0, 2.0]}
        r = client.post("/fit/sensor_1", json=payload)
        assert r.status_code == 422

    def test_non_monotonic_timestamps_returns_422(self, client):
        payload = {
            "timestamps": list(range(30)) + [28, 29, 30],
            "values": [float(i) for i in range(33)],
        }
        r = client.post("/fit/sensor_1", json=payload)
        assert r.status_code == 422
        assert r.json()["detail"]["rule"] == "monotonic_timestamps"

    def test_nan_value_returns_422(self, client):
        import math

        payload = {
            "timestamps": list(range(50)),
            "values": [float(i) for i in range(49)] + [math.nan],
        }
        r = client.post("/fit/sensor_1", json=payload)
        assert r.status_code == 422
