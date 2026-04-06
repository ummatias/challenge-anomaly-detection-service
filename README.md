# TSAD - Time Series Anomaly Detection Service

A  service for training and serving anomaly detection models on univariate time series. Each series is identified by a `series_id`, supports versioned retraining, and persists model state to disk.

## How it works

The model fits a Gaussian distribution over the training values and flags any point more than 3 standard deviations above the mean as an anomaly

Each `POST /fit/{series_id}` call creates a new immutable version (`v1`, `v2`, ...) stored under `storage/<series_id>/`. 
> Predictions default to the latest version but any past version can be queried explicitly.

---

## Requirements

- Docker + Docker Compose (Recommended)
- Python 3.11+, only if running locally without Docker

---

## Setup

### With Docker (recommended)

```bash
cp .env.example .env
docker compose up --build
```

This starts three services:

| Service    | URL                   | Purpose                       |
|------------|-----------------------|-------------------------------|
| API        | http://localhost:8000 | API          |
| API Docs   | http://localhost:8000/docs | Interactive Docs          |
| Prometheus | http://localhost:9090 | Metrics scraping              |
| Grafana    | http://localhost:3000 | Pre-built dashboard (admin/admin) |


### Local (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --require-hashes -r requirements.txt

cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

---

## Configuration

All config lives in `.env`:

### Train

```http
POST /fit/{series_id}
```

```bash
curl -X POST http://localhost:8000/fit/sensor_temp \
  -H "Content-Type: application/json" \
  -d '{
    "timestamps": [
      1700000000, 1700000060, 1700000120, 1700000180, 1700000240,
      1700000300, 1700000360, 1700000420, 1700000480, 1700000540,
      1700000600, 1700000660, 1700000720, 1700000780, 1700000840,
      1700000900, 1700000960, 1700001020, 1700001080, 1700001140,
      1700001200, 1700001260, 1700001320, 1700001380, 1700001440,
      1700001500, 1700001560, 1700001620, 1700001680, 1700001740
    ],
    "values": [
      21.3, 21.5, 21.4, 21.6, 21.2, 21.8, 21.3, 21.5, 21.7, 21.4,
      21.6, 21.3, 21.5, 21.4, 21.6, 21.2, 21.7, 21.4, 21.5, 21.3,
      21.6, 21.4, 21.5, 21.3, 21.7, 21.4, 21.6, 21.5, 21.3, 21.4
    ]
  }'
```

**Response**

```json
{
  "series_id": "sensor_temp",
  "version": "v1",
  "points_used": 30
}
```

Calling `/fit` again on the same `series_id` creates `v2`, `v3`, etc. The previous versions remain queryable.

---

### Predict

```http
POST /predict/{series_id}?version=<optional>
```

```bash
# Use latest version (default)
curl -X POST http://localhost:8000/predict/sensor_temp \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1700000180", "value": 21.6}'

# Pin to a specific version
curl -X POST "http://localhost:8000/predict/sensor_temp?version=v1" \
  -H "Content-Type: application/json" \
  -d '{"timestamp": "1700000180", "value": 99.9}'
```

**Response**

```json
{
  "anomaly": false,
  "model_version": "v1"
}
```

---

### Health Check

```http
GET /healthcheck
```

```bash
curl http://localhost:8000/healthcheck
```

```json
{
  "series_trained": 3,
  "inference_latency_ms": { "avg": 1.2, "p95": 2.8 },
  "training_latency_ms": { "avg": 45.0, "p95": 60.1 }
}
```

---

### Plot (optional)

```http
GET http://localhost:8000/plot?series_id=sensor_temp&version=v1
```

Returns an interactive Plotly chart of the training data with the anomaly threshold overlaid. 

> version defaults to latest.

---

## Input validation

Training requests are rejected with `422` if:

- Fewer than `MIN_TRAIN_POINTS` data points (default 30)
- Any `NaN` or `Inf` value
- All values are constant, std=0 makes the model meaningless
- Timestamps are not strictly increasing, duplicates or out of order entries are rejected

---

## Storage layout

Each series gets its own folder under `STORAGE_ROOT`. Inside it, there's a `manifest.json` that tracks all versions with metadata (mean, std, trained_at, point count). Each version then gets its own subfolder containing the serialized model (`model.joblib`) and the raw training data used to fit it (`training_data.json`), which is what the `/plot` endpoint reads.

```
storage/                
    <series_id>/                # one directory per series_id
        manifest.json           # version list + metadata / version
        v1/                     # one directory / version
            model.joblib        # serialized object
            training_data.json  # raw training data used for plotting and validation
        v2/
        ...
```
---

## Tests

```bash
# Unit tests only
pytest tests/unit -v

# Integration tests (requires a running server)
pytest tests/integration -v
```

---

## Load testing

Requires a running server. Simulates 100 concurrent users, each owning a unique series.

```bash
pip install locust
locust -f benchmarks/locustfile.py --host http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 60s --headless
```

Results are printed to stdout at the end. A HTML report is available at `benchmarks/results/report.html`.

---

## Observability

Prometheus scrapes `/metrics`. 

The Grafana dashboard at `http://localhost:3000` shows request rate, latency percentiles, and active series count.

---

## Implementation Decisions

> Each decision below is documented with its full rationale in the source file where it is applied.
> Other minor decisions are in the source code.

**Timestamp accepted but not used in prediction** (`app/services/anomaly_service.py`): the timestamp is accepted in the request payload (per contract on OpenApi Spec) but ignored during inference, since the adopted statistical model (z-score) is time-invariant.

**ThreadPoolExecutor over ProcessPoolExecutor** (`app/core/executor.py`): numpy releases the GIL when running operations implemented in C, so threads are sufficient without the overhead of spawning separate processes.

**Per-series asyncio locks** (`app/services/anomaly_service.py`): concurrent requests on the same series are serialized; different series never block each other.

**joblib over pickle** (`app/core/persistence.py`): more efficient serialization for numpy-heavy objects; models are stored as lightweight `ModelParams` dataclasses rather than full model objects.

**Path traversal guard on `series_id`** (`app/core/persistence.py`): resolved path is checked against `STORAGE_ROOT` before any filesystem access.

**Single Uvicorn worker** (`.env.example`): scaling is done horizontally at the container level; multiple in-process workers would require distributed locking.

**`MIN_TRAIN_POINTS=30`** (`app/core/validation.py`): sample std is unreliable below this threshold.