"""
AnomalyService

Singleton service class with all login for training and prediction.

- Maintain a series asyncio.Lock registry
- Run fit() in a thread pool (must not block the event loop)
- Coordinate validation -> fit -> persistence flow
- Load and run predictions with models saved

Concurrency:
  - Same series -> lock prevents reading a model that is being written simultaneously
  - Different series -> no blocking, they can train/predict concurrenly
  - fit() runs in ThreadPoolExecutor so numpy doesn't block the loop
"""

import asyncio
import time
from collections import defaultdict
from typing import Optional

from app.core import persistence, versioning, validation
from app.core.executor import executor
from app.core.metrics import LatencyTracker
from app.core.model import AnomalyDetectionModel
from app.core.metrics import series_trained_gauge
from app.schemas.training import TrainData, TrainResponse
from app.schemas.prediction import PredictResponse
from app.schemas.health import HealthCheckResponse, LatencyMetrics

class AnomalyService:
    def __init__(self) -> None:
        # Create a lock for each series
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._train_latency = LatencyTracker()
        self._infer_latency = LatencyTracker()

    def _get_lock(self, series_id: str) -> asyncio.Lock:
        # private helper to get the lock for a series_id
        return self._locks[series_id]


    async def train(self, series_id: str, data: TrainData) -> TrainResponse:
        """
            Runs validation (read-only) before acquiring the lock
            since it doesn't mutate state and can fail fast.
        """

        validation.validate(data.timestamps, data.values)
        async with self._get_lock(series_id):
            
            start_time = time.perf_counter()

            loop = asyncio.get_event_loop()
            params = await loop.run_in_executor(
                executor,
                self._fit_sync,
                data.values,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._train_latency.record(elapsed_ms)

            
            manifest = persistence.load_manifest(series_id)
            version = versioning.next_version(manifest)
            persistence.save_model(series_id, version, params)
            persistence.save_training_data(series_id, version, data.timestamps, data.values)
            updated_manifest = versioning.append_version(manifest, version, params)
            persistence.save_manifest(series_id, updated_manifest)

            # Update Prometheus gauge with the current number of trained series
            series_trained_gauge.set(len(persistence.list_series()))

        return TrainResponse(
            series_id=series_id,
            version=version,
            points_used=params.n_points,
        )

    @staticmethod
    def _fit_sync(values: list[float]):
        # Runs in a separate thread to avoid blocking the event loop.
        model = AnomalyDetectionModel()
        return model.fit(values)


    async def predict(
        self, series_id: str, value: float, version: Optional[str] = None
    ) -> PredictResponse:
        """
        Prediction needs to acquire the lock to ensure it doesn't read a model
        while its being updated by a concurrent train() call.
        """
        async with self._get_lock(series_id):
            start_time = time.perf_counter()

            manifest = persistence.load_manifest(series_id)
            if not manifest.get("versions"):
                raise LookupError(
                    f"No trained model found for series='{series_id}'. "
                    f"Train it first via POST /fit/{series_id}."
                )

            # If version is not specified, use the latest
            resolved = version or versioning.latest_version(manifest)
            if not versioning.version_exists(manifest, resolved):
                raise LookupError(
                    f"Version '{resolved}' not found for series='{series_id}'. "
                    f"Available: {[v['version'] for v in manifest['versions']]}"
                )

            params = persistence.load_model(series_id, resolved)
            model = AnomalyDetectionModel()
            model.load_params(params)
            is_anomaly = model.predict(value)

            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._infer_latency.record(elapsed_ms)

        return PredictResponse(anomaly=is_anomaly, model_version=resolved)


    async def health(self) -> HealthCheckResponse:
        series = persistence.list_series()
        return HealthCheckResponse(
            series_trained=len(series),
            inference_latency_ms=LatencyMetrics(avg=self._infer_latency.avg(), p95=self._infer_latency.p95()),
            training_latency_ms=LatencyMetrics(avg=self._train_latency.avg(), p95=self._train_latency.p95()),
        )

# Singleton instance of the service to be used across the app
anomaly_service = AnomalyService()

def get_service() -> AnomalyService:
    """
        Factory function to get the singleton service instance
        - Production: always returns the same instance
        - Testing: can be overridden to return a fresh instance for isolation
    """
    return anomaly_service