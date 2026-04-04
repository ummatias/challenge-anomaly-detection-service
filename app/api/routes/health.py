from fastapi import APIRouter
from app.schemas.health import HealthCheckResponse
from app.services.anomaly_service import anomaly_service

router = APIRouter(tags=["Health Check"])


@router.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck() -> HealthCheckResponse:
    """
    Returns system metrics:
      - Number of trained series in storage
      - Avg and p95 latency for training and inference (last 1000 requests)
    """
    return await anomaly_service.health()
