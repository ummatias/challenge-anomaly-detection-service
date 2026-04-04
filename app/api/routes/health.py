from fastapi import APIRouter, Depends
from app.schemas.health import HealthCheckResponse
from app.services.anomaly_service import AnomalyService, get_service

router = APIRouter(tags=["Health Check"])


@router.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck(service: AnomalyService = Depends(get_service)) -> HealthCheckResponse:
    """
    Returns system metrics:
      - Number of trained series in storage
      - Avg and p95 latency for training and inference (last 1000 requests)
    """
    return await service.health()
