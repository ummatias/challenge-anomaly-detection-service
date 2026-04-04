from typing import Optional
from fastapi import APIRouter, HTTPException, Path, Query
from app.schemas.prediction import PredictData, PredictResponse
from app.services.anomaly_service import anomaly_service

router = APIRouter(tags=["Prediction"])


@router.post("/predict/{series_id}", response_model=PredictResponse)
async def predict(
    series_id: str = Path(..., description="Unique identifier for the time series"),
    version: Optional[str] = Query(None, description="Model version (defaults to latest)"),
    body: PredictData = ...,
) -> PredictResponse:
    """
    Predict if a value is an anomaly for a given series_id using a trained model.
    If version is not specified, the latest trained model for that series_id will be used.
    """
    try:
        return await anomaly_service.predict(series_id, body.value, version)
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
