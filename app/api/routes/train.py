from fastapi import APIRouter, HTTPException, Path
from app.schemas.training import TrainData, TrainResponse
from app.services.anomaly_service import anomaly_service
from app.core.validation import ValidationError

router = APIRouter(tags=["Training"])


@router.post("/fit/{series_id}", response_model=TrainResponse)
async def fit(
    series_id: str = Path(..., description="Unique identifier for the time series"),
    body: TrainData = ...,
) -> TrainResponse:
    """
    Train (or retrain) an anomaly detection model for a series_id.
    Each call creates a new version of the model, and the latest version is used for inference.

    The training process is thread-safe, allowing concurrent training and inference on the same series_id. 
    """
    try:
        return await anomaly_service.train(series_id, body)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"rule": e.rule, "detail": e.detail})
