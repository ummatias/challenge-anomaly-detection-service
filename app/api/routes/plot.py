from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse

from app.services.plot_service import PlotService, get_plot_service

router = APIRouter(tags=["Visualization"])


@router.get("/plot", response_class=HTMLResponse, summary="Plot training data")
async def plot(
    series_id: str = Query(..., description="Time series identifier"),
    version: str | None = Query(None, description="Model version; defaults to latest"),
    service: PlotService = Depends(get_plot_service),
) -> HTMLResponse:
    """
    Generate an HTML plot of the training data for a given series_id and model version.
    If version is not specified, the latest trained model for that series_id will be used.
    """
    try:
        return HTMLResponse(service.plot(series_id, version))
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
