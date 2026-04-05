from contextlib import asynccontextmanager
from app.core import persistence
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.routes import train, predict, health, plot
from app.core.executor import executor
from app.core.metrics import series_trained_gauge



@asynccontextmanager
async def lifespan(app: FastAPI):
    series_trained_gauge.set(len(persistence.list_series()))
    yield    
    executor.shutdown(wait=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TSAD API", #Time Series Anomaly Detection API
        version="1.0.0",
        description=(
            "Train and serve anomaly detection model for univariate time series."
        ),
        lifespan=lifespan,
    )

    # CORS: open for development, can be restricted in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    """
        Prometheus instrumentation, tracks latency and request count for all endpoints.
        include_in_schema=False hides the /metrics endpoint from the OpenAPI, Grafana
        will scrape directly from /metrics.
    """
    Instrumentator().instrument(app).expose(
        app,
        include_in_schema=False,
        endpoint="/metrics",
        tags=["monitoring"],
    )
    
    app.include_router(train.router)
    app.include_router(predict.router)
    app.include_router(health.router)
    app.include_router(plot.router)

    return app


app = create_app()
