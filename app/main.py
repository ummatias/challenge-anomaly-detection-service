from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import train, predict
from app.core.executor import executor


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield    
    executor.shutdown(wait=True)


def create_app() -> FastAPI:
    app = FastAPI(
        title="TSAD API",
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

    # Routers
    app.include_router(train.router)
    app.include_router(predict.router)

    return app


app = create_app()
