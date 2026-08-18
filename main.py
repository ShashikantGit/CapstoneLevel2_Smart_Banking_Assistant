from fastapi import FastAPI

from src.api.v1.routes.health_routes import (
    router as health_router,
)

from src.api.v1.routes.query_routes import (
    router as query_router,
)

from src.api.v1.upload.upload import (
    router as upload_router,
)


app = FastAPI(
    title="Smart Banking Assistant",
    description=(
        "Agentic AI Smart Banking Assistant "
        "for BFSI domain."
    ),
    version="1.0.0",
)


@app.get(
    "/",
    tags=["Root"],
)
def home():
    return {
        "message": "Smart Banking Assistant is running",
        "application": "Smart Banking Assistant",
        "version": "1.0.0",
    }


app.include_router(
    health_router,
    prefix="/api/v1",
)

app.include_router(
    query_router,
    prefix="/api/v1",
)

app.include_router(
    upload_router,
    prefix="/api/v1",
)