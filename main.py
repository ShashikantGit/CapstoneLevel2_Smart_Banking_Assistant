from fastapi import FastAPI

from app.routes.query_routes import router as query_router


app = FastAPI(
    title="Smart Banking Assistant",
    version="1.0.0",
)


@app.get("/")
def home():
    return {"message": "Smart Banking Assistant is running"}


app.include_router(query_router)