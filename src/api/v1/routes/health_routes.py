from fastapi import APIRouter

from src.core import config


router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


@router.get("")
def health_check():
    return {
        "status": "UP",
        "application": "Smart Banking Assistant",
        "environment": "development",
    }