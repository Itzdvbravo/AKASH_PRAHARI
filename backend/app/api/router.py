"""Main API router mounting all versioned sub-routers."""
from fastapi import APIRouter
from app.api.v1.router import v1_router

api_router = APIRouter(prefix="/api")
api_router.include_router(v1_router, prefix="/v1")
