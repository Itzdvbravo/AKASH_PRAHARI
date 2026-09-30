"""Mount all v1 API routers."""
from fastapi import APIRouter
from .health import router as health_router
from .search import router as search_router
from .images import router as images_router
from .comparison import router as comparison_router
from .change_detection import router as cd_router

v1_router = APIRouter()

v1_router.include_router(health_router)
v1_router.include_router(search_router)
v1_router.include_router(images_router)
v1_router.include_router(comparison_router)
v1_router.include_router(cd_router)
