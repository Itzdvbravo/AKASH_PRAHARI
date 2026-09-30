"""Health check endpoint."""
from fastapi import APIRouter, Request
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health(request: Request):
    app_state = request.app.state
    vector_size = app_state.vector_store.size if hasattr(app_state, "vector_store") else 0
    db_tiles = app_state.tile_repo.count_tiles() if hasattr(app_state, "tile_repo") else 0
    emb_model = getattr(app_state, "embedding_model_name", "mock")
    cd_detector = getattr(app_state, "change_detector_name", "pixel_diff")

    return HealthResponse(
        status="ok",
        version="0.1.0",
        embedding_model=emb_model,
        change_detector=cd_detector,
        faiss_index_size=vector_size,
        db_tiles=db_tiles
    )
