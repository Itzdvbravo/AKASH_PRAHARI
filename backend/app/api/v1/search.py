"""Semantic retrieval search endpoint."""
from fastapi import APIRouter, Request, HTTPException, Query
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    orchestrator = request.app.state.query_orchestrator
    return orchestrator.search(req)


@router.get("/similar-sites/{tile_id}")
async def similar_sites(tile_id: str, request: Request, top_k: int = Query(10, ge=1, le=100)):
    repo = request.app.state.tile_repo
    dates = repo.list_dates_for_tile(tile_id)
    record = repo.get_by_id_and_date(tile_id, dates[-1]) if dates else repo.get_any_by_id(tile_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Tile is not present in the catalog")
    image = request.app.state.image_service.get_tile_array(tile_id, record.date)
    return request.app.state.retrieval_service.search_similar_sites(
        image, record.location_id, top_k
    )
