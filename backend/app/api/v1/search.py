"""Semantic retrieval search endpoint."""
from fastapi import APIRouter, Request
from app.schemas.search import SearchRequest, SearchResponse

router = APIRouter(tags=["Search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, request: Request):
    orchestrator = request.app.state.query_orchestrator
    return orchestrator.search(req)
