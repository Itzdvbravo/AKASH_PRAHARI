"""Query orchestrator coordinating multi-modal search and filtering."""
import time
import uuid
from typing import Optional
from app.services.retrieval_service import RetrievalService
from app.schemas.search import SearchRequest, SearchResponse


class QueryOrchestrator:
    """Dispatches search requests to retrieval services and applies ranking & filters."""

    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    def search(self, req: SearchRequest) -> SearchResponse:
        start_t = time.perf_counter()
        query_id = str(uuid.uuid4())

        if req.query_type == "text":
            items = self.retrieval_service.search_by_text(
                query=req.query_text or "",
                filters=req.filters,
                top_k=req.top_k
            )
        elif req.query_type == "image" and req.query_image_b64:
            items = self.retrieval_service.search_by_image(
                image_b64=req.query_image_b64,
                filters=req.filters,
                top_k=req.top_k
            )
        else:
            items = []

        query_ms = int((time.perf_counter() - start_t) * 1000)

        return SearchResponse(
            query_id=query_id,
            results=items,
            total=len(items),
            query_ms=max(query_ms, 1)
        )
