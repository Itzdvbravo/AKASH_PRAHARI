"""Analyst review queue, append-only feedback, and GeoJSON export."""
from typing import Literal

from fastapi import APIRouter, Request, Query
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Analyst Review"])


class ReviewRequest(BaseModel):
    decision: Literal["confirmed", "rejected"]
    comment: str = Field(default="", max_length=2000)
    reviewer: str = Field(default="", max_length=200)


@router.get("/review-queue")
async def review_queue(request: Request, limit: int = Query(100, ge=1, le=1000)):
    return request.app.state.review_service.list_candidates(limit)


@router.get("/review-queue/export.geojson")
async def export_review_queue(request: Request):
    return JSONResponse(request.app.state.review_service.export_geojson())


@router.get("/review-queue/{job_id}")
async def get_review_candidate(job_id: str, request: Request):
    return request.app.state.review_service.get_candidate(job_id)


@router.post("/review-queue/{job_id}/reviews", status_code=201)
async def submit_review(job_id: str, payload: ReviewRequest, request: Request):
    return request.app.state.review_service.add_review(
        job_id, payload.decision, payload.comment, payload.reviewer
    )

