"""Comparison endpoint for bi-temporal tile pairing."""
from fastapi import APIRouter, Request
from app.schemas.comparison import ComparisonRequest, ComparisonResponse

router = APIRouter(tags=["Comparison"])


@router.post("/comparison", response_model=ComparisonResponse)
@router.post("/compare", response_model=ComparisonResponse)
async def create_comparison(req: ComparisonRequest, request: Request):
    comparison_service = request.app.state.comparison_service
    return comparison_service.create_comparison(req)
