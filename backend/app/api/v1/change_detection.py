"""Change detection endpoint executing model inference and mask generation."""
from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from app.schemas.change_detection import ChangeDetectionRequest, ChangeDetectionResponse

router = APIRouter(tags=["Change Detection"])


@router.post("/change-detection", response_model=ChangeDetectionResponse)
async def run_change_detection(req: ChangeDetectionRequest, request: Request):
    cd_service = request.app.state.change_detection_service
    return await run_in_threadpool(cd_service.run_detection, req)
