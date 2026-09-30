"""Image streaming and dates discovery endpoints."""
from fastapi import APIRouter, Request, Response, Query
from app.schemas.images import DatesResponse
from app.schemas.common import SensorType

router = APIRouter(tags=["Images"])


@router.get("/images/{location_id}/dates", response_model=DatesResponse)
async def get_location_dates(location_id: str, request: Request):
    image_service = request.app.state.image_service
    dates = image_service.get_available_dates(location_id)
    return DatesResponse(
        location_id=location_id,
        dates=dates,
        sensor=SensorType.SENTINEL_2
    )


@router.get("/images/{tile_id}/{date}/thumbnail")
async def get_tile_thumbnail(
    tile_id: str,
    date: str,
    request: Request
):
    image_service = request.app.state.image_service
    img_bytes = image_service.get_tile_bytes(tile_id, date, fmt="png", thumbnail=True)
    return Response(content=img_bytes, media_type="image/png")


@router.get("/images/{tile_id}/{date}")
async def get_tile_image(
    tile_id: str,
    date: str,
    request: Request,
    bands: str = Query("RGB"),
    format: str = Query("png")
):
    image_service = request.app.state.image_service
    img_bytes = image_service.get_tile_bytes(tile_id, date, fmt=format, thumbnail=False)
    media_type = "image/png" if format.lower() == "png" else "image/jpeg"
    return Response(content=img_bytes, media_type=media_type)


@router.get("/masks/{mask_filename}")
async def get_mask_image(mask_filename: str, request: Request):
    image_service = request.app.state.image_service
    mask_bytes = image_service.get_mask_bytes(mask_filename)
    return Response(content=mask_bytes, media_type="image/png")
