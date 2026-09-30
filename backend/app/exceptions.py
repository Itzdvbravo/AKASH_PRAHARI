"""Custom exceptions and error response mapping for TerraEyes."""
from typing import Any, Dict, Optional
from fastapi import HTTPException, status


class TerraEyesError(Exception):
    """Base exception for all TerraEyes domain errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class TileNotFoundError(TerraEyesError):
    """Raised when a requested tile ID or date is not found."""
    pass


class CoregistrationError(TerraEyesError):
    """Raised when two temporal tiles cannot be co-registered."""
    pass


class ModelInferenceError(TerraEyesError):
    """Raised when an embedding or change detection model fails during inference."""
    pass


class DatasetError(TerraEyesError):
    """Raised when the underlying dataset or files are invalid or missing."""
    pass


def tile_not_found_exception(tile_id: str, date: Optional[str] = None) -> HTTPException:
    msg = f"Tile '{tile_id}'" + (f" for date '{date}'" if date else "") + " not found."
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=msg)


def invalid_query_exception(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def model_service_unavailable(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=detail)
