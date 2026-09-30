"""Health check request and response schemas."""
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")
    embedding_model: str
    change_detector: str
    faiss_index_size: int = Field(default=0)
    db_tiles: int = Field(default=0)
