"""Application configuration module using Pydantic Settings."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment & Server
    TERRAEYES_ENV: str = "development"
    TERRAEYES_HOST: str = "0.0.0.0"
    TERRAEYES_PORT: int = 8000
    TERRAEYES_LOG_LEVEL: str = "INFO"

    # Storage Paths
    TERRAEYES_DB_PATH: str = "./data/terraeyes.db"
    TERRAEYES_FAISS_INDEX_PATH: str = "./data/faiss.index"
    TERRAEYES_TEMPORAL_STORE_PATH: str = "./data/temporal_states.h5"
    TERRAEYES_MODELS_DIR: str = "./models"
    TERRAEYES_DATA_DIR: str = "./data"
    TERRAEYES_OSCD_DIR: str = "./data/oscd"

    # Model Selection
    TERRAEYES_EMBEDDING_MODEL: str = "mock"  # mock | remote_clip | georscclip
    TERRAEYES_CHANGE_DETECTOR: str = "pixel_diff"  # pixel_diff | bit_cd | mamba_cd
    TERRAEYES_TILE_SIZE: int = 256
    TERRAEYES_EMBEDDING_DIM: int = 512

    # Postprocessing
    TERRAEYES_MIN_CHANGE_AREA_PX: int = 25
    TERRAEYES_MASK_THRESHOLD: float = 0.25

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def ensure_directories(self) -> None:
        """Create configured data and model directories if they do not exist."""
        Path(self.TERRAEYES_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
