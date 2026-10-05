"""Application configuration module using Pydantic Settings."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    # Environment & Server
    TERRAEYES_ENV: str = "development"
    TERRAEYES_HOST: str = "0.0.0.0"
    TERRAEYES_PORT: int = 8000
    TERRAEYES_LOG_LEVEL: str = "INFO"

    # Storage Paths
    TERRAEYES_DB_PATH: str = "./data/dynamicearthnet.db"
    TERRAEYES_FAISS_INDEX_PATH: str = "./data/faiss_dynamicearthnet.index"
    TERRAEYES_TEMPORAL_STORE_PATH: str = "./data/temporal_states.h5"
    TERRAEYES_MODELS_DIR: str = "./models"
    TERRAEYES_DATA_DIR: str = "./data"
    TERRAEYES_DNE_ARCHIVE: str = "./data/dynamicearthnet/dynamicearthnet-video-71psnr.tacozip"
    TERRAEYES_DNE_CACHE_DIR: str = "./data/dynamicearthnet/decoded_frames"

    # Model Selection
    TERRAEYES_EMBEDDING_MODEL: str = "clip_vit_b32"  # mock | clip_vit_b32 | remote_clip
    TERRAEYES_CHANGE_DETECTOR: str = "pixel_diff"  # semantic_mamba is experimental until held-out validation passes
    TERRAEYES_TILE_SIZE: int = 256
    TERRAEYES_EMBEDDING_DIM: int = 512
    TERRAEYES_CLIP_CHECKPOINT_PATH: str = "./models/clip_vit_b32.pt"
    TERRAEYES_REMOTECLIP_CHECKPOINT_PATH: str = "./models/remoteclip_vit_b32.pt"
    TERRAEYES_MAMBA_CHECKPOINT_PATH: str = "./models/change_detection_candidates/mamba_oscd_best.pt"
    TERRAEYES_SEMANTIC_MAMBA_CHECKPOINT_PATH: str = "./models/change_detection_candidates/mamba_dynamicearthnet_semantic.pt"

    # Postprocessing
    TERRAEYES_MIN_CHANGE_AREA_PX: int = 25
    TERRAEYES_MASK_THRESHOLD: float = 0.3

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def model_post_init(self, __context) -> None:
        """Anchor configured filesystem paths to the repo, not the shell cwd."""
        path_fields = (
            "TERRAEYES_DB_PATH",
            "TERRAEYES_FAISS_INDEX_PATH",
            "TERRAEYES_TEMPORAL_STORE_PATH",
            "TERRAEYES_MODELS_DIR",
            "TERRAEYES_DATA_DIR",
            "TERRAEYES_DNE_ARCHIVE",
            "TERRAEYES_DNE_CACHE_DIR",
            "TERRAEYES_CLIP_CHECKPOINT_PATH",
            "TERRAEYES_REMOTECLIP_CHECKPOINT_PATH",
            "TERRAEYES_MAMBA_CHECKPOINT_PATH",
            "TERRAEYES_SEMANTIC_MAMBA_CHECKPOINT_PATH",
        )
        for field in path_fields:
            path = Path(getattr(self, field)).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            object.__setattr__(self, field, str(path.resolve()))

    def ensure_directories(self) -> None:
        """Create configured data and model directories if they do not exist."""
        Path(self.TERRAEYES_DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_MODELS_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(self.TERRAEYES_FAISS_INDEX_PATH).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
