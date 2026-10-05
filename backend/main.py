"""TerraEyes FastAPI application factory and lifecycle startup."""
from contextlib import asynccontextmanager
import json
from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend and data-handling paths are in sys.path
backend_root = Path(__file__).resolve().parent
data_handling_root = backend_root.parent / "data-handling"
for p in (backend_root, data_handling_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from config import settings
from logging_config import setup_logging, get_logger
from app.api.router import api_router
from app.db.connection import DatabaseManager
from app.db.models import TileRecord
from app.db.repositories.tile_repo import TileRepository
from app.vector_store.faiss_store import FaissVectorStore
from app.models.embedding.mock_embedding import MockEmbeddingModel
from app.models.embedding.clip_vit_b32 import CLIPViTB32EmbeddingModel
from app.models.embedding.remote_clip import RemoteCLIPEmbeddingModel
from app.models.change_detection.pixel_diff import PixelDiffChangeDetector
from app.models.change_detection.bit_cd import BITChangeDetector
from app.models.change_detection.mamba_cd import MambaChangeDetector
from app.services.image_service import ImageService
from app.services.retrieval_service import RetrievalService
from app.services.comparison_service import ComparisonService
from app.services.change_detection_service import ChangeDetectionService
from app.services.temporal_state_store import TemporalStateStore
from app.services.summary_service import SummaryService
from app.services.query_orchestrator import QueryOrchestrator
from app.services.review_service import ReviewService
from adapters.oscd.oscd_metadata import OSCD_CITY_COORDINATES

logger = get_logger("terraeyes.main")

PROTOTYPE_EXTRA_CITIES = {
    "norcia": {"west": 13.08, "south": 42.78, "east": 13.11, "north": 42.81},
}


def seed_initial_catalog(tile_repo: TileRepository, vector_store: FaissVectorStore, embedding_model) -> None:
    """Adds any missing demo catalog entries without duplicating indexed tiles."""
    cities = {**OSCD_CITY_COORDINATES, **PROTOTYPE_EXTRA_CITIES}
    logger.info("seeding_initial_catalog", cities_count=len(cities))
    vectors_to_add = []
    ids_to_add = []

    for city, bbox in cities.items():
        tile_id = f"{city}_0001_0001_sentinel-2"
        if tile_repo.get_any_by_id(tile_id) is not None:
            continue
        dates = ["2016-03-15", "2018-06-20"]
        for dt in dates:
            rec = TileRecord(
                tile_id=tile_id,
                location_id=city,
                sensor="sentinel-2",
                date=dt,
                bbox_json=json.dumps(bbox),
                crs="EPSG:4326",
                resolution_m=10.0,
                embedding_id=tile_id,
                ingested_at="2026-09-30T12:00:00Z"
            )
            tile_repo.insert_tile(rec)

        # Generate embedding vector for retrieval catalog
        text_desc = f"satellite view of {city} urban area and surrounding landscape"
        vec = embedding_model.encode_text(text_desc)
        vectors_to_add.append(vec)
        ids_to_add.append(tile_id)

    if vectors_to_add:
        import numpy as np
        vector_store.add(np.array(vectors_to_add), ids_to_add)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.TERRAEYES_LOG_LEVEL)
    logger.info("startup_init", env=settings.TERRAEYES_ENV)
    settings.ensure_directories()

    # 1. Initialize SQLite Database
    db_mgr = DatabaseManager(settings.TERRAEYES_DB_PATH)
    db_mgr.init_db()
    tile_repo = TileRepository(db_mgr)

    # 2. Model Factory
    if settings.TERRAEYES_EMBEDDING_MODEL == "clip_vit_b32":
        embedding_model = CLIPViTB32EmbeddingModel(
            checkpoint_path=settings.TERRAEYES_CLIP_CHECKPOINT_PATH
        )
    elif settings.TERRAEYES_EMBEDDING_MODEL == "remote_clip":
        embedding_model = RemoteCLIPEmbeddingModel(
            checkpoint_path=settings.TERRAEYES_REMOTECLIP_CHECKPOINT_PATH
        )
    elif settings.TERRAEYES_EMBEDDING_MODEL == "mock":
        embedding_model = MockEmbeddingModel(dim=settings.TERRAEYES_EMBEDDING_DIM)
    else:
        raise ValueError(
            f"Unsupported embedding model: {settings.TERRAEYES_EMBEDDING_MODEL}"
        )

    # 3. Load an index only when its model fingerprint matches the configured
    # encoder. CLIP and mock vectors share a dimension but are not compatible.
    vector_store = FaissVectorStore(dim=embedding_model.embedding_dim)
    index_metadata_path = Path(f"{settings.TERRAEYES_FAISS_INDEX_PATH}.meta.json")
    index_metadata = None
    if index_metadata_path.is_file():
        try:
            index_metadata = json.loads(index_metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("vector_index_metadata_invalid", path=str(index_metadata_path))

    index_compatible = bool(
        index_metadata
        and index_metadata.get("model") == settings.TERRAEYES_EMBEDDING_MODEL
        and index_metadata.get("embedding_dim") == embedding_model.embedding_dim
    )
    persisted_index_metadata = dict(index_metadata) if index_compatible else {}
    if index_compatible:
        vector_store.load(settings.TERRAEYES_FAISS_INDEX_PATH)
    elif settings.TERRAEYES_EMBEDDING_MODEL == "mock" and index_metadata is None:
        # Read indexes produced by the original Phase A prototype.
        vector_store.load(settings.TERRAEYES_FAISS_INDEX_PATH)
    elif settings.TERRAEYES_EMBEDDING_MODEL != "mock":
        logger.warning(
            "vector_index_not_loaded",
            configured_model=settings.TERRAEYES_EMBEDDING_MODEL,
            metadata_path=str(index_metadata_path),
        )

    if settings.TERRAEYES_EMBEDDING_MODEL == "clip_vit_b32" and vector_store.size == 0:
        raise RuntimeError(
            "No CLIP-compatible image index is available. Run scripts/ingest_oscd.py "
            "and scripts/build_index.py before starting the API."
        )

    temporal_state_store = None
    if settings.TERRAEYES_CHANGE_DETECTOR == "bit_cd":
        change_detector = BITChangeDetector()
    elif settings.TERRAEYES_CHANGE_DETECTOR == "mamba_cd":
        change_detector = MambaChangeDetector(
            weights_path=settings.TERRAEYES_MAMBA_CHECKPOINT_PATH,
        )
        temporal_state_store = TemporalStateStore(settings.TERRAEYES_TEMPORAL_STORE_PATH)
    else:
        change_detector = PixelDiffChangeDetector()

    # 4. Keep the synthetic catalog confined to prototype mode. Phase B uses
    # the train-only image index built from the local OSCD archive.
    if settings.TERRAEYES_EMBEDDING_MODEL == "mock":
        seed_initial_catalog(tile_repo, vector_store, embedding_model)

    # 5. Instantiate Services
    image_service = ImageService(
        tile_repo=tile_repo,
        oscd_dir=settings.TERRAEYES_OSCD_DIR,
        masks_cache_dir=str(Path(settings.TERRAEYES_DATA_DIR) / "masks"),
        incremental_tiles_dir=str(Path(settings.TERRAEYES_DATA_DIR) / "incremental_tiles"),
    )
    retrieval_service = RetrievalService(
        embedding_model=embedding_model,
        vector_store=vector_store,
        tile_repo=tile_repo
    )
    comparison_service = ComparisonService(image_service=image_service)
    summary_service = SummaryService()
    review_service = ReviewService(db_mgr)
    cd_service = ChangeDetectionService(
        change_detector=change_detector,
        image_service=image_service,
        comparison_service=comparison_service,
        summary_service=summary_service,
        min_change_area_px=settings.TERRAEYES_MIN_CHANGE_AREA_PX,
        temporal_state_store=temporal_state_store,
        review_service=review_service,
    )
    query_orchestrator = QueryOrchestrator(retrieval_service=retrieval_service)

    # 6. Bind Singletons to App State
    app.state.settings = settings
    app.state.db_mgr = db_mgr
    app.state.tile_repo = tile_repo
    app.state.vector_store = vector_store
    app.state.embedding_model = embedding_model
    app.state.embedding_model_name = settings.TERRAEYES_EMBEDDING_MODEL
    app.state.change_detector = change_detector
    app.state.change_detector_name = settings.TERRAEYES_CHANGE_DETECTOR
    app.state.image_service = image_service
    app.state.retrieval_service = retrieval_service
    app.state.comparison_service = comparison_service
    app.state.change_detection_service = cd_service
    app.state.temporal_state_store = temporal_state_store
    app.state.summary_service = summary_service
    app.state.query_orchestrator = query_orchestrator
    app.state.review_service = review_service

    yield

    # Shutdown
    try:
        vector_store.save(settings.TERRAEYES_FAISS_INDEX_PATH)
        persisted_index_metadata.update({
            "model": settings.TERRAEYES_EMBEDDING_MODEL,
            "embedding_dim": embedding_model.embedding_dim,
            "tile_count": vector_store.size,
        })
        index_metadata_path.parent.mkdir(parents=True, exist_ok=True)
        index_metadata_path.write_text(
            json.dumps(persisted_index_metadata, indent=2), encoding="utf-8"
        )
        logger.info("shutdown_complete")
    except Exception as e:
        logger.error("shutdown_error", error=str(e))


def create_app() -> FastAPI:
    app = FastAPI(
        title="TerraEyes Analyst API",
        version="0.1.0",
        description="Geospatial Semantic Retrieval & Multi-Temporal Change Detection Subsystem",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.TERRAEYES_HOST,
        port=settings.TERRAEYES_PORT,
        reload=True
    )
