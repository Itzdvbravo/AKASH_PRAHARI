import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ingestion.incremental_ingest import ingest_single_scene

backend_root = Path(__file__).resolve().parents[2] / "backend"
if str(backend_root) not in sys.path:
    sys.path.insert(0, str(backend_root))

from app.models.embedding.mock_embedding import MockEmbeddingModel
from app.services.image_service import ImageService
from app.vector_store.faiss_store import FaissVectorStore
from app.db.connection import DatabaseManager
from app.db.repositories.tile_repo import TileRepository


def test_ingest_scene_appends_tiles_and_preserves_existing_index(tmp_path):
    source = tmp_path / "new_scene.npy"
    np.save(source, np.random.default_rng(4).uniform(0, 1, (300, 300, 3)).astype(np.float32))
    output = tmp_path / "artifacts"
    index_path = tmp_path / "faiss.index"

    model = MockEmbeddingModel(dim=32)
    store = FaissVectorStore(dim=32)
    db = DatabaseManager(str(tmp_path / "tiles.db"))
    db.init_db()
    tile_repo = TileRepository(db)
    original = model.encode_text("existing indexed scene")
    store.add(original[None, :], ["existing_0000_0000_sentinel-2"])
    store.save(str(index_path))

    result = ingest_single_scene(
        source,
        "New Location",
        "2024-04-12",
        output_dir=output,
        tile_size=256,
        embedding_model=model,
        vector_store=store,
        vector_index_path=index_path,
        tile_repo=tile_repo,
        bbox={"west": 1.0, "south": 2.0, "east": 3.0, "north": 4.0},
    )

    assert result["status"] == "success"
    assert result["tiles_created"] == 1
    assert result["index_size"] == 2
    persisted = FaissVectorStore(dim=32)
    persisted.load(str(index_path))
    assert persisted.size == 2
    assert persisted.search(original, 2)[0][0] == "existing_0000_0000_sentinel-2"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tiles"][0]["date"] == "2024-04-12"
    assert Path(output / manifest["tiles"][0]["array_path"]).is_file()
    assert tile_repo.list_dates_for_tile(result["tile_ids"][0]) == ["2024-04-12"]
    image_service = ImageService(
        tile_repo=None,
        oscd_dir=str(tmp_path / "missing_oscd"),
        masks_cache_dir=str(tmp_path / "masks"),
        incremental_tiles_dir=str(output),
    )
    loaded = image_service.get_tile_array(result["tile_ids"][0], "2024-04-12")
    assert loaded.shape == (256, 256, 3)


def test_ingest_rejects_duplicate_without_changing_index_or_manifest(tmp_path):
    source = tmp_path / "scene.npy"
    np.save(source, np.ones((256, 256, 3), dtype=np.float32))
    output = tmp_path / "artifacts"
    kwargs = {"output_dir": output, "tile_size": 256}
    ingest_single_scene(source, "paris", "2024-01-01", **kwargs)
    manifest_before = (output / "manifest.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Tile IDs already exist"):
        ingest_single_scene(source, "paris", "2024-01-02", **kwargs)

    assert (output / "manifest.json").read_text(encoding="utf-8") == manifest_before
    assert len(list((output / "paris").glob("*.npz"))) == 1


def test_ingest_temporal_acquisitions_reuse_spatial_tile_ids(tmp_path):
    output = tmp_path / "artifacts"
    db = DatabaseManager(str(tmp_path / "tiles.db"))
    db.init_db()
    tile_repo = TileRepository(db)
    model = MockEmbeddingModel(dim=32)
    store = FaissVectorStore(dim=32)
    image_service = ImageService(
        tile_repo=None,
        oscd_dir=str(tmp_path / "missing_oscd"),
        masks_cache_dir=str(tmp_path / "masks"),
        incremental_tiles_dir=str(output),
    )

    first_source = tmp_path / "first.npy"
    second_source = tmp_path / "second.npy"
    np.save(first_source, np.zeros((256, 256, 3), dtype=np.float32))
    np.save(second_source, np.ones((256, 256, 3), dtype=np.float32))
    kwargs = {
        "location_id": "temporal-city",
        "sensor": "planet",
        "output_dir": output,
        "tile_size": 256,
        "allow_same_tile_across_dates": True,
        "tile_repo": tile_repo,
        "embedding_model": model,
        "vector_store": store,
        "vector_index_path": tmp_path / "temporal.index",
        "bbox": {"west": 1.0, "south": 2.0, "east": 3.0, "north": 4.0},
    }

    before = ingest_single_scene(first_source, date="2018-01-01", **kwargs)
    after = ingest_single_scene(second_source, date="2018-02-01", **kwargs)

    assert before["tile_ids"] == after["tile_ids"]
    assert before["vector_ids"] == [f"{before['tile_ids'][0]}@2018-01-01"]
    assert after["vector_ids"] == [f"{after['tile_ids'][0]}@2018-02-01"]
    assert store.size == 2
    assert tile_repo.list_dates_for_tile(before["tile_ids"][0]) == [
        "2018-01-01", "2018-02-01"
    ]
    assert tile_repo.get_by_id_and_date(before["tile_ids"][0], "2018-01-01").embedding_id == before["vector_ids"][0]
    assert tile_repo.get_by_id_and_date(after["tile_ids"][0], "2018-02-01").embedding_id == after["vector_ids"][0]
    assert float(image_service.get_tile_array(before["tile_ids"][0], "2018-01-01").mean()) < 0.01
    assert float(image_service.get_tile_array(after["tile_ids"][0], "2018-02-01").mean()) > 0.99

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert [row["date"] for row in manifest["tiles"]] == ["2018-01-01", "2018-02-01"]

    with pytest.raises(ValueError, match="Tile IDs already exist"):
        ingest_single_scene(
            second_source,
            "temporal-city",
            "2018-02-01",
            sensor="planet",
            output_dir=output,
            tile_size=256,
            allow_same_tile_across_dates=True,
        )


def test_ingest_validates_scene_and_date(tmp_path):
    source = tmp_path / "scene.npy"
    np.save(source, np.ones((16, 16), dtype=np.float32))

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        ingest_single_scene(source, "paris", "not-a-date")
    with pytest.raises(ValueError, match="Expected an HxWx3"):
        ingest_single_scene(source, "paris", "2024-01-01")
