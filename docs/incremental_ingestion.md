# Incremental scene ingestion

`data-handling/ingestion/incremental_ingest.py` supports local prototype
ingestion of one RGB acquisition from NPY/NPZ, a common image format, or a
three-band GeoTIFF. It preprocesses the scene, creates canonical tile IDs, and
writes dated arrays under `data/incremental_tiles` by default. The backend image
service reads those arrays for indexed results.

The caller can pass the already loaded embedding model and vector store to add
searchable embeddings. Pass `vector_index_path` to persist the updated index and
pass `tile_repo` plus a WGS84 `bbox` to persist dates and location metadata.
The function does not load models, create a database, or expose a network upload
endpoint; the planned `POST /api/v1/ingest` remains outside the prototype.

For a multi-temporal sequence on one fixed spatial grid, set
`allow_same_tile_across_dates=True`. This keeps geographic tile IDs stable,
allows one artifact per tile and date, and rejects re-ingesting the same
tile/date. If embeddings are enabled, the vector IDs include the date to keep
each acquisition distinct. The default remains one acquisition per tile ID.

```python
from ingestion.incremental_ingest import ingest_single_scene

result = ingest_single_scene(
    scene_path="./incoming/new_city.tif",
    location_id="new-city",
    date="2024-04-12",
    output_dir="./data/incremental_tiles",
    embedding_model=embedding_model,  # initialized with the same model as the index
    vector_store=vector_store,        # already loaded from the matching index
    vector_index_path="./data/faiss.index",
    tile_repo=tile_repo,
    bbox={"west": 1.0, "south": 2.0, "east": 3.0, "north": 4.0},
)
```

Use a new `location_id` for new geography. Reusing an existing location/grid
tile ID is rejected to avoid adding ambiguous duplicate vectors. GeoTIFF reading
requires `rasterio`; other supported image formats use Pillow.
