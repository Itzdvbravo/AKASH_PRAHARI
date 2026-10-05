"""Semantic retrieval service combining embedding model and vector store."""
import base64
import io
import json
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image

from app.models.interfaces import EmbeddingModel
from app.vector_store.store_interface import VectorStore
from app.db.repositories.tile_repo import TileRepository
from app.schemas.common import GeoBBox, Confidence, TileRef, SensorType
from app.schemas.search import SearchResultItem, SearchFilters
from app.geospatial.tiling import parse_tile_id

import sys
from pathlib import Path
data_handling_dir = Path(__file__).resolve().parents[3] / "data-handling"
if str(data_handling_dir) not in sys.path:
    sys.path.insert(0, str(data_handling_dir))
from adapters.oscd.oscd_metadata import OSCD_CITY_COORDINATES


class RetrievalService:
    """Orchestrates embedding generation and ANN similarity search."""

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        tile_repo: TileRepository,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.tile_repo = tile_repo

    def search_by_text(
        self, query: str, filters: Optional[SearchFilters] = None, top_k: int = 10
    ) -> List[SearchResultItem]:
        q_text = query.strip() if query and query.strip() else "satellite observation of geographic area"
        query_vec = self.embedding_model.encode_text(q_text)
        candidate_count = self.vector_store.size if filters else top_k * 2
        matches = self.vector_store.search(query_vec, k=candidate_count)

        return self._format_results(matches, filters, top_k)

    def search_by_image(
        self, image_b64: str, filters: Optional[SearchFilters] = None, top_k: int = 10
    ) -> List[SearchResultItem]:
        image_bytes = base64.b64decode(image_b64)
        with Image.open(io.BytesIO(image_bytes)) as pil_img:
            arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0

        query_vec = self.embedding_model.encode_image(arr)
        candidate_count = self.vector_store.size if filters else top_k * 2
        matches = self.vector_store.search(query_vec, k=candidate_count)

        return self._format_results(matches, filters, top_k)

    def search_similar_sites(
        self, image: np.ndarray, source_location: str, top_k: int = 10
    ) -> List[SearchResultItem]:
        """Find cross-location embedding neighbors for analyst-led discovery."""
        query_vec = self.embedding_model.encode_image(np.asarray(image, dtype=np.float32))
        matches = self.vector_store.search(query_vec, k=self.vector_store.size)
        cross_site = []
        seen_locations = {source_location.lower()}
        for indexed_id, score in matches:
            tile_id = indexed_id.rsplit("@", 1)[0] if "@" in indexed_id else indexed_id
            record = self.tile_repo.get_any_by_id(tile_id)
            location = record.location_id if record else parse_tile_id(tile_id)["location_id"]
            if not location or location.lower() in seen_locations:
                continue
            seen_locations.add(location.lower())
            cross_site.append((tile_id, score))
            if len(cross_site) >= top_k:
                break
        return self._format_results(cross_site, None, top_k)

    def _format_results(
        self,
        matches: List[Tuple[str, float]],
        filters: Optional[SearchFilters],
        top_k: int
    ) -> List[SearchResultItem]:
        results: List[SearchResultItem] = []
        rank = 1

        for indexed_id, score in matches:
            # Incremental multi-date ingestion uses tile_id@date as the vector
            # key while relational metadata remains keyed by tile_id/date.
            tile_id, indexed_date = (
                indexed_id.rsplit("@", 1) if "@" in indexed_id else (indexed_id, None)
            )
            dates = self.tile_repo.list_dates_for_tile(tile_id)
            result_date = indexed_date if indexed_date in dates else (dates[-1] if dates else None)
            record = (
                self.tile_repo.get_by_id_and_date(tile_id, result_date)
                if result_date else self.tile_repo.get_any_by_id(tile_id)
            )

            tile_parts = parse_tile_id(tile_id)
            loc_id = tile_parts["location_id"] or "unknown"
            sensor_str = tile_parts["sensor"]

            if record:
                loc_id = record.location_id
                sensor_str = record.sensor
                try:
                    bbox_dict = json.loads(record.bbox_json)
                except Exception:
                    bbox_dict = OSCD_CITY_COORDINATES.get(loc_id, {"west": 0.0, "south": 0.0, "east": 0.05, "north": 0.05})
                cur_date = record.date
            else:
                bbox_dict = OSCD_CITY_COORDINATES.get(loc_id, {"west": 0.0, "south": 0.0, "east": 0.05, "north": 0.05})
                dates = [indexed_date] if indexed_date else ["2016-01-01", "2018-01-01"]
                cur_date = indexed_date or dates[0]

            # Apply filters
            if filters:
                if filters.location and filters.location.lower() != "all" and filters.location.lower() not in loc_id.lower():
                    continue
                if filters.sensor and filters.sensor.value != "any" and filters.sensor.value != sensor_str:
                    continue
                if filters.date_from and cur_date < filters.date_from:
                    continue
                if filters.date_to and cur_date > filters.date_to:
                    continue
                if filters.area_of_interest:
                    aoi = filters.area_of_interest
                    if (bbox_dict["east"] < aoi.west or bbox_dict["west"] > aoi.east
                            or bbox_dict["north"] < aoi.south or bbox_dict["south"] > aoi.north):
                        continue

            geo_bbox = GeoBBox(
                west=bbox_dict["west"],
                south=bbox_dict["south"],
                east=bbox_dict["east"],
                north=bbox_dict["north"]
            )

            sensor_enum = SensorType.SENTINEL_2
            if "sentinel-1" in sensor_str:
                sensor_enum = SensorType.SENTINEL_1
            elif "landsat" in sensor_str:
                sensor_enum = SensorType.LANDSAT

            results.append(SearchResultItem(
                rank=rank,
                tile_ref=TileRef(
                    tile_id=tile_id,
                    location_id=loc_id,
                    date=cur_date,
                    sensor=sensor_enum,
                ),
                confidence=Confidence(
                    score=round(max(0.0, min(float(score), 1.0)), 2),
                    method="cosine_similarity",
                    calibrated=False
                ),
                thumbnail_url=f"/api/v1/images/{tile_id}/{cur_date}/thumbnail",
                available_dates=dates,
                geo_bbox=geo_bbox
            ))

            rank += 1
            if len(results) >= top_k:
                break

        return results
