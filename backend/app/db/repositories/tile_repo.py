"""Tile repository for metadata persistence and querying."""
import json
from typing import Dict, List, Optional
from app.db.connection import DatabaseManager
from app.db.models import TileRecord


class TileRepository:
    """Provides structured queries against the tiles and metadata tables."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    def insert_tile(self, tile: TileRecord) -> None:
        sql = """
        INSERT OR REPLACE INTO tiles (
            tile_id, location_id, sensor, date, bbox_json, crs, resolution_m, embedding_id, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self.db.get_connection() as conn:
            conn.execute(sql, (
                tile.tile_id,
                tile.location_id,
                tile.sensor,
                tile.date,
                tile.bbox_json,
                tile.crs,
                tile.resolution_m,
                tile.embedding_id,
                tile.ingested_at,
            ))

    def get_by_id_and_date(self, tile_id: str, date: str) -> Optional[TileRecord]:
        sql = "SELECT * FROM tiles WHERE tile_id = ? AND date = ?"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, (tile_id, date))
            row = cur.fetchone()
            if row:
                return TileRecord(**dict(row))
        return None

    def get_any_by_id(self, tile_id: str) -> Optional[TileRecord]:
        sql = "SELECT * FROM tiles WHERE tile_id = ? LIMIT 1"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, (tile_id,))
            row = cur.fetchone()
            if row:
                return TileRecord(**dict(row))
        return None

    def list_dates_for_location(self, location_id: str) -> List[str]:
        sql = "SELECT DISTINCT date FROM tiles WHERE location_id = ? ORDER BY date ASC"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, (location_id,))
            return [row["date"] for row in cur.fetchall()]

    def list_dates_for_tile(self, tile_id: str) -> List[str]:
        sql = "SELECT DISTINCT date FROM tiles WHERE tile_id = ? ORDER BY date ASC"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, (tile_id,))
            return [row["date"] for row in cur.fetchall()]

    def list_locations(self) -> List[str]:
        sql = "SELECT DISTINCT location_id FROM tiles ORDER BY location_id ASC"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            return [row["location_id"] for row in cur.fetchall()]

    def count_tiles(self) -> int:
        sql = "SELECT COUNT(*) as cnt FROM tiles"
        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            row = cur.fetchone()
            return int(row["cnt"]) if row else 0

    def search_filtered(
        self,
        location: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        sensor: Optional[str] = None
    ) -> List[TileRecord]:
        conditions = []
        params = []

        if location:
            conditions.append("LOWER(location_id) LIKE ?")
            params.append(f"%{location.lower()}%")
        if date_from:
            conditions.append("date >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("date <= ?")
            params.append(date_to)
        if sensor and sensor != "any":
            conditions.append("LOWER(sensor) = ?")
            params.append(sensor.lower())

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        sql = f"SELECT * FROM tiles {where_clause} ORDER BY date DESC LIMIT 100"

        with self.db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [TileRecord(**dict(row)) for row in cur.fetchall()]
