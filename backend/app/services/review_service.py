"""Durable analyst review trail and provenance-preserving candidate export."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Dict, List

from fastapi import HTTPException

from app.db.connection import DatabaseManager


class ReviewService:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def save_candidate(
        self,
        job_id: str,
        tile_id: str,
        location_id: str,
        date_before: str,
        date_after: str,
        candidate: Dict[str, Any],
        provenance: Dict[str, Any],
    ) -> None:
        with self.db.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO detection_candidates
                (job_id, tile_id, location_id, date_before, date_after,
                 candidate_json, provenance_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (job_id, tile_id, location_id, date_before, date_after,
                 json.dumps(candidate), json.dumps(provenance),
                 datetime.now(timezone.utc).isoformat()),
            )

    def list_candidates(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.db.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM detection_candidates ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            output = []
            for row in rows:
                item = dict(row)
                item["candidate"] = json.loads(item.pop("candidate_json"))
                item["provenance"] = json.loads(item.pop("provenance_json"))
                item["reviews"] = self._reviews(conn, item["job_id"])
                output.append(item)
            return output

    def get_candidate(self, job_id: str) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM detection_candidates WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Detection candidate not found")
            item = dict(row)
            item["candidate"] = json.loads(item.pop("candidate_json"))
            item["provenance"] = json.loads(item.pop("provenance_json"))
            item["reviews"] = self._reviews(conn, job_id)
            return item

    def add_review(self, job_id: str, decision: str, comment: str, reviewer: str) -> Dict[str, Any]:
        if decision not in {"confirmed", "rejected"}:
            raise HTTPException(status_code=422, detail="decision must be confirmed or rejected")
        with self.db.get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM detection_candidates WHERE job_id = ?", (job_id,)
            ).fetchone()
            if exists is None:
                raise HTTPException(status_code=404, detail="Detection candidate not found")
            cur = conn.execute(
                "INSERT INTO analyst_reviews (job_id, decision, comment, reviewer, created_at) VALUES (?, ?, ?, ?, ?)",
                (job_id, decision, comment, reviewer, datetime.now(timezone.utc).isoformat()),
            )
            row = conn.execute(
                "SELECT * FROM analyst_reviews WHERE review_id = ?", (cur.lastrowid,)
            ).fetchone()
            return dict(row)

    def export_geojson(self) -> Dict[str, Any]:
        features = []
        for item in self.list_candidates(limit=100000):
            bbox = item["provenance"].get("geo_bbox")
            if not bbox:
                continue
            ring = [[bbox["west"], bbox["south"]], [bbox["east"], bbox["south"]],
                    [bbox["east"], bbox["north"]], [bbox["west"], bbox["north"]],
                    [bbox["west"], bbox["south"]]]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [ring]},
                "properties": {
                    "job_id": item["job_id"], "tile_id": item["tile_id"],
                    "location_id": item["location_id"], "date_before": item["date_before"],
                    "date_after": item["date_after"], "candidate": item["candidate"],
                    "provenance": item["provenance"], "reviews": item["reviews"],
                },
            })
        return {"type": "FeatureCollection", "features": features}

    @staticmethod
    def _reviews(conn, job_id: str) -> List[Dict[str, Any]]:
        return [dict(row) for row in conn.execute(
            "SELECT review_id, decision, comment, reviewer, created_at FROM analyst_reviews WHERE job_id = ? ORDER BY review_id",
            (job_id,),
        ).fetchall()]
