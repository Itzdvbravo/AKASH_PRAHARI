"""SQLite database connection and schema initialization."""
from pathlib import Path
import sqlite3
from typing import Generator
from contextlib import contextmanager

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tiles (
    tile_id TEXT NOT NULL,
    location_id TEXT NOT NULL,
    sensor TEXT NOT NULL,
    date TEXT NOT NULL,
    bbox_json TEXT NOT NULL,
    crs TEXT DEFAULT 'EPSG:4326',
    resolution_m REAL DEFAULT 10.0,
    embedding_id TEXT,
    ingested_at TEXT,
    PRIMARY KEY (tile_id, date)
);

CREATE INDEX IF NOT EXISTS idx_tiles_location ON tiles(location_id);
CREATE INDEX IF NOT EXISTS idx_tiles_date ON tiles(date);
CREATE INDEX IF NOT EXISTS idx_tiles_sensor ON tiles(sensor);

CREATE TABLE IF NOT EXISTS tile_metadata (
    tile_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (tile_id, key)
);

CREATE TABLE IF NOT EXISTS temporal_states (
    tile_id TEXT NOT NULL,
    date TEXT NOT NULL,
    state_path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (tile_id, date)
);

CREATE TABLE IF NOT EXISTS detection_candidates (
    job_id TEXT PRIMARY KEY,
    tile_id TEXT NOT NULL,
    location_id TEXT NOT NULL,
    date_before TEXT NOT NULL,
    date_after TEXT NOT NULL,
    candidate_json TEXT NOT NULL,
    provenance_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyst_reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    decision TEXT NOT NULL CHECK(decision IN ('confirmed', 'rejected')),
    comment TEXT NOT NULL DEFAULT '',
    reviewer TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES detection_candidates(job_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_job ON analyst_reviews(job_id, review_id);
"""


class DatabaseManager:
    """Manages SQLite database connections and schema migration."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_db(self) -> None:
        """Initializes tables and pragmas."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
