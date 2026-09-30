"""Ingestion scripts and pipelines for TerraEyes."""
from .ingest_oscd import ingest_oscd_dataset
from .incremental_ingest import ingest_single_scene

__all__ = ["ingest_oscd_dataset", "ingest_single_scene"]
