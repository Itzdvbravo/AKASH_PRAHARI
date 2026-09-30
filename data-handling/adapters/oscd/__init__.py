"""OSCD (Onera Satellite Change Detection) dataset adapter package."""
from .oscd_adapter import OSCDAdapter
from .oscd_loader import OSCDLoader
from .oscd_metadata import OSCDMetadataExtractor

__all__ = ["OSCDAdapter", "OSCDLoader", "OSCDMetadataExtractor"]
