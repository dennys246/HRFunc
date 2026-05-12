"""Path-level I/O utilities for HRfunc — format detection, folder scanning,
and Raw caching for the v1.3.0 GUI.

Public API:
    classify_path(path) -> FormatHit | None
    scan_folder(root, ...) -> Manifest
    RawCache(maxsize=3) -> LRU cache of loaded MNE Raw objects
    FormatHit, ScanEntry, ScanError, Manifest (dataclasses)
"""

from .detect import FormatHit, classify_path
from .manifest import Manifest, ScanEntry, ScanError
from .raw_cache import RawCache
from .scan import scan_folder

__all__ = [
    "FormatHit",
    "Manifest",
    "RawCache",
    "ScanEntry",
    "ScanError",
    "classify_path",
    "scan_folder",
]
