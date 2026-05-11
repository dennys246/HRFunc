"""Path-level I/O utilities for HRFunc — format detection, folder scanning,
and Raw caching for the v1.3.0 GUI.

Public API:
    classify_path(path) -> FormatHit | None
    scan_folder(root, ...) -> Manifest
    FormatHit, ScanEntry, ScanError, Manifest (dataclasses)
"""

from .detect import FormatHit, classify_path
from .manifest import Manifest, ScanEntry, ScanError
from .scan import scan_folder

__all__ = [
    "FormatHit",
    "Manifest",
    "ScanEntry",
    "ScanError",
    "classify_path",
    "scan_folder",
]
