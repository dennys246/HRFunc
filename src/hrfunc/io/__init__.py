"""Path-level I/O utilities for HRFunc — format detection, folder scanning,
and Raw caching for the v1.3.0 GUI.

Public API:
    classify_path(path) -> FormatHit | None
    FormatHit (dataclass)
"""

from .detect import FormatHit, classify_path

__all__ = ["FormatHit", "classify_path"]
