"""HRfunc workspace folder — outputs the GUI saves to disk.

Researchers running HRfunc accumulate analysis outputs (ROI-averaged
HRFs, exported HRF JSONs, etc.) that need a stable on-disk home so
they aren't fishing through temp directories or downloading folders
to find their saved files. This module owns the convention:

- Default location: ``~/hrfunc_workspace/`` (cross-platform — works on
  macOS, Linux, and Windows since every OS has a home directory).
- Overridable via the ``HRFUNC_WORKSPACE`` environment variable for
  researchers who keep their analysis outputs on a shared drive or
  lab cluster.
- Created lazily on first save — no directory clutter if the user
  never saves anything.

The naming ``workspace_io`` (vs just ``workspace``) is to keep this
module distinct from the ``workspace`` *page* in ``pages/workspace.py``
— that one is the scan-analysis surface; this one is on-disk file I/O.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)


def workspace_dir() -> Path:
    """Return the HRfunc workspace directory, creating it if missing.

    Honors the ``HRFUNC_WORKSPACE`` environment variable for researchers
    who keep their analysis outputs elsewhere; otherwise defaults to
    ``~/hrfunc_workspace/``.

    Created lazily on first call. Safe to call repeatedly; ``mkdir``
    with ``exist_ok=True`` is a no-op when the directory already
    exists.
    """
    override = os.environ.get("HRFUNC_WORKSPACE")
    if override:
        path = Path(override).expanduser()
    else:
        path = Path.home() / "hrfunc_workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_filename_fragment(name: str) -> str:
    """Sanitize a string for use as a filename fragment.

    Replaces any character that's not alphanumeric, dash, underscore, or
    dot with an underscore; collapses runs of underscores. Same idea as
    the icon-bundle filename sanitizer in ``hrfunc.cli.install_shortcut``.
    """
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    sanitized = re.sub(r"__+", "_", sanitized)
    return sanitized.strip("._") or "untitled"


def save_roi_average(
    *,
    anchor: Dict[str, Any],
    roi_keys: Iterable[str],
    hrf_mean: Any,
    hrf_std: Any,
    sfreq: float,
    radius_m: float,
    library_filter: Optional[Dict[str, Any]] = None,
    workspace: Optional[Path] = None,
) -> Path:
    """Save an ROI-averaged HRF to the workspace folder.

    The output JSON extends the standard HRF entry schema (so it can
    be re-loaded into ``hrfunc.tree`` if desired) with ROI provenance
    fields so a researcher can audit what went into the average:

    - ``hrf_mean`` / ``hrf_std`` — the averaged trace + per-sample std
    - ``sfreq`` — sampling rate (same as the anchor)
    - ``oxygenation`` — anchor's oxygenation (HbO or HbR)
    - ``location`` — anchor's location (the spatial center of the ROI)
    - ``context`` — extended with ROI metadata: anchor key, radius,
      member keys, library filter that was active

    File name: ``roi_<anchor_key_sanitized>_<YYYYMMDDTHHMMSSZ>.json``,
    placed in the workspace folder. Returns the absolute Path.

    Module-level so tests can call it without the GUI.
    """
    if workspace is None:
        workspace = workspace_dir()

    import numpy as np

    mean_list = np.asarray(hrf_mean).tolist()
    std_list = np.asarray(hrf_std).tolist()

    anchor_key = anchor.get("_key") or "unknown"
    payload: Dict[str, Any] = {
        "hrf_mean": mean_list,
        "hrf_std": std_list,
        "sfreq": float(sfreq),
        "oxygenation": bool(anchor.get("oxygenation")),
        "location": anchor.get("location"),
        "context": {
            **(anchor.get("context") or {}),
            "roi_average": True,
            "roi_anchor_key": anchor_key,
            "roi_radius_m": float(radius_m),
            "roi_member_keys": sorted(roi_keys),
            "roi_library_filter": dict(library_filter or {}),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    fragment = _safe_filename_fragment(anchor_key)
    out_path = workspace / f"roi_{fragment}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved ROI average to %s", out_path)
    return out_path
