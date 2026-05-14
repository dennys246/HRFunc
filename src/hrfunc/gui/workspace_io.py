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
    roi_keys: Iterable[str],
    hrf_mean: Any,
    hrf_std: Any,
    sfreq: float,
    # Anchor is optional in PR #49 to support free-floating shape ROIs.
    # When a click-anchor IS set, its location / oxygenation / context
    # ride into the saved JSON so the file matches the v1.2 schema and
    # can still be re-loaded into ``hrfunc.tree``. When the anchor is
    # None, the ROI's shape centre stands in as the saved location and
    # the saved oxygenation comes from ``oxygenation_filter``.
    anchor: Optional[Dict[str, Any]] = None,
    # Spatial-layer Shape describing the ROI extent. Defaults to None
    # to keep the v1.2 anchor + radius_m call sites compatible.
    shape: Optional[Any] = None,
    # Pre-PR-#49 legacy field. When the new ``shape`` is provided this
    # is ignored for geometry but still recorded in the JSON as the
    # ``roi_radius_m`` context key (v1.2 audit scripts read it).
    radius_m: Optional[float] = None,
    library_filter: Optional[Dict[str, Any]] = None,
    # Oxygenation that the membership computation filtered on (True for
    # HbO, False for HbR, None for mixed). Used to set the saved
    # ``oxygenation`` field when there's no anchor.
    oxygenation_filter: Optional[bool] = None,
    workspace: Optional[Path] = None,
) -> Path:
    """Save an ROI-averaged HRF to the workspace folder.

    The output JSON extends the standard HRF entry schema (so it can
    be re-loaded into ``hrfunc.tree`` if desired) with ROI provenance
    fields so a researcher can audit what went into the average:

    - ``hrf_mean`` / ``hrf_std`` -- the averaged trace + per-sample std
    - ``sfreq`` -- sampling rate
    - ``oxygenation`` -- anchor's oxygenation when present; otherwise
      the explicit ``oxygenation_filter``; falls back to ``None``
      when neither is available.
    - ``location`` -- anchor's location when present; otherwise the
      shape's centre (converted from MNI mm into MNE-meter coords so
      the saved JSON round-trips back into ``hrfunc.tree`` without
      unit fixups).
    - ``context`` -- extended with ROI metadata: anchor key (or
      ``None`` for free-floating), shape descriptor, member keys,
      library filter that was active.

    File name: ``roi_<anchor_or_shape_key>_<YYYYMMDDTHHMMSSZ>.json``,
    placed in the workspace folder. Returns the absolute Path.

    Module-level so tests can call it without the GUI.
    """
    if workspace is None:
        workspace = workspace_dir()

    import numpy as np

    mean_list = np.asarray(hrf_mean).tolist()
    std_list = np.asarray(hrf_std).tolist()

    anchor_key: Optional[str] = (
        anchor.get("_key") if anchor is not None else None
    ) or None

    # Compute the saved location (in MNE meters, matching the v1.2
    # schema so the JSON round-trips into ``hrfunc.tree.load_hrfs``):
    #
    # 1. With anchor: anchor.location wins (preserves v1.2 behaviour).
    # 2. Without anchor, shape has ``center_mm``: convert mm -> meters.
    # 3. Without anchor, shape is an :class:`AtlasRegion`: compute the
    #    region's voxel centroid in MNI mm via the atlas affine, then
    #    convert to meters. Gives users a sensible "location" field
    #    for region ROIs that lack a clicked-anchor.
    # 4. Otherwise: None (saved file won't round-trip into tree).
    if anchor is not None:
        location: Optional[list] = anchor.get("location")
    elif shape is not None and hasattr(shape, "center_mm"):
        cx, cy, cz = shape.center_mm
        location = [cx / 1000.0, cy / 1000.0, cz / 1000.0]
    elif shape is not None and hasattr(shape, "region_name") and hasattr(shape, "atlas"):
        try:
            mask = shape.atlas.region_mask(shape.region_name)
            if mask is not None and mask.any():
                voxel_indices = np.argwhere(mask)
                centroid_voxel = voxel_indices.mean(axis=0)
                homo = np.append(centroid_voxel, 1.0)
                centroid_mm = (shape.atlas.affine @ homo)[:3]
                location = (centroid_mm / 1000.0).tolist()
            else:
                location = None
        except Exception:  # noqa: BLE001
            location = None
    else:
        location = None

    # Resolve oxygenation: anchor wins, then explicit filter, then None.
    if anchor is not None and anchor.get("oxygenation") is not None:
        oxygenation: Optional[bool] = bool(anchor.get("oxygenation"))
    elif oxygenation_filter is not None:
        oxygenation = bool(oxygenation_filter)
    else:
        oxygenation = None

    # Shape descriptor for the saved JSON. Captures enough that a
    # downstream reader can reconstruct the geometry exactly.
    #
    # PR #52 extended the box descriptor with an optional
    # ``orientation_mm`` field. To keep the schema clean for the
    # common (axis-aligned) case we only emit the orientation when
    # the box's matrix differs from identity -- readers that pre-date
    # PR #52 simply don't see the new field and treat the box as
    # axis-aligned, which matches their assumption.
    shape_descriptor: Optional[Dict[str, Any]] = None
    if shape is not None:
        if hasattr(shape, "half_extents_mm"):  # Box
            shape_descriptor = {
                "type": "box",
                "center_mm": list(shape.center_mm),
                "half_extents_mm": list(shape.half_extents_mm),
            }
            is_aligned = getattr(shape, "is_axis_aligned", None)
            if callable(is_aligned) and not is_aligned():
                # Non-identity orientation: include the rotation matrix
                # as a nested list so it round-trips through json.
                shape_descriptor["orientation_mm"] = (
                    np.asarray(shape.orientation_mm).tolist()
                )
        elif hasattr(shape, "region_name"):  # AtlasRegion
            # PR #53: atlas-defined region. Capture the atlas name +
            # region label so readers can reconstruct the same ROI
            # against the bundled (or a compatible) atlas.
            atlas_name = (
                getattr(shape.atlas, "name", None)
                if hasattr(shape, "atlas")
                else None
            )
            shape_descriptor = {
                "type": "atlas_region",
                "atlas": atlas_name,
                "region_name": shape.region_name,
            }
        elif hasattr(shape, "radius_mm"):  # Sphere
            shape_descriptor = {
                "type": "sphere",
                "center_mm": list(shape.center_mm),
                "radius_mm": float(shape.radius_mm),
            }
    elif radius_m is not None:
        # Legacy v1.2 call site (no shape passed). Record the radius as
        # a sphere descriptor for forward-compat with v1.3+ readers.
        shape_descriptor = {
            "type": "sphere",
            "radius_m": float(radius_m),
        }

    context_extra: Dict[str, Any] = {
        "roi_average": True,
        "roi_anchor_key": anchor_key,
        "roi_shape": shape_descriptor,
        "roi_member_keys": sorted(roi_keys),
        "roi_library_filter": dict(library_filter or {}),
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    # Preserve the v1.2 ``roi_radius_m`` key when the call site is the
    # legacy radius-based path, or when the new path uses a sphere.
    # v1.2 audit scripts grep for this exact key; emit it whenever
    # possible to keep them working.
    if radius_m is not None:
        context_extra["roi_radius_m"] = float(radius_m)
    elif (
        shape is not None
        and hasattr(shape, "radius_mm")
        and not hasattr(shape, "half_extents_mm")
    ):
        context_extra["roi_radius_m"] = float(shape.radius_mm) / 1000.0

    payload: Dict[str, Any] = {
        "hrf_mean": mean_list,
        "hrf_std": std_list,
        "sfreq": float(sfreq),
        "oxygenation": oxygenation,
        "location": location,
        "context": {
            **((anchor or {}).get("context") or {}),
            **context_extra,
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # Filename fragment: anchor key first (v1.2 convention), else a
    # shape-derived stand-in so free-floating saves still have a
    # human-recognisable prefix.
    if anchor_key:
        fragment = _safe_filename_fragment(anchor_key)
    elif shape_descriptor is not None:
        fragment = _safe_filename_fragment(
            f"freefloat_{shape_descriptor.get('type', 'shape')}"
        )
    else:
        fragment = "freefloat_unknown"
    out_path = workspace / f"roi_{fragment}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved ROI average to %s", out_path)
    return out_path
