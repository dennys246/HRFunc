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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Schema version for the montage.json file. Bump when the wrapper or
# per-ROI block changes shape in a way readers need to detect. ROIs
# inside the list keep the per-ROI single-file schema that pre-PR-#55
# audit scripts already understand (anchor / location / hrf_mean /
# context.roi_*), so most consumers can skim past the wrapper.
MONTAGE_SCHEMA_VERSION = "1.3"


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


def build_roi_entry(
    *,
    roi_keys: Iterable[str],
    hrf_mean: Any,
    hrf_std: Any,
    sfreq: float,
    anchor: Optional[Dict[str, Any]] = None,
    shape: Optional[Any] = None,
    radius_m: Optional[float] = None,
    library_filter: Optional[Dict[str, Any]] = None,
    oxygenation_filter: Optional[bool] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the per-ROI block of a montage.json file.

    Pure -- no I/O, no clocks except for the ``saved_at`` timestamp
    that's stamped at construction time. Same content as the pre-PR-#55
    single-file ``roi_*.json`` payload (anchor / location / hrf_mean /
    context.roi_*), plus an optional ``name`` field for the per-ROI
    display label in the multi-ROI list.

    Extracted from ``save_roi_average`` so ``save_montage`` can build
    each ROI's block via the same path. Keeping it pure also lets
    tests inspect the entry shape without exercising disk I/O.
    """
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

    entry: Dict[str, Any] = {
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
    if name:
        entry["name"] = name
    return entry


def _entry_filename_fragment(entry: Dict[str, Any]) -> str:
    """Pick a recognisable filename fragment for a single ROI entry.

    Preference order matches the pre-PR-#55 ``save_roi_average``
    convention: anchor key first, then a ``freefloat_<shape>`` stand-
    in derived from the shape descriptor, then ``freefloat_unknown``.
    """
    anchor_key = entry.get("context", {}).get("roi_anchor_key")
    if anchor_key:
        return _safe_filename_fragment(anchor_key)
    shape_desc = entry.get("context", {}).get("roi_shape") or {}
    shape_type = shape_desc.get("type") if isinstance(shape_desc, dict) else None
    if shape_type:
        return _safe_filename_fragment(f"freefloat_{shape_type}")
    return "freefloat_unknown"


def _build_alignment_block(
    alignment_offset_mm: Optional[Tuple[float, float, float]],
    alignment_affine: Optional[Any],
) -> Dict[str, Any]:
    """Build the wrapper's ``alignment`` block.

    Alignment is a property of the HRF library's coord frame, not of
    any individual ROI (locked decision 2026-05-14 -- see the
    v1-3-spatial-viz-compartmentalization memory). The wrapper carries
    it once for the whole montage.

    ``offset_mm`` is always present (zeros when unset) so readers
    don't have to special-case its absence. ``affine`` is null when
    the user hasn't loaded one.
    """
    import numpy as np

    offset = alignment_offset_mm or (0.0, 0.0, 0.0)
    if alignment_affine is None:
        affine_block: Optional[List[List[float]]] = None
    else:
        affine_block = np.asarray(alignment_affine, dtype=np.float64).tolist()
    return {
        "offset_mm": [float(v) for v in offset],
        "affine": affine_block,
    }


def save_montage(
    *,
    rois: Sequence[Dict[str, Any]],
    alignment_offset_mm: Optional[Tuple[float, float, float]] = None,
    alignment_affine: Optional[Any] = None,
    workspace: Optional[Path] = None,
) -> Path:
    """Write a multi-ROI montage to the workspace as a single JSON.

    Schema:

    .. code-block:: json

        {
          "version": "1.3",
          "alignment": {
            "offset_mm": [dx, dy, dz],
            "affine": [[...], ...] | null
          },
          "rois": [<per-ROI entry>, ...]
        }

    Each per-ROI entry is the pre-PR-#55 single-file payload (anchor
    metadata + ROI provenance under ``context.roi_*``). Build entries
    with :func:`build_roi_entry`. Caller is responsible for excluding
    ROIs that don't average (the canonical guard is
    ``compute_roi_average() is None``).

    Filename: ``montage_<descriptor>_<YYYYMMDDTHHMMSSZ>.json``. The
    descriptor is the first ROI's anchor key (or shape stand-in) so
    single-ROI montages keep the recognisable filename shape of the
    legacy single-file save; multi-ROI montages get the first ROI's
    fragment plus a ``_plus<N-1>`` suffix.
    """
    if not rois:
        raise ValueError("save_montage requires at least one ROI entry")

    if workspace is None:
        workspace = workspace_dir()

    payload: Dict[str, Any] = {
        "version": MONTAGE_SCHEMA_VERSION,
        "alignment": _build_alignment_block(
            alignment_offset_mm, alignment_affine
        ),
        "rois": list(rois),
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head_fragment = _entry_filename_fragment(rois[0])
    if len(rois) > 1:
        head_fragment = f"{head_fragment}_plus{len(rois) - 1}"
    out_path = workspace / f"montage_{head_fragment}_{timestamp}.json"

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("Saved montage (%d ROIs) to %s", len(rois), out_path)
    return out_path


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
    name: Optional[str] = None,
    # PR #55: alignment is now a wrapper-level property of the
    # montage, not of any single ROI. ``save_roi_average`` accepts it
    # for callers that want to write a single-ROI montage with the
    # alignment already known; defaults to identity / zeros so existing
    # callers keep producing alignment=None / zero-offset montages.
    alignment_offset_mm: Optional[Tuple[float, float, float]] = None,
    alignment_affine: Optional[Any] = None,
    workspace: Optional[Path] = None,
) -> Path:
    """Save a single-ROI average to the workspace as a 1-ROI montage.

    PR #55 unified the on-disk format: every save -- single ROI or
    multi-ROI montage -- writes the same wrapper schema with a list
    of ROI entries. ``save_roi_average`` builds a 1-entry list and
    calls :func:`save_montage`. The pre-PR-#55 ``roi_<key>_<ts>.json``
    filename is replaced by ``montage_<key>_<ts>.json`` (same
    descriptive fragment, new prefix) and the trace data moves under
    ``rois[0]`` instead of sitting at the top level.

    Use this when you have a single ROI's averaged trace already
    computed (e.g. the legacy single-save flow). Use
    :func:`save_montage` directly when you have multiple pre-built
    ROI entries to write to one file (the multi-ROI list flow).

    Module-level so tests can call it without the GUI.
    """
    entry = build_roi_entry(
        roi_keys=roi_keys,
        hrf_mean=hrf_mean,
        hrf_std=hrf_std,
        sfreq=sfreq,
        anchor=anchor,
        shape=shape,
        radius_m=radius_m,
        library_filter=library_filter,
        oxygenation_filter=oxygenation_filter,
        name=name,
    )
    return save_montage(
        rois=[entry],
        alignment_offset_mm=alignment_offset_mm,
        alignment_affine=alignment_affine,
        workspace=workspace,
    )
