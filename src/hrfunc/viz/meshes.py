"""Anatomical mesh loaders.

Bundled fsaverage surfaces ship in :mod:`hrfunc.assets` as decimated
NPZ files so no runtime fsaverage download is required. Two layers
are available:

- ``"pial"`` — fsaverage ``lh.pial`` + ``rh.pial`` stitched into a
  single cortical-surface mesh.
- ``"scalp"`` — fsaverage ``bem/outer_skin.surf``; the head's outer
  skin, where forehead/head-mounted fNIRS optodes physically sit.

Both meshes are pre-decimated to ~2.5k verts / 5k triangles and live
in MNI-meter coordinates so they overlay directly on bundled HRF
locations without any transform.

This module was extracted from ``hrfunc.gui.components.hrtree_panel``
during the v1.3 spatial/viz compartmentalization refactor. The GUI
panel re-exports the public functions so existing callers (and tests)
that import ``library.load_mesh`` keep working.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


MESH_FILENAMES: Dict[str, str] = {
    "pial": "fsaverage_pial_lowpoly.npz",
    "scalp": "fsaverage_scalp_lowpoly.npz",
}


# Module-level cache shared across the process. Each layer is loaded
# at most once; ``None`` is cached on load failure so repeated misses
# don't keep retrying the asset import.
MESH_CACHE: Dict[str, Optional[Tuple["np.ndarray", "np.ndarray"]]] = {}


def load_mesh(layer: str) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Return a bundled MNI anatomical mesh as ``(vertices, faces)``.

    Args:
        layer: ``"pial"`` for the cortical surface or ``"scalp"`` for
            the outer-skin head surface. Unknown layers return
            ``None`` (and the unknown name is cached so repeated calls
            don't keep warning).

    Returns:
        ``(verts, faces)`` as numpy arrays where ``verts`` is shape
        ``(N, 3)`` of MNI-meter coordinates and ``faces`` is shape
        ``(M, 3)`` of triangle vertex indices. Returns ``None`` if
        numpy can't be imported, the asset is missing, or the layer
        name is unknown — callers should fall back to no-overlay
        rendering rather than crashing.
    """
    if layer in MESH_CACHE:
        return MESH_CACHE[layer]
    filename = MESH_FILENAMES.get(layer)
    if filename is None:
        logger.warning("load_mesh: unknown layer %r", layer)
        MESH_CACHE[layer] = None
        return None
    try:
        import numpy as np
        from importlib import resources

        ref = resources.files("hrfunc.assets") / filename
        with resources.as_file(ref) as path:
            data = np.load(path)
            verts = data["vertices"]
            faces = data["faces"]
        MESH_CACHE[layer] = (verts, faces)
        return MESH_CACHE[layer]
    except Exception as exc:  # noqa: BLE001
        logger.warning("mesh load failed for layer=%r: %s", layer, exc)
        MESH_CACHE[layer] = None
        return None


def load_brain_mesh() -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Deprecated alias for ``load_mesh("scalp")``.

    Kept so external callers and the test suite that imported
    ``load_brain_mesh`` from the pre-dual-layer GUI module keep
    working. Defaults to the scalp layer because that's the new
    visible default. New code should call :func:`load_mesh` with an
    explicit layer name.
    """
    return load_mesh("scalp")
