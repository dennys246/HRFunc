"""Modality-agnostic 3D visualisation primitives.

Companion package to :mod:`hrfunc.spatial`. Where ``spatial`` owns
the data and geometry, ``viz`` owns the rendering: anatomical surface
loaders (fsaverage today, user-supplied NIfTI in v1.3.1) and the
plotly-trace builders that compose them into a brain scene.

Like ``spatial``, nothing here knows about HbO/HbR or any other
fNIRS-specific concept. The future fMRI HRF pipeline will share this
rendering layer unchanged — point clouds, shape overlays, and
anatomical surfaces are all modality-agnostic.

Public surface:

- :func:`load_mesh` / :func:`load_brain_mesh` — fsaverage anatomical
  surface loader (pial cortical + outer-skin scalp), bundled with
  the wheel as decimated NPZ assets. ``load_brain_mesh`` is the
  back-compat alias for callers that pre-date the dual-layer split.
- :func:`make_surface_trace` — build a plotly ``Mesh3d`` trace from
  ``(vertices, faces)`` arrays. Used by the HRtree panel to render
  fsaverage surfaces today; will also be used by the v1.3.1
  anatomical-NIfTI viewer.
"""

from .brain_scene import make_surface_trace
from .meshes import (
    MESH_CACHE,
    MESH_FILENAMES,
    load_brain_mesh,
    load_mesh,
)

__all__ = [
    "MESH_CACHE",
    "MESH_FILENAMES",
    "load_brain_mesh",
    "load_mesh",
    "make_surface_trace",
]
