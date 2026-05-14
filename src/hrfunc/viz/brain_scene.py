"""Plotly trace builders for anatomical surfaces.

Today this module exposes :func:`make_surface_trace`, the helper used
by the HRtree panel to render the fsaverage scalp + pial overlays.
The same helper will be re-used by the v1.3.1 anatomical-NIfTI viewer
once user-supplied surfaces enter the pipeline — the brain-scene
layer doesn't care whether the verts came from a bundled fsaverage
NPZ or a marching-cubes pass over a user image.

Kept deliberately thin in PR #46. As shape overlays (PR #47) and
user anatomicals (PR #48) land, this module will grow a
``BrainScene`` composer that owns ``add_surface`` / ``add_points``
/ ``add_shape`` builder methods. For now, the figure assembly stays
in the panel and only the per-trace construction lives here.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def make_surface_trace(
    verts: np.ndarray,
    faces: np.ndarray,
    *,
    color: str,
    opacity: float,
    name: str,
    ambient: float = 0.5,
    diffuse: float = 0.6,
) -> Any:
    """Build a plotly ``Mesh3d`` trace for a 3D anatomical surface.

    Args:
        verts: ``(N, 3)`` array of vertex coordinates in the scene's
            spatial units (MNI meters for the fsaverage bundle).
        faces: ``(M, 3)`` array of triangle vertex indices.
        color: CSS-style colour string for the surface.
        opacity: Surface opacity in ``[0, 1]``.
        name: Trace name; appears in plotly's hoverable trace list
            even though we hide it from the legend.
        ambient / diffuse: Plotly ``lighting`` parameters. The
            fsaverage scalp uses a higher ambient (skin-tone reads
            warmer); the brain mesh uses higher diffuse (so the
            sulcal geometry is more legible).

    The trace is configured with ``hoverinfo="skip"`` and
    ``showlegend=False`` so anatomical surfaces don't clutter the
    legend or steal hover events from the HRF point markers above
    them.
    """
    import plotly.graph_objects as go

    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        showlegend=False,
        lighting=dict(ambient=ambient, diffuse=diffuse),
    )
