"""Plotly trace builders for anatomical surfaces and ROI shape overlays.

Exposes three trace builders consumed by the HRtree panel:

- :func:`make_surface_trace` — the original anatomical-surface
  builder. Used for the bundled fsaverage scalp / pial overlays;
  will also be used by the v1.3.1 anatomical-NIfTI viewer once
  user-supplied surfaces enter the pipeline. The brain-scene layer
  doesn't care whether the verts came from a bundled NPZ or a
  marching-cubes pass over a user image.
- :func:`make_box_overlay_trace` — translucent cuboid for the
  Cluster sub-tab's box-mode ROI shape.
- :func:`make_sphere_overlay_trace` — translucent UV-sphere for
  the Cluster sub-tab's sphere-mode ROI shape.

All three return ``plotly.graph_objects.Mesh3d`` traces configured
with ``hoverinfo="skip"`` and ``showlegend=False`` so the overlays
don't steal hover events from the HRF scatter markers above them
or clutter the legend.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..spatial.shapes import Box, Sphere

# Standard cuboid triangulation indices. Each of the 6 quad faces is
# split into 2 triangles; 12 triangles total. The vertex order matches
# :meth:`hrfunc.spatial.shapes.Box.corners_mm` (binary sign-flip count),
# so face indices reference the same corner each time.
#
#       6 +-----------+ 7
#        /|          /|
#       / |         / |
#    4 +-----------+ 5|
#      |  |        |  |
#      |  + 2 -----|--+ 3
#      | /         | /
#      |/          |/
#    0 +-----------+ 1
#
# Corner indexing: bit 0 -> +x, bit 1 -> +y, bit 2 -> +z.
_BOX_FACES = np.array(
    [
        # -z face (corners 0,1,2,3)
        [0, 1, 3], [0, 3, 2],
        # +z face (corners 4,5,6,7)
        [4, 6, 7], [4, 7, 5],
        # -y face (corners 0,1,4,5)
        [0, 4, 5], [0, 5, 1],
        # +y face (corners 2,3,6,7)
        [2, 3, 7], [2, 7, 6],
        # -x face (corners 0,2,4,6)
        [0, 2, 6], [0, 6, 4],
        # +x face (corners 1,3,5,7)
        [1, 5, 7], [1, 7, 3],
    ],
    dtype=np.int64,
)


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


def make_box_overlay_trace(
    box: Box,
    *,
    color: str = "#a78bfa",
    opacity: float = 0.18,
    name: str = "ROI box",
    ambient: float = 0.55,
    diffuse: float = 0.55,
) -> Any:
    """Build a plotly ``Mesh3d`` cuboid for an axis-aligned ROI box.

    The box's :meth:`~hrfunc.spatial.shapes.Box.corners_mm` is used
    directly as the vertex array; the 12-triangle face table is the
    module-level constant ``_BOX_FACES``. Default colour is a soft
    violet that reads on both bright and dark fsaverage surfaces and
    doesn't collide with the HbO red / HbR blue / ROI gold of the
    existing scatter palette.
    """
    import plotly.graph_objects as go

    verts = box.corners_mm()
    return go.Mesh3d(
        x=verts[:, 0],
        y=verts[:, 1],
        z=verts[:, 2],
        i=_BOX_FACES[:, 0],
        j=_BOX_FACES[:, 1],
        k=_BOX_FACES[:, 2],
        color=color,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
        showlegend=False,
        lighting=dict(ambient=ambient, diffuse=diffuse),
        flatshading=True,
    )


def _uv_sphere_mesh(
    centre: np.ndarray,
    radius: float,
    n_lat: int,
    n_lon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """UV-sphere triangulation: ``(verts, faces)`` for a sphere mesh.

    Latitude rings are stacked along the z axis. ``n_lat`` controls
    the number of latitude bands (more = smoother poles); ``n_lon``
    controls longitude segments (more = smoother equator). The two
    poles are shared single vertices so the triangulation closes
    cleanly without gaps.
    """
    if n_lat < 2 or n_lon < 3:
        raise ValueError(
            f"UV sphere requires n_lat>=2, n_lon>=3; got {n_lat}, {n_lon}"
        )

    verts: list[list[float]] = []
    # North pole.
    verts.append([centre[0], centre[1], centre[2] + radius])

    # Intermediate latitude rings (exclusive of poles).
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat  # 0 at north, pi at south
        z = radius * np.cos(theta)
        ring_r = radius * np.sin(theta)
        for j in range(n_lon):
            phi = 2.0 * np.pi * j / n_lon
            x = ring_r * np.cos(phi)
            y = ring_r * np.sin(phi)
            verts.append([centre[0] + x, centre[1] + y, centre[2] + z])

    # South pole.
    verts.append([centre[0], centre[1], centre[2] - radius])

    faces: list[list[int]] = []
    north = 0
    south = 1 + (n_lat - 1) * n_lon

    # Top cap: triangles between the north pole and the first ring.
    for j in range(n_lon):
        a = 1 + j
        b = 1 + (j + 1) % n_lon
        faces.append([north, a, b])

    # Middle bands: each quad split into two triangles.
    for i in range(n_lat - 2):
        ring0 = 1 + i * n_lon
        ring1 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            a = ring0 + j
            b = ring0 + (j + 1) % n_lon
            c = ring1 + j
            d = ring1 + (j + 1) % n_lon
            faces.append([a, c, d])
            faces.append([a, d, b])

    # Bottom cap: triangles between the last ring and the south pole.
    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        a = last_ring + j
        b = last_ring + (j + 1) % n_lon
        faces.append([south, b, a])

    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def make_sphere_overlay_trace(
    sphere: Sphere,
    *,
    color: str = "#a78bfa",
    opacity: float = 0.18,
    name: str = "ROI sphere",
    n_lat: int = 12,
    n_lon: int = 18,
    ambient: float = 0.55,
    diffuse: float = 0.55,
) -> Any:
    """Build a plotly ``Mesh3d`` UV-sphere for a sphere-mode ROI overlay.

    The UV-sphere triangulation gives a clean visual sphere without
    bringing in scipy/skimage just for icosphere subdivision. Default
    ``n_lat=12, n_lon=18`` is ~200 verts / ~400 triangles -- well
    inside plotly's interactive comfort zone alongside the fsaverage
    overlays (~2.5k verts each). The default colour matches the
    box overlay so users get visual consistency when switching
    shape mode.
    """
    import plotly.graph_objects as go

    centre = np.asarray(sphere.center_mm, dtype=np.float64)
    verts, faces = _uv_sphere_mesh(centre, sphere.radius_mm, n_lat, n_lon)
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
