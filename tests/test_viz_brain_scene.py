"""Unit tests for :mod:`hrfunc.viz.brain_scene`.

The :func:`make_surface_trace` helper is intentionally thin in PR #46
(it just wraps ``go.Mesh3d`` with our default hover / legend / lighting
conventions). These tests pin those defaults so changes to the rendering
behaviour are deliberate.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

from hrfunc.spatial.shapes import Box, Sphere  # noqa: E402
from hrfunc.viz.brain_scene import (  # noqa: E402
    make_box_overlay_trace,
    make_sphere_overlay_trace,
    make_surface_trace,
)


def _tiny_mesh():
    """A degenerate single-triangle mesh — enough to exercise plotly's
    Mesh3d constructor without depending on the bundled fsaverage assets."""
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    return verts, faces


class TestMakeSurfaceTrace:
    def test_returns_mesh3d(self):
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#abc", opacity=0.5, name="x"
        )
        assert trace.type == "mesh3d"

    def test_color_and_opacity_propagated(self):
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#c4b5a0", opacity=0.12, name="MNI head"
        )
        assert trace.color == "#c4b5a0"
        assert trace.opacity == 0.12
        assert trace.name == "MNI head"

    def test_hidden_from_legend_and_hover(self):
        """Anatomical surfaces should never steal hover events from the
        HRF scatter markers above them, and never clutter the legend."""
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#abc", opacity=0.3, name="x"
        )
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"

    def test_lighting_defaults(self):
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#abc", opacity=0.3, name="x"
        )
        assert trace.lighting.ambient == 0.5
        assert trace.lighting.diffuse == 0.6

    def test_lighting_overrides(self):
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#abc", opacity=0.3, name="x",
            ambient=0.6, diffuse=0.4,
        )
        assert trace.lighting.ambient == 0.6
        assert trace.lighting.diffuse == 0.4

    def test_vertex_coords_routed(self):
        verts, faces = _tiny_mesh()
        trace = make_surface_trace(
            verts, faces, color="#abc", opacity=0.3, name="x"
        )
        np.testing.assert_array_equal(trace.x, verts[:, 0])
        np.testing.assert_array_equal(trace.y, verts[:, 1])
        np.testing.assert_array_equal(trace.z, verts[:, 2])
        np.testing.assert_array_equal(trace.i, faces[:, 0])
        np.testing.assert_array_equal(trace.j, faces[:, 1])
        np.testing.assert_array_equal(trace.k, faces[:, 2])


class TestMakeBoxOverlayTrace:
    def _box(self):
        return Box(center_mm=(10.0, 20.0, 30.0), half_extents_mm=(5.0, 5.0, 5.0))

    def test_returns_mesh3d(self):
        trace = make_box_overlay_trace(self._box())
        assert trace.type == "mesh3d"

    def test_eight_vertices_twelve_faces(self):
        trace = make_box_overlay_trace(self._box())
        assert len(trace.x) == 8
        assert len(trace.i) == 12

    def test_vertex_positions_at_corners(self):
        """Verify every vertex sits at the box's centre +/- its half-extents."""
        trace = make_box_overlay_trace(self._box())
        for x, y, z in zip(trace.x, trace.y, trace.z):
            assert abs(x - 10.0) == 5.0
            assert abs(y - 20.0) == 5.0
            assert abs(z - 30.0) == 5.0

    def test_hidden_from_legend_and_hover(self):
        trace = make_box_overlay_trace(self._box())
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"

    def test_default_name(self):
        trace = make_box_overlay_trace(self._box())
        assert trace.name == "ROI box"


class TestMakeSphereOverlayTrace:
    def _sphere(self):
        return Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)

    def test_returns_mesh3d(self):
        trace = make_sphere_overlay_trace(self._sphere())
        assert trace.type == "mesh3d"

    def test_vertices_lie_on_sphere(self):
        """All UV-sphere verts must be within float epsilon of the radius."""
        trace = make_sphere_overlay_trace(self._sphere())
        # Skip plotly's tuple-of-tuples coords by routing through numpy.
        verts = np.column_stack([trace.x, trace.y, trace.z])
        distances = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(distances, 10.0, atol=1e-9)

    def test_vertex_count_scales_with_resolution(self):
        """Higher (n_lat, n_lon) -> more verts. Concretely: 2 poles plus
        (n_lat-1) latitude rings each with n_lon longitudes."""
        trace = make_sphere_overlay_trace(self._sphere(), n_lat=4, n_lon=6)
        # 2 poles + 3 rings * 6 longitudes = 20 verts.
        assert len(trace.x) == 20

    def test_offset_centre(self):
        sphere = Sphere(center_mm=(50.0, -20.0, 10.0), radius_mm=5.0)
        trace = make_sphere_overlay_trace(sphere)
        verts = np.column_stack([trace.x, trace.y, trace.z])
        diffs = verts - np.array([50.0, -20.0, 10.0])
        np.testing.assert_allclose(np.linalg.norm(diffs, axis=1), 5.0, atol=1e-9)

    def test_rejects_too_few_resolution(self):
        with pytest.raises(ValueError, match="n_lat>=2"):
            make_sphere_overlay_trace(self._sphere(), n_lat=1, n_lon=10)
        with pytest.raises(ValueError, match="n_lon>=3"):
            make_sphere_overlay_trace(self._sphere(), n_lat=10, n_lon=2)

    def test_hidden_from_legend_and_hover(self):
        trace = make_sphere_overlay_trace(self._sphere())
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"
