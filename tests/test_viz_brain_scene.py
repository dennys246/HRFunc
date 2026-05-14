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

from hrfunc.viz.brain_scene import make_surface_trace  # noqa: E402


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
