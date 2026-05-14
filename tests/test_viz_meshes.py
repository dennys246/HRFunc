"""Unit tests for :mod:`hrfunc.viz.meshes`.

These tests exercise the mesh-loader's new home in ``hrfunc.viz.meshes``.
The legacy v1.3.0 tests in ``test_gui_library.py::TestMeshLoader`` still
exercise the same loader via the back-compat alias on ``hrtree_panel``;
both test paths must agree because the panel re-exports the names.
"""

from __future__ import annotations

from hrfunc.viz.meshes import (
    MESH_CACHE,
    MESH_FILENAMES,
    load_brain_mesh,
    load_mesh,
)


class TestKnownLayers:
    def test_pial_returns_arrays(self):
        MESH_CACHE.clear()
        result = load_mesh("pial")
        assert result is not None
        verts, faces = result
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert 1_000 < verts.shape[0] < 50_000

    def test_scalp_returns_arrays(self):
        MESH_CACHE.clear()
        result = load_mesh("scalp")
        assert result is not None
        verts, faces = result
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert 1_000 < verts.shape[0] < 50_000

    def test_layers_in_mni_meter_scale(self):
        """Same defensive bound that catches the 360-meter globals bug —
        a future mm-scale rebuild would blow plotly's aspectmode=data."""
        MESH_CACHE.clear()
        for layer in ("pial", "scalp"):
            verts, _ = load_mesh(layer)
            assert float(abs(verts).max()) < 1.0


class TestUnknownLayer:
    def test_returns_none(self):
        MESH_CACHE.clear()
        assert load_mesh("not-a-real-layer") is None


class TestCaching:
    def test_per_layer_caching(self):
        MESH_CACHE.clear()
        a = load_mesh("pial")
        b = load_mesh("pial")
        assert a is b

    def test_unknown_layer_cached_too(self):
        """Repeated misses shouldn't keep re-warning."""
        MESH_CACHE.clear()
        load_mesh("ghost")
        assert "ghost" in MESH_CACHE
        assert MESH_CACHE["ghost"] is None


class TestBackCompatAlias:
    def test_load_brain_mesh_is_scalp(self):
        MESH_CACHE.clear()
        result = load_brain_mesh()
        scalp = load_mesh("scalp")
        assert result is scalp


class TestMeshFilenamesDict:
    def test_known_layers_listed(self):
        assert "pial" in MESH_FILENAMES
        assert "scalp" in MESH_FILENAMES
