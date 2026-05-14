"""Tests for :class:`hrfunc.spatial.shapes.AtlasRegion`.

Most tests use the synthetic atlas from ``test_spatial_atlas`` so we
don't depend on the bundled labels.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrfunc.spatial.atlas import Atlas
from hrfunc.spatial.shapes import AtlasRegion


def _synthetic_atlas() -> Atlas:
    volume = np.zeros((3, 3, 3), dtype=np.int64)
    volume[:, 1, :] = 1   # y=1 slab -> Region_A
    volume[:, 2, :] = 2   # y=2 slab -> Region_B
    affine = np.array([
        [10.0, 0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0, 0.0],
        [0.0, 0.0, 10.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return Atlas(
        name="synthetic",
        volume=volume,
        affine=affine,
        labels=["Background", "Region_A", "Region_B"],
        background_label=0,
    )


class TestConstruction:
    def test_basic(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        assert region.region_name == "Region_A"
        assert region.atlas is atlas

    def test_rejects_none_atlas(self):
        with pytest.raises(ValueError, match="atlas"):
            AtlasRegion(None, "Region_A")  # type: ignore[arg-type]

    def test_rejects_empty_region_name(self):
        atlas = _synthetic_atlas()
        with pytest.raises(ValueError, match="region_name"):
            AtlasRegion(atlas, "")

    def test_rejects_unknown_region(self):
        atlas = _synthetic_atlas()
        with pytest.raises(ValueError, match="not present in atlas"):
            AtlasRegion(atlas, "Ghost")


class TestContains:
    def test_inside_region(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        # Voxel (1, 1, 1) -> (10, 10, 10) mm -> Region_A.
        assert region.contains((10.0, 10.0, 10.0)) is True

    def test_outside_region(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        # Voxel (0, 0, 0) -> origin -> background, not Region_A.
        assert region.contains((0.0, 0.0, 0.0)) is False

    def test_neighbouring_region_excluded(self):
        """A point in Region_B is not in Region_A even though it's
        spatially close to the Region_A slab."""
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        assert region.contains((0.0, 20.0, 0.0)) is False

    def test_out_of_volume(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        assert region.contains((1000.0, 0.0, 0.0)) is False


class TestContainsBatch:
    def test_batch_matches_per_point(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        points = np.array([
            [10.0, 10.0, 10.0],  # in
            [0.0, 20.0, 0.0],    # Region_B, not Region_A
            [0.0, 0.0, 0.0],     # background
            [1000.0, 0.0, 0.0],  # out of volume
        ])
        result = region.contains_batch(points)
        np.testing.assert_array_equal(result, [True, False, False, False])


class TestRepr:
    def test_includes_atlas_and_region(self):
        atlas = _synthetic_atlas()
        region = AtlasRegion(atlas, "Region_A")
        r = repr(region)
        assert "synthetic" in r
        assert "Region_A" in r
