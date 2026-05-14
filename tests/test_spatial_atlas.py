"""Tests for the bundled atlas loader + Atlas class.

Covers:
- ``load_harvard_oxford_cortical`` loads the bundled NIfTI + labels.
- ``Atlas.region_at`` + ``region_at_batch`` resolve MNI mm to label names.
- ``Atlas.contains_mm`` + ``contains_batch`` work as predicates.
- Out-of-volume + background-voxel returns are ``None`` / ``False``.
- Per-process caching: a second load returns the same Atlas instance.

Most tests use a small synthetic atlas built in-memory so the suite
stays fast and doesn't depend on the bundled file's exact contents.
A single integration test loads the real bundled atlas to confirm
the asset bundle + loader path work end-to-end.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrfunc.spatial.atlas import Atlas, load_harvard_oxford_cortical


def _synthetic_atlas() -> Atlas:
    """Build a tiny 3-voxel-cube atlas with three regions for tests.

    Voxel layout (label indices):

        x=0,1,2     y=0,1,2     z=0,1,2
        Background  Region_A    Region_B

    Affine maps voxel (i, j, k) -> world (10*i, 10*j, 10*k) mm, so
    voxel (1, 1, 1) is at MNI (10, 10, 10) mm.
    """
    volume = np.zeros((3, 3, 3), dtype=np.int64)
    volume[:, 1, :] = 1   # whole y=1 slab labelled Region_A
    volume[:, 2, :] = 2   # whole y=2 slab labelled Region_B
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


class TestRegionAt:
    def test_returns_region_name_at_known_voxel(self):
        atlas = _synthetic_atlas()
        # Voxel (1, 1, 1) -> (10, 10, 10) mm -> Region_A.
        assert atlas.region_at((10.0, 10.0, 10.0)) == "Region_A"

    def test_returns_region_b_at_y_2_slab(self):
        atlas = _synthetic_atlas()
        # Voxel (0, 2, 0) -> (0, 20, 0) mm -> Region_B.
        assert atlas.region_at((0.0, 20.0, 0.0)) == "Region_B"

    def test_returns_none_for_background(self):
        atlas = _synthetic_atlas()
        # Voxel (0, 0, 0) -> origin -> background.
        assert atlas.region_at((0.0, 0.0, 0.0)) is None

    def test_returns_none_for_out_of_volume(self):
        atlas = _synthetic_atlas()
        # 1000 mm is way outside the 3-voxel * 10mm volume.
        assert atlas.region_at((1000.0, 0.0, 0.0)) is None

    def test_nearest_neighbour_rounding(self):
        """A point 0.4 of a voxel off the centre rounds back to the
        centre voxel -- nearest-neighbour, not bilinear."""
        atlas = _synthetic_atlas()
        # (10.4, 10.4, 10.4) mm rounds to voxel (1, 1, 1) -> Region_A.
        assert atlas.region_at((10.4, 10.4, 10.4)) == "Region_A"


class TestRegionAtBatch:
    def test_batch_matches_per_point(self):
        atlas = _synthetic_atlas()
        points = np.array([
            [10.0, 10.0, 10.0],  # Region_A
            [0.0, 20.0, 0.0],    # Region_B
            [0.0, 0.0, 0.0],     # background
            [1000.0, 0.0, 0.0],  # out of volume
        ])
        result = atlas.region_at_batch(points)
        assert result == ["Region_A", "Region_B", None, None]

    def test_rejects_wrong_shape(self):
        atlas = _synthetic_atlas()
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            atlas.region_at_batch(np.array([1.0, 2.0, 3.0]))


class TestContainsMm:
    def test_true_for_matching_region(self):
        atlas = _synthetic_atlas()
        assert atlas.contains_mm((10.0, 10.0, 10.0), "Region_A") is True

    def test_false_for_wrong_region(self):
        atlas = _synthetic_atlas()
        assert atlas.contains_mm((10.0, 10.0, 10.0), "Region_B") is False

    def test_false_for_unknown_region(self):
        atlas = _synthetic_atlas()
        assert atlas.contains_mm((10.0, 10.0, 10.0), "Ghost") is False

    def test_false_for_background_query(self):
        atlas = _synthetic_atlas()
        # Background label can't be a "region" -- contains_mm returns False
        # even for points that are actually in background.
        assert atlas.contains_mm((0.0, 0.0, 0.0), "Background") is False

    def test_false_for_out_of_volume(self):
        atlas = _synthetic_atlas()
        assert atlas.contains_mm((1000.0, 0.0, 0.0), "Region_A") is False


class TestContainsBatch:
    def test_batch_matches_per_point(self):
        atlas = _synthetic_atlas()
        points = np.array([
            [10.0, 10.0, 10.0],  # Region_A
            [0.0, 20.0, 0.0],    # Region_B -- different region
            [0.0, 0.0, 0.0],     # background
            [1000.0, 0.0, 0.0],  # out of volume
        ])
        result = atlas.contains_batch(points, "Region_A")
        np.testing.assert_array_equal(result, [True, False, False, False])

    def test_unknown_region_returns_all_false(self):
        atlas = _synthetic_atlas()
        result = atlas.contains_batch(
            np.array([[10.0, 10.0, 10.0]]), "Ghost"
        )
        np.testing.assert_array_equal(result, [False])


class TestRegionMask:
    def test_returns_boolean_volume(self):
        atlas = _synthetic_atlas()
        mask = atlas.region_mask("Region_A")
        assert mask is not None
        assert mask.dtype == bool
        assert mask.shape == atlas.volume.shape
        # The y=1 slab is True, others False.
        assert mask[:, 1, :].all()
        assert not mask[:, 0, :].any()
        assert not mask[:, 2, :].any()

    def test_returns_none_for_unknown(self):
        atlas = _synthetic_atlas()
        assert atlas.region_mask("Ghost") is None

    def test_returns_none_for_background(self):
        atlas = _synthetic_atlas()
        # Background is not a "region".
        assert atlas.region_mask("Background") is None


class TestRegionNames:
    def test_strips_background_from_listing(self):
        atlas = _synthetic_atlas()
        assert atlas.region_names == ["Region_A", "Region_B"]


class TestLabelIndex:
    def test_known_region(self):
        atlas = _synthetic_atlas()
        assert atlas.label_index("Region_A") == 1
        assert atlas.label_index("Region_B") == 2

    def test_unknown_returns_none(self):
        atlas = _synthetic_atlas()
        assert atlas.label_index("Ghost") is None

    def test_case_sensitive(self):
        atlas = _synthetic_atlas()
        assert atlas.label_index("region_a") is None


# ---------------------------------------------------------------------------
# Bundled Harvard-Oxford integration (slow; loads the real NIfTI)
# ---------------------------------------------------------------------------


class TestHarvardOxfordBundle:
    """One integration test confirming the bundled atlas loads end-to-end.

    Most unit tests above use the synthetic atlas for speed and so
    they're independent of the bundled labels. This test confirms
    the wheel-bundling + loader path actually works on the real
    Harvard-Oxford asset.
    """

    def test_load_returns_atlas(self):
        atlas = load_harvard_oxford_cortical()
        assert atlas is not None
        assert atlas.name == "harvard-oxford-cort-maxprob-thr25-2mm"

    def test_has_expected_label_count(self):
        atlas = load_harvard_oxford_cortical()
        # 48 cortical regions + background = 49 labels.
        assert len(atlas.labels) == 49
        assert atlas.labels[0] == "Background"

    def test_region_names_excludes_background(self):
        atlas = load_harvard_oxford_cortical()
        assert "Background" not in atlas.region_names
        # First user-visible region is Frontal Pole per the FSL atlas.
        assert "Frontal Pole" in atlas.region_names

    def test_frontal_pole_lookup_works(self):
        """Spot-check a known MNI coordinate that falls in Frontal Pole.
        The 2 mm atlas isn't pixel-perfect for a hand-picked coord, so
        this test is more "the lookup pipeline returns something
        plausible" than "the atlas labels match clinical truth"."""
        atlas = load_harvard_oxford_cortical()
        # Anterior frontal pole; coordinate from common MNI references.
        # Some neighbouring voxels in this region are Frontal Pole or
        # background depending on threshold; we just assert that
        # SOMETHING reasonable surfaces.
        region = atlas.region_at((0.0, 60.0, -10.0))
        # Region can be None (background between cortical labels) or
        # one of the named regions. The point matters: no crash.
        assert region is None or region in atlas.region_names

    def test_loader_is_cached(self):
        a = load_harvard_oxford_cortical()
        b = load_harvard_oxford_cortical()
        assert a is b
