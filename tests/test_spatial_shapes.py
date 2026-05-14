"""Unit tests for :mod:`hrfunc.spatial.shapes`.

The :class:`Shape` API is the foundation of the upcoming ROI shape
selector (sphere now, box + atlas-region in PRs #47 and #48). These
tests pin down the contains-semantics so the new shapes in those
PRs can be added with the same predicate contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrfunc.spatial.shapes import Box, Shape, Sphere


class TestSphereConstruction:
    def test_basic_construction(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        assert sphere.center_mm == (0.0, 0.0, 0.0)
        assert sphere.radius_mm == 10.0

    def test_list_center_accepted(self):
        sphere = Sphere(center_mm=[1.0, 2.0, 3.0], radius_mm=5.0)
        assert sphere.center_mm == (1.0, 2.0, 3.0)

    def test_rejects_wrong_dimensions(self):
        with pytest.raises(ValueError, match="3 elements"):
            Sphere(center_mm=(0.0, 0.0), radius_mm=5.0)

    def test_rejects_negative_radius(self):
        with pytest.raises(ValueError, match="non-negative"):
            Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=-1.0)

    def test_zero_radius_is_allowed(self):
        """A degenerate point-sphere: only the centre is inside."""
        sphere = Sphere(center_mm=(5.0, 5.0, 5.0), radius_mm=0.0)
        assert sphere.contains((5.0, 5.0, 5.0)) is True
        assert sphere.contains((5.0, 5.0, 5.001)) is False


class TestSphereContains:
    def test_inside_point(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        assert sphere.contains((1.0, 2.0, 3.0)) is True

    def test_outside_point(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        assert sphere.contains((100.0, 0.0, 0.0)) is False

    def test_boundary_point_is_inside(self):
        """Membership is closed (``distance <= radius``) — matches the
        pre-refactor ROI-radius semantics in the HRtree panel."""
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        assert sphere.contains((10.0, 0.0, 0.0)) is True

    def test_offset_centre(self):
        sphere = Sphere(center_mm=(30.0, 20.0, 10.0), radius_mm=5.0)
        assert sphere.contains((30.0, 20.0, 10.0)) is True
        assert sphere.contains((34.0, 20.0, 10.0)) is True   # 4 mm away
        assert sphere.contains((36.0, 20.0, 10.0)) is False  # 6 mm away

    def test_accepts_tuple_list_ndarray(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        assert sphere.contains((1.0, 2.0, 3.0)) is True
        assert sphere.contains([1.0, 2.0, 3.0]) is True
        assert sphere.contains(np.array([1.0, 2.0, 3.0])) is True


class TestSphereBatch:
    def test_basic_batch(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        points = np.array([
            [0.0, 0.0, 0.0],   # inside
            [10.0, 0.0, 0.0],  # boundary — inside (closed)
            [11.0, 0.0, 0.0],  # outside
            [3.0, 4.0, 0.0],   # 5 mm away — inside
        ])
        result = sphere.contains_batch(points)
        np.testing.assert_array_equal(result, [True, True, False, True])

    def test_batch_matches_per_point(self):
        """Vectorised must agree with per-point loop. Catches einsum slip-ups."""
        rng = np.random.default_rng(seed=42)
        points = rng.uniform(-50, 50, size=(100, 3))
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=20.0)
        batch = sphere.contains_batch(points)
        per_point = np.array([sphere.contains(p) for p in points])
        np.testing.assert_array_equal(batch, per_point)

    def test_batch_rejects_wrong_shape(self):
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            sphere.contains_batch(np.array([1.0, 2.0, 3.0]))  # 1D, not (N,3)


class TestShapeABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Shape()  # type: ignore[abstract]

    def test_default_batch_uses_contains(self):
        """A naive subclass that only implements ``contains`` should still
        get a working ``contains_batch`` from the base class."""

        class _OnlyContains(Shape):
            def contains(self, xyz_mm):
                return xyz_mm[0] >= 0

        s = _OnlyContains()
        result = s.contains_batch(np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(result, [True, False])


class TestBoxConstruction:
    def test_basic_construction(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 20.0, 30.0))
        assert box.center_mm == (0.0, 0.0, 0.0)
        assert box.half_extents_mm == (10.0, 20.0, 30.0)

    def test_list_inputs_accepted(self):
        box = Box(center_mm=[1.0, 2.0, 3.0], half_extents_mm=[5.0, 5.0, 5.0])
        assert box.center_mm == (1.0, 2.0, 3.0)
        assert box.half_extents_mm == (5.0, 5.0, 5.0)

    def test_rejects_wrong_centre_dimensions(self):
        with pytest.raises(ValueError, match="center_mm.*3 elements"):
            Box(center_mm=(0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))

    def test_rejects_wrong_extent_dimensions(self):
        with pytest.raises(ValueError, match="half_extents_mm.*3 elements"):
            Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0))

    def test_rejects_negative_extents(self):
        with pytest.raises(ValueError, match="non-negative"):
            Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, -1.0, 10.0))

    def test_zero_extents_allowed(self):
        """Degenerate point-box: only the centre is inside."""
        box = Box(center_mm=(5.0, 5.0, 5.0), half_extents_mm=(0.0, 0.0, 0.0))
        assert box.contains((5.0, 5.0, 5.0)) is True
        assert box.contains((5.001, 5.0, 5.0)) is False


class TestBoxContains:
    def test_inside_point(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        assert box.contains((1.0, 2.0, 3.0)) is True

    def test_outside_on_one_axis(self):
        """Outside any single axis means outside the box."""
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        assert box.contains((5.0, 5.0, 100.0)) is False

    def test_boundary_is_inside(self):
        """Membership is closed on every axis (matches Sphere semantics)."""
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        assert box.contains((10.0, 10.0, 10.0)) is True
        assert box.contains((-10.0, -10.0, -10.0)) is True

    def test_anisotropic_extents(self):
        """Asymmetric box: each axis has its own half-extent."""
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(5.0, 20.0, 100.0))
        # Inside x (5), y (20), z (100) -- all within
        assert box.contains((4.0, 19.0, 99.0)) is True
        # Outside x (>5) -- excluded even though y, z are well inside
        assert box.contains((6.0, 0.0, 0.0)) is False

    def test_offset_centre(self):
        box = Box(center_mm=(30.0, 20.0, 10.0), half_extents_mm=(5.0, 5.0, 5.0))
        assert box.contains((30.0, 20.0, 10.0)) is True
        assert box.contains((34.0, 24.0, 14.0)) is True
        assert box.contains((36.0, 20.0, 10.0)) is False

    def test_accepts_tuple_list_ndarray(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        assert box.contains((1.0, 2.0, 3.0)) is True
        assert box.contains([1.0, 2.0, 3.0]) is True
        assert box.contains(np.array([1.0, 2.0, 3.0])) is True


class TestBoxBatch:
    def test_basic_batch(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        points = np.array([
            [0.0, 0.0, 0.0],     # inside
            [10.0, 10.0, 10.0],  # boundary -- inside (closed)
            [11.0, 0.0, 0.0],    # outside x
            [5.0, 5.0, 100.0],   # outside z
            [-5.0, -5.0, -5.0],  # inside
        ])
        result = box.contains_batch(points)
        np.testing.assert_array_equal(result, [True, True, False, False, True])

    def test_batch_matches_per_point(self):
        """Vectorised must agree with per-point loop -- catches einsum-style slip-ups."""
        rng = np.random.default_rng(seed=7)
        points = rng.uniform(-50, 50, size=(100, 3))
        box = Box(center_mm=(10.0, -5.0, 0.0), half_extents_mm=(15.0, 25.0, 35.0))
        batch = box.contains_batch(points)
        per_point = np.array([box.contains(p) for p in points])
        np.testing.assert_array_equal(batch, per_point)

    def test_batch_rejects_wrong_shape(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            box.contains_batch(np.array([1.0, 2.0, 3.0]))


class TestBoxCorners:
    def test_returns_eight_corners(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(1.0, 2.0, 3.0))
        corners = box.corners_mm()
        assert corners.shape == (8, 3)

    def test_corners_are_centre_plus_signed_extents(self):
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(1.0, 2.0, 3.0))
        corners = box.corners_mm()
        # Expected: every combination of (+/-1, +/-2, +/-3).
        expected_set = {
            (sx * 1.0, sy * 2.0, sz * 3.0)
            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
        }
        actual_set = {tuple(row) for row in corners}
        assert actual_set == expected_set

    def test_corners_with_offset_centre(self):
        box = Box(center_mm=(10.0, 20.0, 30.0), half_extents_mm=(1.0, 1.0, 1.0))
        corners = box.corners_mm()
        for corner in corners:
            # Each corner sits at +/-1 from the centre on every axis.
            assert all(abs(c - centre) == 1.0 for c, centre in zip(corner, (10.0, 20.0, 30.0)))
