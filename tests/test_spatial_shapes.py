"""Unit tests for :mod:`hrfunc.spatial.shapes`.

The :class:`Shape` API is the foundation of the upcoming ROI shape
selector (sphere now, box + atlas-region in PRs #47 and #48). These
tests pin down the contains-semantics so the new shapes in those
PRs can be added with the same predicate contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrfunc.spatial.shapes import Shape, Sphere


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
