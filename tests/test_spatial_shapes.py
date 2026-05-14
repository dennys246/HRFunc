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


# ---------------------------------------------------------------------------
# Box: oriented (PR #52 -- orientation refactor)
# ---------------------------------------------------------------------------


def _rot_z(theta_rad: float) -> np.ndarray:
    """Rotation matrix for a counter-clockwise rotation about the z axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rot_y(theta_rad: float) -> np.ndarray:
    """Rotation matrix for a counter-clockwise rotation about the y axis."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


class TestBoxOrientationDefaults:
    """Identity orientation must produce bit-identical behaviour to the
    pre-PR-#52 AABB implementation -- otherwise the existing 33 Box
    tests would have failed. These tests pin that contract."""

    def test_none_orientation_is_identity(self):
        box = Box(center_mm=(0, 0, 0), half_extents_mm=(5, 5, 5))
        np.testing.assert_array_equal(box.orientation_mm, np.eye(3))

    def test_is_axis_aligned_true_for_default(self):
        box = Box(center_mm=(0, 0, 0), half_extents_mm=(5, 5, 5))
        assert box.is_axis_aligned() is True

    def test_explicit_identity_matches_default(self):
        a = Box(center_mm=(1, 2, 3), half_extents_mm=(5, 5, 5))
        b = Box(
            center_mm=(1, 2, 3),
            half_extents_mm=(5, 5, 5),
            orientation_mm=np.eye(3),
        )
        # Same membership decisions across a sweep of points.
        rng = np.random.default_rng(seed=11)
        points = rng.uniform(-20, 20, size=(50, 3))
        np.testing.assert_array_equal(
            a.contains_batch(points), b.contains_batch(points)
        )

    def test_repr_says_aabb_for_identity(self):
        box = Box(center_mm=(0, 0, 0), half_extents_mm=(1, 1, 1))
        assert "AABB" in repr(box)


class TestBoxOrientationValidation:
    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="3x3"):
            Box(
                center_mm=(0, 0, 0),
                half_extents_mm=(5, 5, 5),
                orientation_mm=np.eye(2),
            )

    def test_non_orthogonal_rejected(self):
        """A scaled or sheared matrix would silently produce wrong
        membership results -- reject at construction time."""
        bad = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError, match="orthogonal"):
            Box(
                center_mm=(0, 0, 0),
                half_extents_mm=(5, 5, 5),
                orientation_mm=bad,
            )

    def test_pure_rotation_accepted(self):
        """A canonical rotation matrix passes validation cleanly."""
        Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(5, 5, 5),
            orientation_mm=_rot_z(np.pi / 4),
        )

    def test_reflection_accepted(self):
        """Reflections are orthogonal too -- we accept them rather than
        check determinant sign, because they produce geometrically
        correct membership for a symmetric box."""
        reflect_x = np.diag([-1.0, 1.0, 1.0])
        Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(5, 5, 5),
            orientation_mm=reflect_x,
        )


class TestBoxOrientationContains:
    """Rotation membership semantics. The key intuition: a point's
    membership depends on its representation in the BOX'S LOCAL
    frame after the box has been rotated; equivalently, rotating
    the box by R is the same as rotating world points by R^T
    before checking the AABB-equivalent extents."""

    def test_90deg_z_long_box_excludes_world_x_axis_point(self):
        """A long-along-x box rotated 90 degrees about z now points
        along world y. A point at world (10, 0, 0) was inside the
        unrotated long box; rotated, it's outside (lies along the
        new short axis)."""
        box = Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(15.0, 5.0, 5.0),
            orientation_mm=_rot_z(np.pi / 2),
        )
        assert box.contains((10.0, 0.0, 0.0)) is False

    def test_90deg_z_long_box_includes_world_y_axis_point(self):
        """Same box -- a point at world (0, 10, 0) sits along the
        rotated long axis and is inside."""
        box = Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(15.0, 5.0, 5.0),
            orientation_mm=_rot_z(np.pi / 2),
        )
        assert box.contains((0.0, 10.0, 0.0)) is True

    def test_45deg_z_membership(self):
        """At 45 degrees, world (5, 5, 0) is on the rotated +x axis at
        distance sqrt(50) ~ 7.07; outside if extent_x is 5."""
        box = Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(5.0, 5.0, 5.0),
            orientation_mm=_rot_z(np.pi / 4),
        )
        # The point (sqrt(2)*2, sqrt(2)*2, 0) ~= (2.83, 2.83, 0)
        # sits at box-local (4.0, 0.0, 0.0) -- inside.
        assert box.contains((2.828, 2.828, 0.0)) is True
        # The point (5, 5, 0) sits at box-local (~7.07, 0, 0) -- outside.
        assert box.contains((5.0, 5.0, 0.0)) is False

    def test_offset_centre_with_rotation(self):
        """Centre + rotation compose correctly. Box at (10, 0, 0)
        rotated 90 degrees about z, half_extents (15, 5, 5): the
        rotated long axis points along world +y from the centre."""
        box = Box(
            center_mm=(10.0, 0.0, 0.0),
            half_extents_mm=(15.0, 5.0, 5.0),
            orientation_mm=_rot_z(np.pi / 2),
        )
        # 10 units along world +y from the centre -- inside.
        assert box.contains((10.0, 10.0, 0.0)) is True
        # 20 units along world +x from the centre -- outside.
        assert box.contains((30.0, 0.0, 0.0)) is False


class TestBoxOrientationBatch:
    def test_batch_matches_per_point_for_rotated(self):
        rng = np.random.default_rng(seed=29)
        points = rng.uniform(-30, 30, size=(200, 3))
        box = Box(
            center_mm=(5.0, -3.0, 1.0),
            half_extents_mm=(8.0, 12.0, 6.0),
            orientation_mm=_rot_z(np.pi / 6),
        )
        batch = box.contains_batch(points)
        per_point = np.array([box.contains(p) for p in points])
        np.testing.assert_array_equal(batch, per_point)


class TestBoxOrientationCorners:
    def test_corners_rotated_into_world(self):
        """For a 90 degree rotation about z, an axis-aligned corner at
        local (1, 0, 0) ends up at world (0, 1, 0)."""
        box = Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(1.0, 2.0, 3.0),
            orientation_mm=_rot_z(np.pi / 2),
        )
        corners = box.corners_mm()
        # Every world corner must equal R @ local_corner where local
        # corners are the +/- combinations of half_extents.
        R = _rot_z(np.pi / 2)
        local_signs = np.array(
            [[s1, s2, s3] for s1 in (-1, 1) for s2 in (-1, 1) for s3 in (-1, 1)],
            dtype=np.float64,
        )
        # Sign order in our class is (-,-,-), (+,-,-), (-,+,-), ...
        # We rebuild it explicitly for comparison.
        expected_local_order = np.array(
            [
                [-1, -1, -1], [+1, -1, -1], [-1, +1, -1], [+1, +1, -1],
                [-1, -1, +1], [+1, -1, +1], [-1, +1, +1], [+1, +1, +1],
            ], dtype=np.float64,
        )
        expected_local = expected_local_order * np.array([1.0, 2.0, 3.0])
        expected_world = expected_local @ R.T
        np.testing.assert_allclose(corners, expected_world, atol=1e-12)

    def test_corners_still_8_under_rotation(self):
        box = Box(
            center_mm=(10, 20, 30),
            half_extents_mm=(5, 5, 5),
            orientation_mm=_rot_y(np.pi / 3),
        )
        assert box.corners_mm().shape == (8, 3)

    def test_is_axis_aligned_false_under_rotation(self):
        box = Box(
            center_mm=(0, 0, 0),
            half_extents_mm=(5, 5, 5),
            orientation_mm=_rot_z(np.pi / 7),
        )
        assert box.is_axis_aligned() is False
        assert "oriented" in repr(box)
