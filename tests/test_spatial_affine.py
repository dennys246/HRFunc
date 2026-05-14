"""Unit tests for :mod:`hrfunc.spatial.affine`.

Affine is a stub in PR #46 — the full registration-QA workflow lands
in PR #49 (v1.3.1 anatomical NIfTI viewer). These tests lock the
minimal contract so PR #49 can build on it without surprises.
"""

from __future__ import annotations

import numpy as np
import pytest

from hrfunc.spatial.affine import apply_affine, identity_affine


class TestIdentityAffine:
    def test_is_eye_four(self):
        np.testing.assert_array_equal(identity_affine(), np.eye(4))

    def test_is_float64(self):
        assert identity_affine().dtype == np.float64


class TestApplyAffine:
    def test_identity_returns_points_unchanged(self):
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = apply_affine(identity_affine(), points)
        np.testing.assert_array_almost_equal(result, points)

    def test_pure_translation(self):
        affine = np.eye(4)
        affine[:3, 3] = [10.0, 20.0, 30.0]
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        result = apply_affine(affine, points)
        np.testing.assert_array_almost_equal(
            result, [[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]]
        )

    def test_pure_scale(self):
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        points = np.array([[1.0, 1.0, 1.0]])
        result = apply_affine(affine, points)
        np.testing.assert_array_almost_equal(result, [[2.0, 3.0, 4.0]])

    def test_single_point_3vec_returns_3vec(self):
        result = apply_affine(identity_affine(), np.array([1.0, 2.0, 3.0]))
        assert result.shape == (3,)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_rejects_wrong_affine_shape(self):
        with pytest.raises(ValueError, match="4x4"):
            apply_affine(np.eye(3), np.array([[1.0, 2.0, 3.0]]))

    def test_rejects_wrong_point_shape(self):
        with pytest.raises(ValueError, match=r"\(N, 3\)"):
            apply_affine(identity_affine(), np.array([[1.0, 2.0]]))
