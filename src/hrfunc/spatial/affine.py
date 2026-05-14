"""4×4 homogeneous affine transform helpers.

Lives in :mod:`hrfunc.spatial` (not :mod:`hrfunc.viz`) because affines
are a coordinate-system concern, not a rendering one. The full
registration UX — loading a user-supplied NIfTI, deriving its
sform/qform, letting the user override the transform — lands in
PR #49 (v1.3.1). This module is the minimal API that PR will build
on; locking it in now keeps the v1.3.1 work focused on the loader and
QA UX rather than the math.
"""

from __future__ import annotations

import numpy as np


def identity_affine() -> np.ndarray:
    """Return the 4×4 identity matrix.

    Convenience for callers that want an "no transform" affine without
    having to reach for ``numpy.eye(4)`` directly.
    """
    return np.eye(4, dtype=np.float64)


def apply_affine(affine: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4×4 homogeneous affine to an ``(N, 3)`` array of points.

    Args:
        affine: 4×4 transform matrix (e.g. a NIfTI ``sform``/``qform``).
        points: ``(N, 3)`` array of points in the source coordinate
            system. A single ``(3,)`` point is also accepted for
            convenience and returned with the leading axis preserved.

    Returns:
        Transformed points as a float64 array with the same leading-axis
        shape as ``points``.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"affine must be 4x4, got shape {affine.shape}")

    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    if single:
        if pts.shape != (3,):
            raise ValueError(
                f"single point must be shape (3,), got {pts.shape}"
            )
        pts = pts.reshape(1, 3)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {pts.shape}")

    homo = np.column_stack([pts, np.ones(len(pts))])
    transformed = (affine @ homo.T).T[:, :3]
    return transformed[0] if single else transformed
