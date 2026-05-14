"""Spatial-region shapes for ROI selection.

The :class:`Shape` abstract base defines a uniform predicate API:
``contains(xyz_mm)`` returns whether a single MNI-mm point is inside
the region. Concrete shapes (sphere now; box, atlas-region next)
implement the predicate. All shapes operate in **MNI millimeters**
so callers never juggle unit conversions inside their selection
logic.

PR #46 (this PR) ships only :class:`Sphere` and refactors the
existing radius-based ROI selection on top of it. :class:`Box` lands
in PR #47, :class:`AtlasRegion` in PR #48.
"""

from __future__ import annotations

import abc
from typing import Sequence, Tuple

import numpy as np


class Shape(abc.ABC):
    """Abstract spatial region defined in MNI millimeters.

    Subclasses implement :meth:`contains` (per-point) and optionally
    override :meth:`contains_batch` to use a vectorised implementation;
    the default loops over points and calls ``contains`` for each. For
    small point counts (~hundreds, the fNIRS HRF library scale) the
    looping default is fast enough that subclasses rarely need to
    override.
    """

    @abc.abstractmethod
    def contains(self, xyz_mm: Sequence[float]) -> bool:
        """Return True iff ``xyz_mm`` (3-element sequence, MNI mm) is inside."""

    def contains_batch(self, points_mm: np.ndarray) -> np.ndarray:
        """Vectorised version of :meth:`contains` for an ``(N, 3)`` array.

        Returns a boolean ``(N,)`` ndarray. The default implementation
        loops over points and calls :meth:`contains` for each; vectorised
        subclasses override for the hot path.
        """
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be an (N, 3) array, got shape {points.shape}"
            )
        return np.array([self.contains(p) for p in points], dtype=bool)


class Sphere(Shape):
    """A sphere defined by an MNI-mm centre and a radius in mm.

    Membership is closed (``distance <= radius``), matching the
    existing ROI-radius semantics in the HRtree panel before this
    refactor. Vectorised batch check avoids the per-point Python
    call overhead.
    """

    __slots__ = ("center_mm", "radius_mm", "_radius_sq")

    def __init__(self, center_mm: Sequence[float], radius_mm: float):
        center = tuple(float(c) for c in center_mm)
        if len(center) != 3:
            raise ValueError(f"center_mm must have 3 elements, got {len(center)}")
        if radius_mm < 0:
            raise ValueError(f"radius_mm must be non-negative, got {radius_mm}")
        self.center_mm: Tuple[float, float, float] = center
        self.radius_mm: float = float(radius_mm)
        self._radius_sq: float = float(radius_mm) * float(radius_mm)

    def contains(self, xyz_mm: Sequence[float]) -> bool:
        cx, cy, cz = self.center_mm
        dx = float(xyz_mm[0]) - cx
        dy = float(xyz_mm[1]) - cy
        dz = float(xyz_mm[2]) - cz
        return dx * dx + dy * dy + dz * dz <= self._radius_sq

    def contains_batch(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be an (N, 3) array, got shape {points.shape}"
            )
        diff = points - np.asarray(self.center_mm, dtype=np.float64)
        return np.einsum("ij,ij->i", diff, diff) <= self._radius_sq

    def __repr__(self) -> str:
        cx, cy, cz = self.center_mm
        return f"Sphere(center_mm=({cx:.2f}, {cy:.2f}, {cz:.2f}), radius_mm={self.radius_mm:.2f})"
