"""Spatial-region shapes for ROI selection.

The :class:`Shape` abstract base defines a uniform predicate API:
``contains(xyz_mm)`` returns whether a single MNI-mm point is inside
the region. Concrete shapes implement the predicate. All shapes
operate in **MNI millimeters** so callers never juggle unit
conversions inside their selection logic.

Currently shipped:

- :class:`Sphere` — closed-ball membership, used by the existing
  radius-based ROI selection in the HRtree panel.
- :class:`Box` — axis-aligned bounding box, used by the v1.3
  Cluster sub-tab's box-mode ROI selection. Rotation is intentionally
  excluded: it adds significant UX complexity (slider-based 3D
  rotation is a usability trap) for negligible scientific benefit
  (fNIRS publications axis-align to MNI by convention).

Atlas-region membership lands in a follow-up PR with the Harvard-
Oxford atlas integration.
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


class Box(Shape):
    """An axis-aligned bounding box defined by centre + half-extents in mm.

    Membership is closed on all three axes: a point is inside iff
    ``|x - cx| <= hx`` and ``|y - cy| <= hy`` and ``|z - cz| <= hz``.
    Closed semantics match :class:`Sphere` so boundary points behave
    consistently across shape types -- which is the principle of
    least surprise for researchers comparing ROIs.

    Rotation is intentionally out of scope. fNIRS publication
    conventions report ROIs in MNI axes (L/R, A/P, S/I); allowing
    rotation would invite users to slip into non-standard reference
    frames without realising it, and the slider-based 3D rotation UX
    is itself a usability trap. Researchers who need rotated regions
    are better served by atlas-defined regions (a follow-up PR).
    """

    __slots__ = ("center_mm", "half_extents_mm")

    def __init__(
        self,
        center_mm: Sequence[float],
        half_extents_mm: Sequence[float],
    ):
        center = tuple(float(c) for c in center_mm)
        if len(center) != 3:
            raise ValueError(f"center_mm must have 3 elements, got {len(center)}")
        extents = tuple(float(h) for h in half_extents_mm)
        if len(extents) != 3:
            raise ValueError(
                f"half_extents_mm must have 3 elements, got {len(extents)}"
            )
        if any(h < 0 for h in extents):
            raise ValueError(
                f"half_extents_mm must all be non-negative, got {extents}"
            )
        self.center_mm: Tuple[float, float, float] = center
        self.half_extents_mm: Tuple[float, float, float] = extents

    def contains(self, xyz_mm: Sequence[float]) -> bool:
        cx, cy, cz = self.center_mm
        hx, hy, hz = self.half_extents_mm
        dx = abs(float(xyz_mm[0]) - cx)
        dy = abs(float(xyz_mm[1]) - cy)
        dz = abs(float(xyz_mm[2]) - cz)
        return dx <= hx and dy <= hy and dz <= hz

    def contains_batch(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be an (N, 3) array, got shape {points.shape}"
            )
        center = np.asarray(self.center_mm, dtype=np.float64)
        extents = np.asarray(self.half_extents_mm, dtype=np.float64)
        return np.all(np.abs(points - center) <= extents, axis=1)

    def corners_mm(self) -> np.ndarray:
        """Return the 8 corner vertices of the box as an ``(8, 3)`` array.

        Used by the viz layer to render the translucent box overlay.
        Corner order is the standard "binary count of sign flips":
        ``(-x, -y, -z), (+x, -y, -z), (-x, +y, -z), (+x, +y, -z), ...``
        so consumers can deterministically pick the right vertex when
        building face triangles.
        """
        cx, cy, cz = self.center_mm
        hx, hy, hz = self.half_extents_mm
        signs = np.array(
            [
                [-1, -1, -1],
                [+1, -1, -1],
                [-1, +1, -1],
                [+1, +1, -1],
                [-1, -1, +1],
                [+1, -1, +1],
                [-1, +1, +1],
                [+1, +1, +1],
            ],
            dtype=np.float64,
        )
        extents = np.array([hx, hy, hz], dtype=np.float64)
        center = np.array([cx, cy, cz], dtype=np.float64)
        return center + signs * extents

    def __repr__(self) -> str:
        cx, cy, cz = self.center_mm
        hx, hy, hz = self.half_extents_mm
        return (
            f"Box(center_mm=({cx:.2f}, {cy:.2f}, {cz:.2f}), "
            f"half_extents_mm=({hx:.2f}, {hy:.2f}, {hz:.2f}))"
        )
