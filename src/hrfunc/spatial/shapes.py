"""Spatial-region shapes for ROI selection.

The :class:`Shape` abstract base defines a uniform predicate API:
``contains(xyz_mm)`` returns whether a single MNI-mm point is inside
the region. Concrete shapes implement the predicate. All shapes
operate in **MNI millimeters** so callers never juggle unit
conversions inside their selection logic.

Currently shipped:

- :class:`Sphere` — closed-ball membership, used by the existing
  radius-based ROI selection in the HRtree panel.
- :class:`Box` — oriented bounding box (OBB). Defaults to axis-
  aligned (identity orientation) for back-compat with the v1.3
  Cluster sub-tab and the saved-ROI JSON schema. The orientation
  parameter exists in the spatial-layer API so future UIs
  (rotatable-box drag handles, atlas-region oriented patches) can
  use it without retrofitting; the GUI's Cluster sub-tab doesn't
  yet expose rotation controls.
- :class:`AtlasRegion` — membership defined by an MNI label-volume
  atlas (see :mod:`hrfunc.spatial.atlas`). A point is "inside" iff
  the atlas labels its voxel as the chosen region. Used by the v1.3
  Cluster sub-tab's atlas-region selection mode.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    from .atlas import Atlas


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
    """An oriented bounding box (OBB) defined by centre + half-extents + rotation.

    Defaults to axis-aligned: when ``orientation_mm`` is ``None`` or
    the identity matrix, the box is an AABB and behaviour is bit-
    identical to a pure axis-aligned implementation. The orientation
    parameter exists so future UIs (rotatable-box drag handles,
    oriented atlas patches) can construct rotated boxes without a
    second shape class.

    Membership semantics:
        A world-space point is inside iff its representation in the
        box-local frame -- ``R^T (p - c)`` where ``R`` is the
        orientation matrix and ``c`` is the centre -- has every
        component within ``+/-`` half_extents. Closed on every axis,
        matching :class:`Sphere`'s closed-ball semantics so boundary
        points behave consistently across shape types.

    Coordinate convention:
        ``orientation_mm`` is a column-vector rotation matrix:
        ``world_vec = R @ local_vec``. Columns of ``R`` are the
        box's local x / y / z axes expressed in world coordinates.
        The transpose (= inverse, since R is orthogonal) takes a
        world point into the box-local frame.

    Validation:
        ``orientation_mm`` is checked for orthogonality
        (``R @ R.T ≈ I``) so passing a scaled or sheared matrix
        raises rather than silently producing wrong-region results.
        Determinant sign is NOT enforced -- a reflection still
        produces correct membership (the box is symmetric across
        all three axes), it just flips the corner order in
        :meth:`corners_mm`. Acceptable for AABB-equivalent reasoning;
        callers that care about handedness validate themselves.
    """

    __slots__ = ("center_mm", "half_extents_mm", "orientation_mm")

    def __init__(
        self,
        center_mm: Sequence[float],
        half_extents_mm: Sequence[float],
        orientation_mm: Optional[np.ndarray] = None,
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

        # Orientation: None or identity -> AABB fast path; otherwise
        # validate orthogonality so callers can't silently pass a
        # scaled / sheared matrix and get geometrically wrong results.
        if orientation_mm is None:
            orientation = np.eye(3, dtype=np.float64)
        else:
            orientation = np.asarray(orientation_mm, dtype=np.float64)
            if orientation.shape != (3, 3):
                raise ValueError(
                    f"orientation_mm must be a 3x3 matrix, got shape "
                    f"{orientation.shape}"
                )
            if not np.allclose(
                orientation @ orientation.T, np.eye(3), atol=1e-6
            ):
                raise ValueError(
                    "orientation_mm must be orthogonal (a valid rotation "
                    "or rotoreflection matrix)"
                )

        self.center_mm: Tuple[float, float, float] = center
        self.half_extents_mm: Tuple[float, float, float] = extents
        self.orientation_mm: np.ndarray = orientation

    def contains(self, xyz_mm: Sequence[float]) -> bool:
        # Translate to box origin, rotate into box-local frame, then
        # check half-extents on every axis. For identity orientation
        # the rotation is a no-op (result == translated point) and
        # the math collapses to the AABB form.
        offset = np.asarray(xyz_mm, dtype=np.float64) - np.asarray(
            self.center_mm, dtype=np.float64
        )
        local = self.orientation_mm.T @ offset
        hx, hy, hz = self.half_extents_mm
        return (
            abs(float(local[0])) <= hx
            and abs(float(local[1])) <= hy
            and abs(float(local[2])) <= hz
        )

    def contains_batch(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be an (N, 3) array, got shape {points.shape}"
            )
        center = np.asarray(self.center_mm, dtype=np.float64)
        extents = np.asarray(self.half_extents_mm, dtype=np.float64)
        # (N, 3) @ R = (N, 3) where row i is R^T @ row_i.
        # Use the identity ``points @ R == (R^T @ points^T)^T`` to keep
        # the multiplication in (N, 3) shape without an explicit transpose.
        local = (points - center) @ self.orientation_mm
        return np.all(np.abs(local) <= extents, axis=1)

    def corners_mm(self) -> np.ndarray:
        """Return the 8 corner vertices of the box as an ``(8, 3)`` array.

        Used by the viz layer to render the translucent box overlay.
        Corner order is the standard "binary count of sign flips":
        ``(-x, -y, -z), (+x, -y, -z), (-x, +y, -z), (+x, +y, -z), ...``
        so consumers can deterministically pick the right vertex when
        building face triangles.

        For a rotated box, the local-frame corners are first computed
        in canonical sign-flip order, then rotated into world space.
        Identity orientation collapses this to the AABB form (corners
        sit at ``center +/- extents``).
        """
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
        extents = np.asarray(self.half_extents_mm, dtype=np.float64)
        center = np.asarray(self.center_mm, dtype=np.float64)
        local_corners = signs * extents  # (8, 3)
        # Rotate local corners to world: world = R @ local (per-vertex).
        # Vectorised as (8, 3) @ R.T == (R @ local.T).T.
        world_corners = local_corners @ self.orientation_mm.T
        return center + world_corners

    def is_axis_aligned(self) -> bool:
        """True iff the orientation matrix is the identity (within tolerance).

        Used by callers (workspace_io, the viz layer) that want to
        emit a simpler representation for the common AABB case and
        only carry the orientation matrix when it actually differs
        from identity.
        """
        return bool(
            np.allclose(self.orientation_mm, np.eye(3), atol=1e-9)
        )

    def __repr__(self) -> str:
        cx, cy, cz = self.center_mm
        hx, hy, hz = self.half_extents_mm
        aligned = "AABB" if self.is_axis_aligned() else "oriented"
        return (
            f"Box(center_mm=({cx:.2f}, {cy:.2f}, {cz:.2f}), "
            f"half_extents_mm=({hx:.2f}, {hy:.2f}, {hz:.2f}), {aligned})"
        )


class AtlasRegion(Shape):
    """Spatial region defined by an MNI label-volume atlas region.

    A point is "inside" iff the atlas labels its voxel as the chosen
    region. Nearest-neighbour voxel sampling; out-of-volume points are
    "outside" (not background -- callers occasionally distinguish, but
    for membership the answer is the same).

    Used by the v1.3 Cluster sub-tab's atlas-region selection mode.
    The :class:`Atlas` instance carries the label volume + affine +
    label table; this class is the thin :class:`Shape` adapter that
    plugs atlas membership into the same ROI machinery
    (:func:`hrfunc.gui.components.hrtree_panel.compute_roi_keys_by_shape`)
    that sphere and box selection use.

    Membership is read-only -- callers can't translate or rotate the
    region (the atlas defines the geometry). For "select all HRFs in
    Frontal Pole within a 5 mm box", future PRs will compose
    :class:`AtlasRegion` with :class:`Box` via set intersection.
    """

    __slots__ = ("atlas", "region_name")

    def __init__(self, atlas: "Atlas", region_name: str):
        if atlas is None:
            raise ValueError("atlas must not be None")
        if not region_name:
            raise ValueError("region_name must be a non-empty string")
        if atlas.label_index(region_name) is None:
            raise ValueError(
                f"region_name {region_name!r} not present in atlas "
                f"{atlas.name!r}"
            )
        self.atlas = atlas
        self.region_name = region_name

    def contains(self, xyz_mm: Sequence[float]) -> bool:
        return self.atlas.contains_mm(xyz_mm, self.region_name)

    def contains_batch(self, points_mm: np.ndarray) -> np.ndarray:
        return self.atlas.contains_batch(points_mm, self.region_name)

    def __repr__(self) -> str:
        return (
            f"AtlasRegion(atlas={self.atlas.name!r}, "
            f"region_name={self.region_name!r})"
        )
