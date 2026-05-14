"""MNI atlas loader + region-lookup primitives.

This module owns the spatial-layer side of atlas-region ROI selection.
The :class:`Atlas` class wraps a label-volume NIfTI plus a label table,
exposes per-point region lookup, and yields binary masks for specific
regions. :class:`hrfunc.spatial.shapes.AtlasRegion` composes on top of
:class:`Atlas` to provide the ``Shape.contains`` predicate that the
HRtree panel's ROI selection consumes.

Today the only bundled atlas is the **FSL Harvard-Oxford Cortical
Atlas** (2 mm maxprob threshold 25%). It ships with the wheel via
:mod:`hrfunc.assets.atlases` so first-run network anxiety is a
non-issue. Adding more atlases (Harvard-Oxford subcortical, AAL,
Schaefer parcellations) is a small additive change in this module --
the :class:`Atlas` interface is modality-agnostic.

Coordinate convention follows the rest of :mod:`hrfunc.spatial`: all
public API operates in MNI millimeters. The NIfTI's ``affine`` maps
voxel indices to MNI mm; the conversion is centralised in
:meth:`Atlas._mm_to_voxel` so callers never see voxel coordinates.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

if TYPE_CHECKING:
    import nibabel  # noqa: F401

logger = logging.getLogger(__name__)


# Asset filenames for the bundled Harvard-Oxford cortical atlas. Both
# files live in ``hrfunc.assets.atlases`` (added to package_data in
# pyproject.toml).
HARVARD_OXFORD_NIFTI = "HarvardOxford-cort-maxprob-thr25-2mm.nii.gz"
HARVARD_OXFORD_LABELS = "HarvardOxford-cort-maxprob-thr25-2mm.labels.json"


@dataclass
class Atlas:
    """An MNI label-volume atlas with region-name lookup.

    Attributes:
        name: Short atlas identifier (e.g. ``"harvard-oxford-cort"``).
            Used in the saved-ROI JSON descriptor + the GUI dropdown.
        volume: Integer label volume as a 3-D ``ndarray``. Each voxel
            holds the integer label index; 0 means background / no
            region.
        affine: 4x4 NIfTI affine mapping voxel indices -> MNI mm.
            Inverted lazily on first lookup.
        labels: List of region names, indexed by atlas label integer.
            ``labels[0]`` is always the background entry.
        background_label: Integer used for "no region" voxels.
            Almost always 0.
    """

    name: str
    volume: np.ndarray
    affine: np.ndarray
    labels: List[str] = field(default_factory=list)
    background_label: int = 0
    # Cached inverse affine, populated on first use. Stored on the
    # dataclass (not as a property) so the lookup hot path doesn't
    # recompute it.
    _affine_inv: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def region_names(self) -> List[str]:
        """Atlas regions in label-index order, with the background entry stripped.

        Useful for populating UI dropdowns where the user picks a
        region by name and the panel resolves back to the label index
        via :meth:`label_index`.
        """
        return [
            label for i, label in enumerate(self.labels)
            if i != self.background_label and label
        ]

    def label_index(self, region_name: str) -> Optional[int]:
        """Return the integer label index for ``region_name``, or None if missing.

        Case-sensitive exact match -- matches what the bundled labels
        file ships with. Callers that want fuzzy match should normalise
        on their end (e.g. ``.lower()`` both sides).
        """
        try:
            return self.labels.index(region_name)
        except ValueError:
            return None

    def region_at(self, xyz_mm: Sequence[float]) -> Optional[str]:
        """Return the atlas region name covering the given MNI mm point.

        Returns ``None`` for background voxels and for points outside
        the volume bounds. Nearest-neighbour sampling -- the atlas is
        a label volume, so interpolation isn't meaningful.

        Used by the GUI's atlas readout ("Region at center: ...").
        """
        voxel = self._mm_to_voxel(np.asarray(xyz_mm, dtype=np.float64))
        if voxel is None:
            return None
        label = int(self.volume[voxel])
        if label == self.background_label:
            return None
        if 0 <= label < len(self.labels):
            return self.labels[label] or None
        return None

    def region_at_batch(self, points_mm: np.ndarray) -> List[Optional[str]]:
        """Vectorised :meth:`region_at` for an ``(N, 3)`` point array.

        Returns a list of region names (or ``None`` for background /
        out-of-volume points). Used by the GUI when listing the atlas
        region of every HRF in the visible set.
        """
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be (N, 3), got shape {points.shape}"
            )
        voxels = self._mm_to_voxel_batch(points)
        out: List[Optional[str]] = []
        for voxel in voxels:
            if voxel is None:
                out.append(None)
                continue
            label = int(self.volume[voxel])
            if label == self.background_label or not (0 <= label < len(self.labels)):
                out.append(None)
            else:
                out.append(self.labels[label] or None)
        return out

    def region_mask(self, region_name: str) -> Optional[np.ndarray]:
        """Boolean volume mask for one named region.

        Returns ``None`` if the region name isn't in the atlas.
        Used by :class:`hrfunc.spatial.shapes.AtlasRegion` to back
        the ``contains`` predicate.
        """
        idx = self.label_index(region_name)
        if idx is None or idx == self.background_label:
            return None
        return self.volume == idx

    def contains_mm(self, xyz_mm: Sequence[float], region_name: str) -> bool:
        """Convenience: True iff ``xyz_mm`` falls within ``region_name``.

        Equivalent to ``self.region_at(xyz_mm) == region_name`` but
        skips the label-index round-trip on every call.
        """
        idx = self.label_index(region_name)
        if idx is None or idx == self.background_label:
            return False
        voxel = self._mm_to_voxel(np.asarray(xyz_mm, dtype=np.float64))
        if voxel is None:
            return False
        return int(self.volume[voxel]) == idx

    def contains_batch(
        self,
        points_mm: np.ndarray,
        region_name: str,
    ) -> np.ndarray:
        """Vectorised ``contains_mm`` for an ``(N, 3)`` point array."""
        idx = self.label_index(region_name)
        if idx is None or idx == self.background_label:
            return np.zeros(len(points_mm), dtype=bool)
        points = np.asarray(points_mm, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"points_mm must be (N, 3), got shape {points.shape}"
            )
        voxels = self._mm_to_voxel_batch(points)
        out = np.zeros(len(points), dtype=bool)
        for i, voxel in enumerate(voxels):
            if voxel is None:
                continue
            out[i] = int(self.volume[voxel]) == idx
        return out

    # ------------------------------------------------------------------
    # mm <-> voxel conversion (private)
    # ------------------------------------------------------------------

    def _ensure_affine_inv(self) -> np.ndarray:
        if self._affine_inv is None:
            self._affine_inv = np.linalg.inv(self.affine)
        return self._affine_inv

    def _mm_to_voxel(
        self, xyz_mm: np.ndarray
    ) -> Optional[Tuple[int, int, int]]:
        """MNI mm -> integer voxel tuple, or None if out of volume bounds.

        Nearest-neighbour rounding via ``np.rint``. Out-of-volume
        points return None rather than wrapping or clamping -- silent
        clamping would put a stray point on the volume edge into
        whatever the edge voxel labels.
        """
        inv = self._ensure_affine_inv()
        homo = np.array([xyz_mm[0], xyz_mm[1], xyz_mm[2], 1.0], dtype=np.float64)
        voxel = inv @ homo
        i = int(np.rint(voxel[0]))
        j = int(np.rint(voxel[1]))
        k = int(np.rint(voxel[2]))
        shape = self.volume.shape
        if 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]:
            return (i, j, k)
        return None

    def _mm_to_voxel_batch(
        self, points_mm: np.ndarray
    ) -> List[Optional[Tuple[int, int, int]]]:
        inv = self._ensure_affine_inv()
        homo = np.column_stack(
            [points_mm, np.ones(len(points_mm), dtype=np.float64)]
        )
        voxels = (inv @ homo.T).T[:, :3]
        rounded = np.rint(voxels).astype(np.int64)
        shape = self.volume.shape
        within = np.all(
            (rounded >= 0) & (rounded < np.array(shape, dtype=np.int64)),
            axis=1,
        )
        out: List[Optional[Tuple[int, int, int]]] = []
        for ok, vox in zip(within, rounded):
            if ok:
                out.append((int(vox[0]), int(vox[1]), int(vox[2])))
            else:
                out.append(None)
        return out


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


_ATLAS_CACHE: Dict[str, Optional[Atlas]] = {}


def load_harvard_oxford_cortical() -> Optional[Atlas]:
    """Load the bundled Harvard-Oxford cortical atlas (2 mm, maxprob thr=25%).

    The atlas ships with the wheel under
    :mod:`hrfunc.assets.atlases`. No network access required; the
    one-time pre-bundling avoids the first-run download anxiety
    common to nilearn-style on-demand fetches.

    Cached per-process so the NIfTI is parsed at most once. Returns
    ``None`` if nibabel can't be imported or the assets are missing
    (defensive: callers should fall back to "atlas unavailable" UI
    rather than crashing).
    """
    cache_key = "harvard-oxford-cort"
    if cache_key in _ATLAS_CACHE:
        return _ATLAS_CACHE[cache_key]

    try:
        import nibabel
        from importlib import resources
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_harvard_oxford_cortical: nibabel unavailable (%s)", exc)
        _ATLAS_CACHE[cache_key] = None
        return None

    try:
        nifti_ref = resources.files("hrfunc.assets.atlases") / HARVARD_OXFORD_NIFTI
        labels_ref = resources.files("hrfunc.assets.atlases") / HARVARD_OXFORD_LABELS
        with resources.as_file(nifti_ref) as nifti_path:
            img = nibabel.load(str(nifti_path))
            volume = np.asarray(img.get_fdata(), dtype=np.int64)
            affine = np.asarray(img.affine, dtype=np.float64)
        with resources.as_file(labels_ref) as labels_path:
            labels_payload = json.loads(labels_path.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "load_harvard_oxford_cortical: asset load failed (%s)", exc
        )
        _ATLAS_CACHE[cache_key] = None
        return None

    atlas = Atlas(
        name=labels_payload.get("atlas_name", cache_key),
        volume=volume,
        affine=affine,
        labels=list(labels_payload.get("labels") or []),
        background_label=int(labels_payload.get("background_label", 0)),
    )
    _ATLAS_CACHE[cache_key] = atlas
    return atlas
