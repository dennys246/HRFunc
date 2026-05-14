"""Modality-agnostic spatial primitives.

This package holds the data types and geometric helpers used by every
downstream consumer of HRF spatial information: the GUI's HRtree
panel, future ROI / clustering tools, and the planned parallel fMRI
HRF pipeline. None of the code here knows about HbO/HbR, optodes, or
any fNIRS-specific concept; that knowledge stays in :mod:`hrfunc.hrtree`.

The boundary is intentional. fNIRS and fMRI both produce HRFs that
live in MNI space and share the same downstream consumers (spatial
selection, 3D visualization, ROI averaging). Keeping the spatial
layer modality-agnostic lets a future fMRI module emit
:class:`~hrfunc.spatial.point.HRFPoint` instances into the same
pipeline without forking the viz stack.

Public surface:

- :class:`HRFPoint` — the modality-bridge DTO. Holds one HRF's
  spatial location (MNI mm), mean / std trace, sampling rate, free-form
  context dict, and a modality tag.
- :class:`Shape` / :class:`Sphere` — point-in-region predicates for
  ROI selection. Both operate in MNI mm.
- :func:`meters_to_mm` / :func:`mm_to_meters` — the single conversion
  point between MNE's internal meter coordinates and the mm scale
  the spatial layer uses throughout.
- :func:`apply_affine` / :func:`identity_affine` — 4×4 transform
  helpers. Used by the v1.3.1 anatomical NIfTI viewer to map user-
  supplied image coordinates into MNI mm.
"""

from .affine import apply_affine, identity_affine
from .coords import meters_to_mm, mm_to_meters
from .point import HRFPoint
from .shapes import Box, Shape, Sphere

__all__ = [
    "Box",
    "HRFPoint",
    "Shape",
    "Sphere",
    "apply_affine",
    "identity_affine",
    "meters_to_mm",
    "mm_to_meters",
]
