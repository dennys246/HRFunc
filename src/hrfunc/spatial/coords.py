"""Unit conversions between meters (MNE-internal) and millimeters (MNI mm).

MNE-Python stores optode and channel positions in meters because the
sensor-space transforms are defined that way internally. Neuroimaging
atlases (Harvard-Oxford, AAL, MNI152) and the conventions used in
fNIRS / fMRI publications work in millimeters.

To avoid the "is this mm or m?" bug at every site that touches
coordinates, the :mod:`hrfunc.spatial` and :mod:`hrfunc.viz` layers
adopt the convention that **all spatial reasoning happens in MNI mm**.
The meter representation is treated as an MNE storage detail; callers
convert at the boundary using the helpers in this module.
"""

from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, "np.ndarray", list, tuple]


def meters_to_mm(value: ArrayLike) -> "np.ndarray":
    """Convert a scalar, sequence, or array from meters to millimeters.

    Accepts any numeric input that ``numpy.asarray`` can interpret and
    returns a float ndarray in millimeters. A 0-d ndarray is returned
    for scalar input; callers that want a Python float can call
    ``.item()`` on the result.
    """
    arr = np.asarray(value, dtype=np.float64)
    return arr * 1000.0


def mm_to_meters(value: ArrayLike) -> "np.ndarray":
    """Convert a scalar, sequence, or array from millimeters to meters."""
    arr = np.asarray(value, dtype=np.float64)
    return arr / 1000.0
