"""The :class:`HRFPoint` DTO — the modality bridge between fNIRS and fMRI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class HRFPoint:
    """One HRF's spatial location + trace, in a modality-agnostic shape.

    This is the contract between HRF *producers* (today :mod:`hrfunc.hrtree`
    for fNIRS, in the future a parallel fMRI pipeline) and HRF
    *consumers* (the GUI's HRtree viz, spatial-shape ROI selection,
    atlas lookups). Producers emit ``HRFPoint`` streams; consumers read
    them without caring which modality produced the data.

    Fields:

    - ``xyz_mm`` — 3-tuple of MNI millimeters. The spatial layer's
      single coordinate convention; producers convert from whatever
      they store internally (e.g. MNE meters for fNIRS) at the
      boundary.
    - ``hrf_mean`` / ``hrf_std`` — the HRF trace and per-sample
      standard deviation. ``hrf_std`` may be ``None`` for HRFs that
      didn't carry uncertainty.
    - ``sfreq`` — sampling frequency of the trace, in Hz.
    - ``context`` — free-form metadata. Modality-specific fields
      (``oxygenation`` for fNIRS, future fMRI-specific fields) ride
      here so the DTO doesn't need to grow per-modality columns.
    - ``modality_tag`` — short string identifying the source pipeline,
      e.g. ``"fnirs"`` or ``"fmri"``. Consumers can use it for display
      grouping or for routing modality-specific behavior.

    The dataclass is frozen so consumers can pass it around without
    worrying about mutation. Trace arrays are stored as ``np.ndarray``
    for downstream math; producers should convert lists at construction
    time.
    """

    xyz_mm: Tuple[float, float, float]
    hrf_mean: np.ndarray
    hrf_std: Optional[np.ndarray]
    sfreq: float
    context: Dict[str, Any] = field(default_factory=dict)
    modality_tag: str = "unknown"
