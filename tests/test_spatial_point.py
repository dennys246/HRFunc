"""Unit tests for :class:`hrfunc.spatial.point.HRFPoint`.

The DTO is the contract between HRF producers (fNIRS / fMRI) and
HRF consumers (spatial selection, viz). These tests pin down its
shape so a future fMRI module can target the same contract without
breaking fNIRS callers.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from hrfunc.spatial.point import HRFPoint


class TestHRFPointConstruction:
    def test_basic_construction(self):
        trace = np.array([0.0, 1.0, 0.5])
        point = HRFPoint(
            xyz_mm=(10.0, 20.0, 30.0),
            hrf_mean=trace,
            hrf_std=None,
            sfreq=10.0,
            context={"task": "rest"},
            modality_tag="fnirs",
        )
        assert point.xyz_mm == (10.0, 20.0, 30.0)
        np.testing.assert_array_equal(point.hrf_mean, trace)
        assert point.hrf_std is None
        assert point.sfreq == 10.0
        assert point.context == {"task": "rest"}
        assert point.modality_tag == "fnirs"

    def test_std_is_optional(self):
        point = HRFPoint(
            xyz_mm=(0.0, 0.0, 0.0),
            hrf_mean=np.array([1.0]),
            hrf_std=None,
            sfreq=1.0,
        )
        assert point.hrf_std is None

    def test_context_default_is_independent(self):
        """Mutable-default-argument trap regression: each instance must own
        its own context dict so mutating one doesn't leak to others."""
        a = HRFPoint(xyz_mm=(0, 0, 0), hrf_mean=np.array([]), hrf_std=None, sfreq=1.0)
        b = HRFPoint(xyz_mm=(1, 1, 1), hrf_mean=np.array([]), hrf_std=None, sfreq=1.0)
        a.context["leak"] = True
        assert "leak" not in b.context


class TestHRFPointImmutability:
    def test_frozen_prevents_reassignment(self):
        point = HRFPoint(
            xyz_mm=(0.0, 0.0, 0.0),
            hrf_mean=np.array([0.0]),
            hrf_std=None,
            sfreq=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            point.sfreq = 100.0  # type: ignore[misc]


class TestHRFPointModalityTag:
    def test_default_tag_is_unknown(self):
        point = HRFPoint(
            xyz_mm=(0.0, 0.0, 0.0),
            hrf_mean=np.array([]),
            hrf_std=None,
            sfreq=1.0,
        )
        assert point.modality_tag == "unknown"

    def test_modality_tag_distinguishes_sources(self):
        """The whole point of the tag: future fMRI consumers should be
        able to route by it without sniffing context fields."""
        fnirs = HRFPoint(
            xyz_mm=(0, 0, 0), hrf_mean=np.array([]), hrf_std=None,
            sfreq=1.0, modality_tag="fnirs",
        )
        fmri = HRFPoint(
            xyz_mm=(0, 0, 0), hrf_mean=np.array([]), hrf_std=None,
            sfreq=1.0, modality_tag="fmri",
        )
        assert fnirs.modality_tag != fmri.modality_tag
