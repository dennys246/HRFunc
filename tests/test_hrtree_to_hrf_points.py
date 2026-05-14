"""Tests for the :meth:`hrfunc.hrtree.tree.to_hrf_points` adapter.

The adapter is the bridge between the fNIRS-specific kd-tree and the
modality-agnostic :class:`HRFPoint` DTO consumed by spatial selection
and the GUI's 3D viz. These tests pin down the conversion semantics
so a future fMRI tree class can target the same DTO contract.
"""

from __future__ import annotations

import contextlib
import io as _io

import numpy as np

from hrfunc.hrtree import HRF, tree
from hrfunc.spatial.point import HRFPoint


def _silent(fn, *args, **kwargs):
    """tree internals print on insert; suppress for clean test output."""
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*args, **kwargs)


def _make_tree_with_one_hbo_node():
    t = tree()
    hrf = HRF(
        doi="doi/test",
        ch_name="s1 d1 hbo",
        duration=10.0,
        sfreq=5.0,
        trace=np.array([0.0, 0.5, 1.0, 0.5, 0.0]),
        trace_std=np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        location=[0.01, 0.02, 0.03],   # 10, 20, 30 mm
        context={
            "method": "toeplitz",
            "doi": "doi/test",
            "task": "flanker",
            "duration": 10.0,
        },
    )
    _silent(t.insert, hrf)
    return t


class TestEmptyTree:
    def test_returns_no_points(self):
        t = tree()
        assert list(t.to_hrf_points()) == []


class TestSingleNode:
    def test_yields_one_hrf_point(self):
        t = _make_tree_with_one_hbo_node()
        points = list(_silent(lambda: list(t.to_hrf_points())))
        assert len(points) == 1
        assert isinstance(points[0], HRFPoint)

    def test_coordinates_converted_to_mm(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        np.testing.assert_array_almost_equal(
            points[0].xyz_mm, (10.0, 20.0, 30.0)
        )

    def test_trace_carried_as_ndarray(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        assert isinstance(points[0].hrf_mean, np.ndarray)
        np.testing.assert_array_almost_equal(
            points[0].hrf_mean, [0.0, 0.5, 1.0, 0.5, 0.0]
        )

    def test_std_carried_when_present(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        assert points[0].hrf_std is not None
        np.testing.assert_array_almost_equal(
            points[0].hrf_std, [0.1, 0.1, 0.1, 0.1, 0.1]
        )

    def test_sfreq_carried(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        assert points[0].sfreq == 5.0

    def test_modality_tag_defaults_to_fnirs(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        assert points[0].modality_tag == "fnirs"

    def test_modality_tag_can_be_overridden(self):
        """Reserved for a future fMRI tree class to set its own tag."""
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points(modality_tag="fmri")))
        assert points[0].modality_tag == "fmri"


class TestContextCarriesFnirsFields:
    def test_oxygenation_in_context(self):
        """fNIRS-specific oxygenation flag rides in context so the DTO
        stays modality-agnostic. HbO routes via context, not via a
        first-class field."""
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        # Channel name 's1 d1 hbo' → oxygenation=True
        assert points[0].context.get("oxygenation") is True

    def test_hrf_key_in_context(self):
        """Consumers (e.g. spatial-shape ROI selection) need to map points
        back to tree keys so they can recover the source HRF for averaging
        / display. The adapter stores the key in context."""
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        key = points[0].context.get("hrf_key")
        assert key is not None
        assert "s1" in key or "hbo" in key

    def test_existing_context_preserved(self):
        t = _make_tree_with_one_hbo_node()
        points = _silent(lambda: list(t.to_hrf_points()))
        assert points[0].context.get("task") == "flanker"
        assert points[0].context.get("method") == "toeplitz"
