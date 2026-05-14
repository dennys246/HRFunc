"""Unit tests for :mod:`hrfunc.spatial.coords`.

The conversion helpers are tiny but load-bearing: every spatial-layer
call site relies on them to bridge between MNE's internal meter
representation and the MNI-mm convention used by the spatial /
viz / atlas layers. A regression here would cascade into wrong
shape-membership results and wrong atlas labels.
"""

from __future__ import annotations

import numpy as np

from hrfunc.spatial.coords import meters_to_mm, mm_to_meters


class TestMetersToMm:
    def test_scalar_input(self):
        assert float(meters_to_mm(0.05)) == 50.0

    def test_list_input(self):
        result = meters_to_mm([0.01, 0.02, 0.03])
        np.testing.assert_array_almost_equal(result, [10.0, 20.0, 30.0])

    def test_tuple_input(self):
        result = meters_to_mm((0.07, -0.05, 0.0))
        np.testing.assert_array_almost_equal(result, [70.0, -50.0, 0.0])

    def test_array_input(self):
        arr = np.array([[0.01, 0.02], [0.03, 0.04]])
        result = meters_to_mm(arr)
        np.testing.assert_array_almost_equal(result, [[10.0, 20.0], [30.0, 40.0]])

    def test_returns_float64(self):
        result = meters_to_mm([1, 2, 3])
        assert result.dtype == np.float64


class TestMmToMeters:
    def test_scalar_input(self):
        assert float(mm_to_meters(50.0)) == 0.05

    def test_list_input(self):
        result = mm_to_meters([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(result, [0.01, 0.02, 0.03])


class TestRoundTrip:
    def test_meters_round_trip(self):
        """m → mm → m must be exact for head-scale magnitudes."""
        original = np.array([0.01, 0.05, -0.07, 0.123456])
        result = mm_to_meters(meters_to_mm(original))
        np.testing.assert_array_equal(result, original)

    def test_mm_round_trip(self):
        original = np.array([10.0, 50.0, -70.0, 123.456])
        result = meters_to_mm(mm_to_meters(original))
        np.testing.assert_array_equal(result, original)
