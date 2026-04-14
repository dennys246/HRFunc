"""
Targeted unit tests for fix/input-validation (M1, M2a, M2b, M3, H1).

Goal: reject degenerate inputs at API boundaries with clear errors instead
of cryptic crashes deep in the math. Each fix gets at least one passing
input test (validator accepts valid data) and one failing input test
(validator rejects bad data with a clean exception).

No fNIRS data files are required. Montage-level tests construct a bare
montage() so only the top-of-function validators fire — we do not reach
configure() / MNE preprocessing.
"""

import inspect
import json
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# M1: standardize_name short-name guard (_utils.py)
# ---------------------------------------------------------------------------

class TestStandardizeNameGuard:
    def test_valid_hbo_channel_name_passes(self):
        from hrfunc._utils import standardize_name
        assert standardize_name("S1_D1 hbo") == "s1_d1_hbo"

    def test_valid_hbr_channel_name_passes(self):
        from hrfunc._utils import standardize_name
        assert standardize_name("S1_D1 hbr") == "s1_d1_hbr"

    def test_empty_string_raises_valueerror(self):
        from hrfunc._utils import standardize_name
        with pytest.raises(ValueError, match="too short"):
            standardize_name("")

    def test_two_char_name_raises_valueerror(self):
        from hrfunc._utils import standardize_name
        with pytest.raises(ValueError, match="too short"):
            standardize_name("S1")

    def test_non_string_raises_typeerror(self):
        from hrfunc._utils import standardize_name
        with pytest.raises(TypeError):
            standardize_name(None)


# ---------------------------------------------------------------------------
# M2a: estimate_hrf — duration<=0 and empty events
# ---------------------------------------------------------------------------

@pytest.fixture
def bare_montage():
    """A montage with library trees loaded but no nirx_obj. Only top-of-
    function validators run; configure() / preprocess_fnirs are not invoked."""
    from hrfunc.hrfunc import montage
    return montage()


class TestEstimateHrfDurationGuard:
    def test_positive_duration_passes_validator(self, bare_montage):
        # Valid duration + valid events should pass the early validators and
        # then fail later in configure() (no nirx_obj). We catch whatever
        # happens after the validators and only assert it's not our ValueError.
        with pytest.raises(Exception) as exc_info:
            bare_montage.estimate_hrf(None, [1, 0, 0], duration=30.0, lmbda=1e-3)
        assert "duration must be" not in str(exc_info.value)
        assert "events list must not be empty" not in str(exc_info.value)
        assert "lmbda must be" not in str(exc_info.value)

    def test_zero_duration_raises(self, bare_montage):
        with pytest.raises(ValueError, match="duration must be > 0"):
            bare_montage.estimate_hrf(None, [1, 0, 0], duration=0)

    def test_negative_duration_raises(self, bare_montage):
        with pytest.raises(ValueError, match="duration must be > 0"):
            bare_montage.estimate_hrf(None, [1, 0, 0], duration=-5.0)


class TestEstimateHrfEmptyEventsGuard:
    def test_empty_events_raises(self, bare_montage):
        with pytest.raises(ValueError, match="events list must not be empty"):
            bare_montage.estimate_hrf(None, [], duration=30.0)

    def test_non_list_events_raises(self, bare_montage):
        with pytest.raises(ValueError, match="must be of type list"):
            bare_montage.estimate_hrf(None, np.array([1, 0, 0]), duration=30.0)


# ---------------------------------------------------------------------------
# M2b: lmbda<=0 in both estimate_hrf and estimate_activity
# ---------------------------------------------------------------------------

class TestLmbdaGuard:
    def test_estimate_hrf_zero_lmbda_raises(self, bare_montage):
        with pytest.raises(ValueError, match="lmbda must be > 0"):
            bare_montage.estimate_hrf(None, [1, 0, 0], duration=30.0, lmbda=0)

    def test_estimate_hrf_negative_lmbda_raises(self, bare_montage):
        with pytest.raises(ValueError, match="lmbda must be > 0"):
            bare_montage.estimate_hrf(None, [1, 0, 0], duration=30.0, lmbda=-1e-3)

    def test_estimate_activity_zero_lmbda_raises(self, bare_montage):
        with pytest.raises(ValueError, match="lmbda must be > 0"):
            bare_montage.estimate_activity(None, lmbda=0)

    def test_estimate_activity_negative_lmbda_raises(self, bare_montage):
        with pytest.raises(ValueError, match="lmbda must be > 0"):
            bare_montage.estimate_activity(None, lmbda=-1e-4)

    def test_estimate_activity_positive_lmbda_passes_validator(self, bare_montage):
        with pytest.raises(Exception) as exc_info:
            bare_montage.estimate_activity(None, lmbda=1e-4)
        assert "lmbda must be" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# M3: load_montage JSON schema validation
# ---------------------------------------------------------------------------

def _minimal_entry(duration=30.0, sfreq=7.81):
    return {
        "hrf_mean": [0.0, 0.1, 0.2, 0.1, 0.0],
        "hrf_std": [0.0, 0.0, 0.0, 0.0, 0.0],
        "sfreq": sfreq,
        "location": [0.01, 0.02, 0.03],
        "context": {"duration": duration},
        "estimates": [],
        "locations": [],
    }


class TestLoadMontageSchemaValidation:
    def test_valid_entry_loads_successfully(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.0/doi": _minimal_entry()}))
        m = load_montage(str(path))
        assert m is not None

    def test_missing_hrf_mean_raises_valueerror(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        entry = _minimal_entry()
        del entry["hrf_mean"]
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.0/doi": entry}))
        with pytest.raises(ValueError, match="hrf_mean"):
            load_montage(str(path))

    def test_missing_sfreq_raises_valueerror(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        entry = _minimal_entry()
        del entry["sfreq"]
        path = tmp_path / "m.json"
        # first_hrf read at top of load_montage also touches sfreq; a missing
        # sfreq should surface a clean KeyError or our ValueError. We assert
        # the per-entry validator is the one that speaks when we get past the
        # top read by giving the first entry sfreq and corrupting a second.
        good = {"S1_D1 hbo-10.0/doi": _minimal_entry()}
        good["S1_D2 hbo-10.0/doi"] = entry
        path.write_text(json.dumps(good))
        with pytest.raises(ValueError, match="sfreq"):
            load_montage(str(path))

    def test_missing_location_raises_valueerror(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        entry = _minimal_entry()
        del entry["location"]
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.0/doi": entry}))
        with pytest.raises(ValueError, match="location"):
            load_montage(str(path))

    def test_missing_context_duration_raises_valueerror(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        entry = _minimal_entry()
        entry["context"] = {}  # empty context, no duration
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.0/doi": entry}))
        with pytest.raises(ValueError, match="context.duration"):
            load_montage(str(path))

    def test_nondict_entry_raises_valueerror(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        path = tmp_path / "m.json"
        # first entry valid (to satisfy top-of-function reads), second broken
        payload = {
            "S1_D1 hbo-10.0/doi": _minimal_entry(),
            "S1_D2 hbo-10.0/doi": "not-a-dict",
        }
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_montage(str(path))


# ---------------------------------------------------------------------------
# H1: HRF zero-trace guard in estimate_activity deconvolution closure
# ---------------------------------------------------------------------------

class TestZeroTraceGuard:
    """The pre-fix estimate_activity closure did
        hrf_kernel = hrf.trace / np.max(np.abs(hrf.trace))
    which produces NaN when hrf.trace is all zeros. H1 adds a guard that
    detects None / empty / all-zero traces before the division and falls
    back to the canonical HRF, emitting a warning."""

    def test_all_zero_trace_would_produce_nan_without_guard(self):
        """Sanity check — demonstrates the silent-NaN failure mode the
        guard exists to prevent."""
        trace = np.zeros(10)
        with np.errstate(invalid='ignore', divide='ignore'):
            kernel = trace / np.max(np.abs(trace))
        assert np.all(np.isnan(kernel))

    def test_guard_present_in_source(self):
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        assert "trace_invalid" in src, (
            "H1 guard missing: estimate_activity must detect degenerate HRF "
            "traces before dividing by max(abs(trace))"
        )
        assert "np.max(np.abs(hrf.trace)) == 0" in src
        assert "hrf.trace is None" in src

    def test_guard_falls_back_to_canonical(self):
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        # When trace_invalid, the outer loop should take the same branch
        # hrf_model == 'canonical' takes (swap in hbo_tree.root.right /
        # hbr_tree.root.right). Confirm the condition is wired up.
        assert "hrf_model == 'canonical' or trace_invalid" in src

    def test_guard_emits_warning(self):
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        assert "empty or all-zero" in src
