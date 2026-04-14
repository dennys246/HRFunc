"""
Targeted unit tests for fix/state-lifecycle (H3, M4, M5).

Goal: montage and tree objects cannot land in a half-configured or
inconsistent state after partial failures. Each fix gets a success-path
test (normal behavior still works) and a failure-path test (state stays
clean when something goes wrong).

M6 (configure commit-on-success) was moved to fix/tree-delete-filter
because proper tree rollback requires a working tree.delete, which is
fixed in that later branch. Dropping M6 here avoids shipping a
rollback that silently leaks orphan nodes into the spatial tree on
re-configure failures.

No fNIRS data files are required.
"""

import json
import inspect
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# H3: montage.__repr__ safe on unconfigured instances
# ---------------------------------------------------------------------------

class TestReprUnconfigured:
    def test_repr_on_fresh_montage_does_not_crash(self):
        from hrfunc.hrfunc import montage
        m = montage()
        # Pre-fix: AttributeError on self.sfreq / self.hbo_channels / self.hbr_channels
        text = repr(m)
        assert "Montage object" in text
        assert "unconfigured" in text

    def test_repr_reports_sfreq_none_when_unset(self):
        from hrfunc.hrfunc import montage
        m = montage()
        text = repr(m)
        assert "Sampling frequency: None" in text

    def test_repr_on_configured_montage_reports_state(self):
        from hrfunc.hrfunc import montage
        m = montage()
        m.sfreq = 7.81
        m.hbo_channels = ['s1_d1_hbo']
        m.hbr_channels = ['s1_d1_hbr']
        m.configured = True
        text = repr(m)
        assert "configured" in text
        assert "7.81" in text
        assert "s1_d1_hbo" in text

    def test_repr_includes_context(self):
        from hrfunc.hrfunc import montage
        m = montage(task='flanker')
        text = repr(m)
        assert "task" in text
        assert "flanker" in text


# ---------------------------------------------------------------------------
# M4: estimate_activity drops orphaned channel entries on failure
# ---------------------------------------------------------------------------

class TestDropChannelOrphan:
    def test_drop_block_sources_present(self):
        """The fix for M4 pops the dropped channel name out of self.channels
        and both hbo/hbr lists. Check the source pattern exists."""
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        assert "dropped_channels" in src
        assert "self.channels.pop" in src
        assert "self.hbo_channels.remove" in src
        assert "self.hbr_channels.remove" in src

    def test_drop_uses_snapshot_iteration(self):
        """Iterating self.channels while popping from it raises
        RuntimeError('dictionary changed size during iteration').
        The fix snapshots the keys via list(...) to avoid this."""
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        assert "list(self.channels.keys())" in src

    def test_post_drop_dict_consistency(self):
        """Simulate the post-drop cleanup: removing ch_names from
        self.channels and the hbo/hbr lists. Sanity check on the dict/list
        ops we rely on."""
        channels = {'a_hbo': 1, 'b_hbo': 2, 'a_hbr': 3}
        hbo = ['a_hbo', 'b_hbo']
        hbr = ['a_hbr']
        dropped = ['a_hbo']
        for ch in dropped:
            channels.pop(ch, None)
            if ch in hbo:
                hbo.remove(ch)
            if ch in hbr:
                hbr.remove(ch)
        assert 'a_hbo' not in channels
        assert hbo == ['b_hbo']
        assert hbr == ['a_hbr']


# ---------------------------------------------------------------------------
# M5: load_montage partial-failure does not return a half-loaded montage
# ---------------------------------------------------------------------------

def _entry(duration=30.0, sfreq=7.81):
    return {
        "hrf_mean": [0.0, 0.1, 0.2, 0.1, 0.0],
        "hrf_std": [0.0, 0.0, 0.0, 0.0, 0.0],
        "sfreq": sfreq,
        "location": [0.01, 0.02, 0.03],
        "context": {"duration": duration},
        "estimates": [],
        "locations": [],
    }


class TestLoadMontagePartialFailure:
    def test_good_file_loads(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        path = tmp_path / "m.json"
        payload = {
            "S1_D1 hbo-10.0/doi": _entry(),
            "S1_D2 hbo-10.0/doi": _entry(),
        }
        path.write_text(json.dumps(payload))
        m = load_montage(str(path))
        assert m is not None

    def test_one_bad_entry_raises_and_montage_not_returned(self, tmp_path):
        """Second entry is malformed (missing hrf_mean). The whole load
        must raise — load_montage returns nothing, so the caller cannot
        hold a half-loaded object."""
        from hrfunc.hrfunc import load_montage
        bad = _entry()
        del bad['hrf_mean']
        payload = {
            "S1_D1 hbo-10.0/doi": _entry(),
            "S1_D2 hbo-10.0/doi": bad,
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        result = None
        with pytest.raises(ValueError) as exc_info:
            result = load_montage(str(path))
        # Error message must name the offending entry
        assert "S1_D2 hbo-10.0/doi" in str(exc_info.value)
        assert "hrf_mean" in str(exc_info.value)
        # result binding was never assigned
        assert result is None

    def test_wrapped_exception_chains_original_cause(self, tmp_path):
        """M5 wraps per-entry failures with `raise ... from exc` so the
        user can still inspect the original exception via __cause__."""
        from hrfunc.hrfunc import load_montage
        bad = _entry()
        bad['hrf_mean'] = "not-an-array"  # will fail np.asarray(float)
        payload = {"S1_D1 hbo-10.0/doi": bad}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError) as exc_info:
            load_montage(str(path))
        assert exc_info.value.__cause__ is not None

    def test_failure_mentions_entry_key_even_for_nonvalidator_errors(self, tmp_path):
        """If the HRF constructor raises (e.g. bad types), M5 still names
        the failing entry so the user knows where to look."""
        from hrfunc.hrfunc import load_montage
        bad = _entry()
        bad['hrf_mean'] = "garbage"
        payload = {"S1_D1 hbo-10.0/doi": bad}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match="S1_D1 hbo-10.0/doi"):
            load_montage(str(path))


# ---------------------------------------------------------------------------
# M4 strengthening: deconvolution closure catches any exception, not just
# TimeoutError, so the post-loop cleanup path always runs
# ---------------------------------------------------------------------------

class TestDeconvolutionExceptionHandling:
    def test_closure_catches_generic_exceptions(self):
        """The deconvolution closure must catch any solve failure, not
        just TimeoutError. Otherwise a LinAlgError (or similar) propagates
        out of estimate_activity and skips the M4 post-loop cleanup."""
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        assert "except TimeoutError" in src
        assert "except Exception" in src

    def test_generic_exception_path_sets_success_false(self):
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.estimate_activity)
        # Find the broadened exception block and confirm it sets success=False
        broad_idx = src.find("except Exception")
        assert broad_idx != -1
        tail = src[broad_idx:broad_idx + 1200]
        assert "success = False" in tail
        assert "deconvolved_signal = nirx" in tail
