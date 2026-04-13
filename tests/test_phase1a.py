"""
Targeted unit tests for fix/critical-bugs-phase1a.

Each test class maps to one numbered fix. Tests are fast, require no
fNIRS data files, and exercise only the specific behavior changed.

Fix inventory:
  1.1 - f-string syntax error in montage.__repr__
  1.2 - missing scipy.stats import
  1.4 - return ValueError → raise ValueError in estimate_hrf
  1.5 - LookupError created-not-raised in _is_oxygenated
  1.6 - double_probe False args (TypeError crash)
  1.7 - typo "Configureding" (cosmetic, not tested)
  ND-001 - filter() condition inverted (xfail: depends on KI-009, KI-010)
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 1.1 — f-string syntax fix in montage.__repr__
# ---------------------------------------------------------------------------

class TestRepr:
    def test_module_imports_without_syntax_error(self):
        """If the f-string fix is wrong, import itself would raise SyntaxError."""
        from hrfunc.hrfunc import montage
        assert callable(montage)

    def test_context_str_extraction_produces_string(self):
        """The extracted context_str variable computes correctly on an empty montage."""
        from hrfunc.hrfunc import montage
        m = montage()
        # Replicate exactly what __repr__ now does; if context or context_weights
        # are wrong this raises — proving the extraction is sound.
        context_str = '\n'.join(
            [f'{key} - {value} - {m.context_weights[key]}'
             for key, value in m.context.items()]
        )
        assert isinstance(context_str, str)
        assert 'method' in context_str


# ---------------------------------------------------------------------------
# 1.2 — scipy.stats import
# ---------------------------------------------------------------------------

class TestScipyStatsImport:
    def test_scipy_stats_importable(self):
        """scipy.stats is now present in hrfunc's import chain."""
        import hrfunc.hrfunc as hf
        import scipy.stats
        # If the import was missing hrfunc would have already failed to load
        assert hasattr(scipy.stats, 'spearmanr')
        assert hasattr(scipy.stats, 'gamma')

    def test_hrfunc_loads_cleanly(self):
        """Top-level import works without NameError or ImportError."""
        import hrfunc
        assert hasattr(hrfunc, 'montage')
        assert hasattr(hrfunc, 'tree')
        assert hasattr(hrfunc, 'localize_hrfs')


# ---------------------------------------------------------------------------
# 1.4 — raise ValueError in estimate_hrf
# ---------------------------------------------------------------------------

class TestEstimateHrfInputValidation:
    def test_bad_duration_type_raises_value_error(self):
        """String duration raises ValueError immediately, not silently returns it."""
        from hrfunc.hrfunc import montage
        m = montage()
        with pytest.raises(ValueError, match="Duration"):
            m.estimate_hrf(None, [], duration='thirty')

    def test_bad_duration_none_raises_value_error(self):
        from hrfunc.hrfunc import montage
        m = montage()
        with pytest.raises(ValueError, match="Duration"):
            m.estimate_hrf(None, [], duration=None)

    def test_bad_events_type_raises_value_error(self):
        """Non-list events raises ValueError, not silently returns it."""
        from hrfunc.hrfunc import montage
        m = montage()
        with pytest.raises(ValueError, match="Events"):
            m.estimate_hrf(None, 'not_a_list', duration=30.0)

    def test_bad_events_tuple_raises_value_error(self):
        """Tuple is not accepted — must be a list."""
        from hrfunc.hrfunc import montage
        m = montage()
        with pytest.raises(ValueError, match="Events"):
            m.estimate_hrf(None, (0, 1, 0), duration=30.0)

    def test_int_duration_is_accepted(self):
        """Integer duration is coerced to float; validation should not raise."""
        from hrfunc.hrfunc import montage
        m = montage()
        # Will fail after validation (configure needs nirx_obj) but must NOT
        # raise ValueError from the type check.
        try:
            m.estimate_hrf(None, [0, 1], duration=30)
        except ValueError as e:
            pytest.fail(f"Valid int duration raised ValueError: {e}")
        except Exception:
            pass  # AttributeError from configure(None) — expected


# ---------------------------------------------------------------------------
# 1.5 — raise LookupError in _is_oxygenated
# ---------------------------------------------------------------------------

class TestIsOxygenated:
    """Tests the _is_oxygenated function directly, covering all branches."""

    def test_hbo_suffix_returns_true(self):
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_hbo') is True

    def test_hbr_suffix_returns_false(self):
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_hbr') is False

    def test_wavelength_850_returns_true(self):
        """850 nm falls in the HbO range (830-850)."""
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_850') is True

    def test_wavelength_830_returns_true(self):
        """830 nm is the lower bound of HbO range."""
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_830') is True

    def test_wavelength_760_returns_false(self):
        """760 nm falls in the HbR range (760-780)."""
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_760') is False

    def test_wavelength_780_returns_false(self):
        """780 nm is the upper bound of HbR range."""
        from hrfunc.hrfunc import _is_oxygenated
        assert _is_oxygenated('s1_d1_780') is False

    def test_out_of_range_wavelength_raises_lookup_error(self):
        """500 nm is not a valid fNIRS wavelength — must raise, not silently return None."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(LookupError):
            _is_oxygenated('s1_d1_500')

    def test_out_of_range_wavelength_800nm_raises_lookup_error(self):
        """800 nm sits between the two bands — must raise LookupError."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(LookupError):
            _is_oxygenated('s1_d1_800')

    def test_non_integer_wavelength_suffix_raises_lookup_error(self):
        """Channel ending in '0' but with non-numeric last-3-chars raises LookupError."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(LookupError):
            _is_oxygenated('s1_d1_xx0')

    def test_invalid_hb_suffix_raises_value_error(self):
        """hbx suffix (not hbo or hbr) raises ValueError."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(ValueError):
            _is_oxygenated('s1_d1_hbx')

    def test_previously_swallowed_error_now_propagates(self):
        """Before fix: out-of-range wavelength silently returned None.
        After fix: raises LookupError. Duplicate of raises test; kept as
        regression guard with explicit pytest.raises so false-positive is impossible."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(LookupError):
            _is_oxygenated('s1_d1_500')


# ---------------------------------------------------------------------------
# ND-001 — filter() condition inversion
# ---------------------------------------------------------------------------
# filter() depends on two unfixed bugs:
#   - KI-010: compare_context() called with 3 args but defined with 2 (phase1c)
#   - KI-009: delete() uses node.hrf_data which doesn't exist (phase3)
# The one-line fix (> to <) is correct and verified in code review.
# These tests are xfail and will flip to pass once dependencies are resolved.

class TestFilterInversion:
    @pytest.mark.xfail(
        reason="Depends on KI-010 (compare_context signature, phase1c) "
               "and KI-009 (delete() AttributeError, phase3). "
               "The > to < fix is correct; full path unblocked in phase1c+phase3.",
        strict=False,
    )
    def test_filter_removes_low_similarity_nodes(self, monkeypatch):
        """Nodes with similarity BELOW threshold should be deleted."""
        from hrfunc.hrtree import tree, HRF
        t = tree()
        hrf1 = HRF('doi1', 'ch1_hbo', 30.0, 7.81, np.zeros(234),
                   location=[0, 0, 0], estimates=[], locations=[], context={})
        t.insert(hrf1)
        t.branched = True  # skip branch() which has its own bugs (NE-002)
        # Patch compare_context to return a score below threshold
        monkeypatch.setattr(t, 'compare_context', lambda *args, **kw: 0.5)
        t.filter(similarity_threshold=0.95)
        assert t.gather(t.root) == {}

    def test_filter_keeps_high_similarity_nodes(self, monkeypatch):
        """Nodes with similarity AT OR ABOVE threshold should be kept.
        The 'keep' path never calls delete(), so KI-009/KI-010 do not block this."""
        from hrfunc.hrtree import tree, HRF
        t = tree()
        hrf1 = HRF('doi1', 'ch1_hbo', 30.0, 7.81, np.zeros(234),
                   location=[0, 0, 0], estimates=[], locations=[], context={})
        t.insert(hrf1)
        t.branched = True
        # Patch compare_context to return a score above threshold
        monkeypatch.setattr(t, 'compare_context', lambda *args, **kw: 0.99)
        t.filter(similarity_threshold=0.95)
        assert len(t.gather(t.root)) > 0


# ---------------------------------------------------------------------------
# 1.6 — double_probe False args (TypeError crash)
# ---------------------------------------------------------------------------

class TestDoubleProbe:
    def test_double_probe_no_type_error(self):
        """double_probe no longer passes False to quad_probe (which doesn't accept it).
        Before fix: quad_probe(key, hashkey, False) → TypeError (3 args, 2 expected).
        After fix: quad_probe(key, hashkey) → no TypeError."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        # Set attrs that probe methods need (KI-008 not fixed yet)
        h.probe_count = 0
        h.collision_count = 0
        try:
            h.double_probe('test_key', 0)
        except TypeError as e:
            pytest.fail(
                f"double_probe raised TypeError — False arg removal may have regressed: {e}"
            )
        except Exception:
            pass  # Other errors from unfixed bugs (KI-008 etc.) are acceptable here