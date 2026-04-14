"""
Targeted unit tests for fix/observer-and-typos.

Scope:
- **3.9**: lens.__init__ initializes self.sfreq (and self.channels) so
  calc_snr can be called directly without first running compare_subject.
- **L1**: "Cannonical" → "Canonical" in hrfunc.py plot titles/labels.
- **L2**: "intiialized" → "initialized" in the tree init warning.
- **L3**: "ommited" → "omitted" in the estimate_hrf edge-expansion warning.

Sweep-adjacent bug fixes (found during lint/type pass, in-scope):
- calc_snr noise_bands mutable default replaced with None sentinel.
- calc_snr psd_noise_slow / psd_noise_fast variable swap: pre-fix read
  PSD from the wrong filtered signal for each band, producing near-zero
  noise power and effectively infinite SNR.
- montage.correlate_canonical now raises a clean ValueError on an
  unconfigured montage instead of AttributeError on self.root.trace.

Fast, no fNIRS data files.
"""

import inspect
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 3.9: lens.__init__ initializes sfreq
# ---------------------------------------------------------------------------

class TestLensSfreqInit:
    def test_lens_has_sfreq_after_construction(self):
        from hrfunc.observer import lens
        observer = lens()
        assert hasattr(observer, 'sfreq')
        assert isinstance(observer.sfreq, (int, float))

    def test_lens_accepts_sfreq_kwarg(self):
        from hrfunc.observer import lens
        observer = lens(sfreq=5.0)
        assert observer.sfreq == 5.0

    def test_lens_has_channels_list(self):
        from hrfunc.observer import lens
        observer = lens()
        assert hasattr(observer, 'channels')
        assert observer.channels == []


# ---------------------------------------------------------------------------
# L1/L2/L3: typos
# ---------------------------------------------------------------------------

class TestTyposFixed:
    def test_cannonical_removed_from_hrfunc(self):
        import hrfunc.hrfunc as mod
        src = inspect.getsource(mod)
        assert 'Cannonical' not in src
        assert 'cannonical' not in src

    def test_intiialized_removed_from_hrtree(self):
        import hrfunc.hrtree as mod
        src = inspect.getsource(mod)
        assert 'intiialized' not in src

    def test_ommited_removed_from_hrfunc(self):
        import hrfunc.hrfunc as mod
        src = inspect.getsource(mod)
        assert 'ommited' not in src
        # Positive: correct spelling is present
        assert 'omitted' in src


# ---------------------------------------------------------------------------
# Sweep bonus: calc_snr mutable default + variable swap
# ---------------------------------------------------------------------------

class TestCalcSnrSweepFixes:
    def test_noise_bands_default_is_none_sentinel(self):
        from hrfunc.observer import lens
        sig = inspect.signature(lens.calc_snr)
        # Pre-fix was a mutable list literal default; post-fix is None
        assert sig.parameters['noise_bands'].default is None

    def test_calc_snr_wires_psd_to_correct_filtered_signal(self):
        """Regression guard for the variable swap bug caught in the
        lint sweep. Source inspection: psd_noise_slow must read from
        preproc_noise_slow, and psd_noise_fast from preproc_noise_fast."""
        from hrfunc.observer import lens
        src = inspect.getsource(lens.calc_snr)
        # Both correct pairings must be present
        assert 'psd_noise_slow = preproc_noise_slow.compute_psd' in src
        assert 'psd_noise_fast = preproc_noise_fast.compute_psd' in src
        # The swapped pre-fix pairings must be gone
        assert 'psd_noise_slow = preproc_noise_fast' not in src
        assert 'psd_noise_fast = preproc_noise_slow' not in src


# ---------------------------------------------------------------------------
# Sweep bonus: correlate_canonical None guard
# ---------------------------------------------------------------------------

class TestCorrelateCanonicalGuard:
    def test_correlate_canonical_raises_on_unconfigured(self):
        from hrfunc.hrfunc import montage
        m = montage()
        with pytest.raises(ValueError, match="configured montage"):
            m.correlate_canonical()

    def test_guard_is_before_root_trace_access(self):
        """The guard must fire before `len(self.root.trace)` which would
        AttributeError on a None self.root."""
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.correlate_canonical)
        guard_idx = src.find('self.root is None')
        access_idx = src.find('len(self.root.trace)')
        assert guard_idx != -1
        assert access_idx != -1
        assert guard_idx < access_idx
