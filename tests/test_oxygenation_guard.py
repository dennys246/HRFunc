"""
Targeted unit tests for fix/oxygenation-guard.

Background: M1 in fix/input-validation guarded standardize_name against
short / non-str inputs, but the sibling helper _is_oxygenated is called
directly from ~11 call sites across hrfunc.py and hrtree.py without going
through standardize_name. A degenerate channel name (empty string, None,
1-char) crashed _is_oxygenated with a cryptic IndexError instead of
raising a clean ValueError naming the bad input.

This branch adds the same entry-point guard to _is_oxygenated so both
helpers in _utils.py share one consistent input-validation story. No
call-site changes are required.

Fast, no fNIRS data files.
"""

import pytest


# ---------------------------------------------------------------------------
# _is_oxygenated entry guards
# ---------------------------------------------------------------------------

class TestIsOxygenatedTypeGuard:
    def test_none_raises_typeerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(TypeError, match="expected a str"):
            _is_oxygenated(None)

    def test_int_raises_typeerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(TypeError, match="expected a str"):
            _is_oxygenated(5)

    def test_bytes_raises_typeerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(TypeError, match="expected a str"):
            _is_oxygenated(b"s1_d1_hbo")


class TestIsOxygenatedLengthGuard:
    def test_empty_string_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="too short"):
            _is_oxygenated("")

    def test_one_char_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="too short"):
            _is_oxygenated("a")

    def test_two_char_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="too short"):
            _is_oxygenated("hb")


class TestIsOxygenatedValidInputs:
    def test_hbo_suffix_returns_true(self):
        from hrfunc._utils import _is_oxygenated
        assert _is_oxygenated("s1_d1_hbo") is True

    def test_hbr_suffix_returns_false(self):
        from hrfunc._utils import _is_oxygenated
        assert _is_oxygenated("s1_d1_hbr") is False

    def test_wavelength_760_returns_false(self):
        from hrfunc._utils import _is_oxygenated
        assert _is_oxygenated("s1_d1_760") is False

    def test_wavelength_850_returns_true(self):
        from hrfunc._utils import _is_oxygenated
        assert _is_oxygenated("s1_d1_850") is True

    def test_unrecognized_suffix_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError):
            _is_oxygenated("s1_d1_xyz")


class TestIsOxygenatedBStructureGuard:
    """3+ char strings whose second-to-last character is 'b' but which
    contain no 'hb' substring (or end in 'hb' with nothing after) used to
    crash split[1][0] with IndexError. These must now raise ValueError."""

    def test_three_char_abc_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="'hb' oxygenation suffix"):
            _is_oxygenated("abc")

    def test_three_char_abb_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="'hb' oxygenation suffix"):
            _is_oxygenated("abb")

    def test_three_char_bbb_raises_valueerror(self):
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError, match="'hb' oxygenation suffix"):
            _is_oxygenated("bbb")

    def test_ends_in_hb_with_empty_tail_raises_valueerror(self):
        """'xhb' matches [-2]=='h' so it actually bypasses the b-branch
        and falls to the final else. But 'xxhb' has [-2]='h' too. The
        real trap is a trailing 'hb' in certain positions: 'xhbhb' →
        [-2]='h'. Hard to construct a natural 'split[1] empty' case,
        but a direct split('hb') on 'ahb' gives ['a',''] — ensure the
        empty-tail check is exercised."""
        # 'ahb': [-2]='h' so it skips the b-branch entirely. That's fine.
        # But if we *do* enter the b-branch with 'hb' at end, that's via
        # a ch_name where [-2]=='b' and 'hb' is present ending at [-2:].
        # E.g. 'hbb' → split('hb')=['', 'b'], split[1]='b', [0]='b' →
        # falls to ValueError in the existing "could not be determined"
        # branch. No crash. So this edge is already covered by the
        # existing suffix-check logic; the structure guard is defensive.
        from hrfunc._utils import _is_oxygenated
        with pytest.raises(ValueError):
            _is_oxygenated("hbb")


# ---------------------------------------------------------------------------
# Regression: configure path no longer crashes with IndexError on degenerate
# channel names. It should raise a clean ValueError / TypeError.
# ---------------------------------------------------------------------------

class TestConfigureCallSiteBehavior:
    """The real trigger for this branch was M6 test dev hitting an
    IndexError from inside _is_oxygenated during a configure() call on a
    fake raw with empty-string channel names. Now the same path raises a
    clean ValueError."""

    def test_configure_on_empty_channel_name_raises_value_or_type_error(self):
        from hrfunc.hrfunc import montage

        class _FakeInfo(dict):
            pass

        class _FakeRaw:
            def __init__(self):
                self.ch_names = ['']
                info = _FakeInfo()
                info['sfreq'] = 7.81
                info['chs'] = [{'ch_name': '', 'loc': [0] * 12}]
                self.info = info

        m = montage()
        with pytest.raises((ValueError, TypeError)):
            m.configure(_FakeRaw())

    def test_load_montage_on_short_channel_key_raises_valueerror(self, tmp_path):
        """load_montage iterates JSON keys and extracts ch_name via
        '-'.join(...) then calls _is_oxygenated on it. A key whose
        ch_name component is too short should raise a clean ValueError
        wrapped by M5's per-entry exception handler."""
        import json
        from hrfunc.hrfunc import load_montage

        entry = {
            "hrf_mean": [0.0, 0.1, 0.2],
            "hrf_std": [0.0, 0.0, 0.0],
            "sfreq": 7.81,
            "location": [0.01, 0.02, 0.03],
            "context": {"duration": 30.0},
            "estimates": [],
            "locations": [],
        }
        # Key 'ab-doi' → ch_name='ab' (len 2) → too short for _is_oxygenated.
        # _montage.hbo_channels comprehension at load_montage top runs
        # _is_oxygenated on 'ab' before the per-entry loop even starts.
        payload = {"ab-doi": entry}
        path = tmp_path / "m.json"
        path.write_text(json.dumps(payload))
        with pytest.raises((ValueError, TypeError)):
            load_montage(str(path))
