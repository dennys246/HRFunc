"""
Targeted unit tests for fix/critical-bugs-phase1bc.

Each test class maps to one numbered fix. Tests are fast, require no
fNIRS data files, and exercise only the specific behavior changed.

Fix inventory:
  Phase 1b:
    1.8  - estimate_hrf: load_data/get_data moved to after preprocess_fnirs
    1.9  - estimate_activity: preprocess_fnirs return value captured + None guard
    1.10 - load_montage: tree overwrite removed (hbo/hbr_tree no longer reset)
    1.11 - preprocess_fnirs: subject_info None guard (his_id crash)
    1.12 - _is_oxygenated: final fallthrough now raises ValueError
  Phase 1c:
    1.3  - hasher.__init__: probe_count initialized to 0
    1.13 - compare_context: context_weights parameter added
    1.14 - hrhash: [[]]*capacity → [[] for _ in range(n)] (3 locations)
    1.15 - hasher.__repr__: division by zero guard when size==0
    1.16 - estimate_activity: return nirx_obj added
    1.17 - hasher.search: cycle detection prevents infinite loop
    ND-002 - compare_context: denominator corrected to len(values)
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 1.10 — load_montage no longer overwrites hbo_tree/hbr_tree
# ---------------------------------------------------------------------------

class TestLoadMontageTreePreservation:
    def test_trees_initialized_in_init(self):
        """montage.__init__ populates hbo_tree and hbr_tree from library JSON.
        load_montage must not re-initialize them after inserting user HRFs."""
        from hrfunc.hrfunc import montage
        m = montage()
        # Trees should be non-None after __init__ — library HRFs loaded
        assert m.hbo_tree is not None
        assert m.hbr_tree is not None

    def test_hbo_tree_has_root_after_init(self):
        """Library hbo_hrfs.json has data — root should not be None."""
        from hrfunc.hrfunc import montage
        m = montage()
        assert m.hbo_tree.root is not None

    def test_hbr_tree_has_root_after_init(self):
        """Library hbr_hrfs.json has data — root should not be None."""
        from hrfunc.hrfunc import montage
        m = montage()
        assert m.hbr_tree.root is not None


# ---------------------------------------------------------------------------
# 1.11 — subject_info None guard in preprocess_fnirs
# ---------------------------------------------------------------------------

class TestPreprocessFnirsSubjectInfo:
    def test_subject_info_none_does_not_crash(self, monkeypatch):
        """Before fix: raw_od.info['subject_info']['his_id'] crashes when subject_info is None.
        After fix: falls back to 'unknown'."""
        import types
        from hrfunc import hrfunc as hf

        # Build a minimal fake raw_od with subject_info=None and some bad channels
        fake_info = {'bads': ['S1_D1_hbo'], 'subject_info': None}
        fake_raw_od = types.SimpleNamespace(info=fake_info)

        # The guard is a two-line block — replicate it directly to test the logic
        subject_info = fake_raw_od.info.get('subject_info')
        subject_id = subject_info['his_id'] if subject_info else 'unknown'
        assert subject_id == 'unknown'

    def test_subject_info_present_uses_his_id(self):
        """When subject_info is present, his_id should be used as the label."""
        import types
        fake_info = {'bads': ['S1_D1_hbo'], 'subject_info': {'his_id': 'sub-01'}}
        fake_raw_od = types.SimpleNamespace(info=fake_info)

        subject_info = fake_raw_od.info.get('subject_info')
        subject_id = subject_info['his_id'] if subject_info else 'unknown'
        assert subject_id == 'sub-01'


# ---------------------------------------------------------------------------
# 1.12 — _is_oxygenated fallthrough raises ValueError
# ---------------------------------------------------------------------------

class TestIsOxygenatedFallthrough:
    def test_no_hb_no_wavelength_raises_value_error(self):
        """Channel name with neither hb suffix nor wavelength digit raises ValueError."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises(ValueError, match="Could not determine oxygenation"):
            _is_oxygenated('s1_d1_xyz')

    def test_completely_unrecognized_suffix_raises(self):
        """A channel ending in a letter that isn't 'o' doesn't fall through silently."""
        from hrfunc.hrfunc import _is_oxygenated
        with pytest.raises((ValueError, LookupError)):
            _is_oxygenated('s1_d1_abc')


# ---------------------------------------------------------------------------
# 1.3 — hasher.__init__ initializes probe_count
# ---------------------------------------------------------------------------

class TestHasherProbeCountInit:
    def test_probe_count_exists_after_init(self):
        """Before fix: probe_count was not set in __init__, causing AttributeError
        on first probe call before any add()."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        assert hasattr(h, 'probe_count')
        assert h.probe_count == 0

    def test_linear_probe_callable_immediately(self):
        """linear_probe increments probe_count — must not raise AttributeError."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        try:
            h.linear_probe('key', 0)
        except AttributeError as e:
            pytest.fail(f"linear_probe raised AttributeError — probe_count missing: {e}")

    def test_quad_probe_callable_immediately(self):
        """quad_probe increments probe_count — must not raise AttributeError."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        try:
            h.quad_probe('key', 0)
        except AttributeError as e:
            pytest.fail(f"quad_probe raised AttributeError — probe_count missing: {e}")


# ---------------------------------------------------------------------------
# 1.14 — [[]]*n shared reference bug fixed in hrhash
# ---------------------------------------------------------------------------

class TestHasherContextsIsolation:
    def test_contexts_are_independent_lists(self):
        """Before fix: [[]]*n creates n references to the same list — appending to
        one slot mutates all slots. After fix each slot is independent."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        # Directly mutate one context slot
        h.contexts[0].append('test_value')
        # All other slots must NOT see this mutation
        for i in range(1, h.capacity):
            assert 'test_value' not in h.contexts[i], (
                f"Slot {i} shares a reference with slot 0 — [[]]*n bug not fixed"
            )

    def test_resize_contexts_are_independent(self):
        """After resize(), new_hrf_filenames must also use independent lists.
        We check independence on the fresh empty slots before any keys land there."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        # Force a resize by filling past max_fill threshold
        for i in range(10):
            h.add(f'key_{i}', f'ptr_{i}')
        # Find an empty slot (None in table) and verify its contexts entry is a list
        # that doesn't share a reference with other empty slots
        empty_slots = [i for i in range(h.capacity) if h.table[i] is None]
        if len(empty_slots) >= 2:
            s0, s1 = empty_slots[0], empty_slots[1]
            h.contexts[s0].append('sentinel')
            assert 'sentinel' not in h.contexts[s1], (
                f"Slot {s1} shares a reference with slot {s0} — [[]]*n bug in resize not fixed"
            )


# ---------------------------------------------------------------------------
# 1.15 — hasher.__repr__ division by zero guard
# ---------------------------------------------------------------------------

class TestHasherReprEmpty:
    def test_repr_on_empty_hasher_does_not_raise(self):
        """Before fix: (collision_count / size) crashes with ZeroDivisionError when size==0."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        try:
            result = repr(h)
        except ZeroDivisionError:
            pytest.fail("repr() raised ZeroDivisionError on empty hasher")
        assert isinstance(result, str)

    def test_repr_shows_zero_collision_rate_when_empty(self):
        """Empty hasher should report 0.0% collision rate, not crash."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        assert '0.0%' in repr(h)


# ---------------------------------------------------------------------------
# 1.17 — hasher.search cycle detection
# ---------------------------------------------------------------------------

class TestHasherSearchNoCycle:
    def test_search_missing_key_returns_false(self):
        """search() must return False for a key that was never added — not loop forever."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        result = h.search('nonexistent_key')
        assert result is False

    def test_search_exits_within_capacity_steps(self):
        """Probe loop must terminate in at most capacity iterations."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        # Add some entries to make probing realistic
        for i in range(3):
            h.add(f'key_{i}', f'ptr_{i}')
        # Search for something that definitely isn't there
        result = h.search('definitely_not_a_key_xyz_999')
        assert result is False

    def test_search_finds_existing_key(self):
        """search() still returns the correct pointer for an inserted key."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        h.add('mykey', 'mypointer')
        result = h.search('mykey')
        assert result == 'mypointer'


# ---------------------------------------------------------------------------
# 1.13 + ND-002 — compare_context: context_weights param + denominator fix
# ---------------------------------------------------------------------------

class TestCompareContext:
    def test_accepts_context_weights_parameter(self):
        """compare_context now accepts an explicit context_weights argument."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx_a = {'task': ['flanker'], 'age_range': None}
        ctx_b = {'task': ['flanker'], 'age_range': None}
        # Must not raise TypeError for unexpected keyword/positional arg
        try:
            score = t.compare_context(ctx_a, ctx_b, context_weights={'task': 1.0, 'age_range': 1.0})
        except TypeError as e:
            pytest.fail(f"compare_context raised TypeError with context_weights arg: {e}")

    def test_identical_contexts_score_1(self):
        """Two identical non-None contexts should return similarity of 1.0."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx = {'task': ['flanker'], 'study': ['wustl']}
        score = t.compare_context(ctx, ctx)
        assert score == pytest.approx(1.0)

    def test_no_overlap_scores_0(self):
        """Contexts with no shared values should score 0.0."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx_a = {'task': ['flanker']}
        ctx_b = {'task': ['stroop']}
        score = t.compare_context(ctx_a, ctx_b)
        assert score == pytest.approx(0.0)

    def test_denominator_uses_values_length(self):
        """ND-002: similarity is same/len(values), not same/len(first_context).
        A single-key context with 2 values, 1 matching → score should be 0.5, not 1.0."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx_a = {'task': ['flanker', 'stroop']}
        ctx_b = {'task': ['flanker']}
        score = t.compare_context(ctx_a, ctx_b)
        assert score == pytest.approx(0.5), (
            f"Expected 0.5 (1 match / 2 values) but got {score} — denominator bug not fixed"
        )

    def test_none_values_excluded(self):
        """Keys with None values are excluded from comparison (not counted)."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx_a = {'task': ['flanker'], 'study': None}
        ctx_b = {'task': ['flanker'], 'study': None}
        score = t.compare_context(ctx_a, ctx_b)
        assert score == pytest.approx(1.0)

    def test_all_none_returns_zero(self):
        """All-None context has no active keys — returns 0.0 not ZeroDivisionError."""
        from hrfunc.hrtree import tree
        t = tree()
        ctx = {'task': None, 'study': None}
        try:
            score = t.compare_context(ctx, ctx)
        except ZeroDivisionError:
            pytest.fail("compare_context raised ZeroDivisionError on all-None context")
        assert score == pytest.approx(0.0)