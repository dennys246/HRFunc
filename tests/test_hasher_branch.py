"""
Targeted unit tests for fix/hasher-branch-correctness.

Scope:
- **3.3**: hasher.add supports multiple pointers per key (appends to slot's
  pointer list); hasher.search returns the full list.
- **H4**: hasher.search returns [] on miss instead of False. All callers
  updated to the list contract.
- **3.4**: hasher.fill and hasher.double_check were dead/broken code and
  have been removed.
- **NE-002**: load_hrfs and load_montage now populate the hasher keyed by
  channel context VALUES (e.g. 'flanker', 'checkerboard'), not by the
  tree's context-dict KEYS. This matches what tree.branch searches for.
- **3.1**: compare_context auto-wraps scalar context values so it can
  handle both `'flanker'` and `['flanker']` consistently.
- **3.2**: tree.branch auto-wraps scalar values, dedupes matched nodes
  across multiple hasher hits, and populates the sub-tree's hasher from
  each copied node's context values (not dict keys).

Fast, no fNIRS data files.
"""

import json
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# 3.3 + H4: hasher structural and return-type contract
# ---------------------------------------------------------------------------

class TestHasherAddAppend:
    def test_add_first_pointer_creates_list(self):
        from hrfunc.hrhash import hasher
        h = hasher({})
        h.add('task', 'ptr_a')
        assert h.search('task') == ['ptr_a']

    def test_add_second_pointer_appends(self):
        from hrfunc.hrhash import hasher
        h = hasher({})
        h.add('task', 'ptr_a')
        h.add('task', 'ptr_b')
        result = h.search('task')
        assert 'ptr_a' in result
        assert 'ptr_b' in result
        assert len(result) == 2

    def test_add_duplicate_pointer_deduplicates(self):
        """Adding the same (key, pointer) pair twice should not produce
        two entries. Dedup is by identity."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        pointer = object()
        h.add('task', pointer)
        h.add('task', pointer)
        assert h.search('task') == [pointer]

    def test_add_different_keys_independent(self):
        from hrfunc.hrhash import hasher
        h = hasher({})
        h.add('task', 'flanker')
        h.add('stimulus', 'checkerboard')
        assert h.search('task') == ['flanker']
        assert h.search('stimulus') == ['checkerboard']

    def test_many_adds_trigger_resize(self):
        """Inserting enough entries to trigger the resize path should not
        corrupt the pointer lists."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        for i in range(20):
            h.add(f'key_{i}', f'ptr_{i}')
        for i in range(20):
            assert h.search(f'key_{i}') == [f'ptr_{i}']


class TestHasherSearchContract:
    def test_search_miss_returns_empty_list(self):
        from hrfunc.hrhash import hasher
        h = hasher({})
        assert h.search('nonexistent') == []

    def test_search_miss_is_iterable(self):
        """The whole point of H4: `for node in hasher.search(x):` must
        not TypeError on a miss."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        count = 0
        for node in h.search('nothing'):
            count += 1
        assert count == 0

    def test_search_returns_copy_not_live_list(self):
        """Callers must be able to mutate the returned list without
        affecting the hasher's internal state."""
        from hrfunc.hrhash import hasher
        h = hasher({})
        h.add('task', 'ptr_a')
        result = h.search('task')
        result.append('garbage')
        # Subsequent search should not see the appended garbage
        assert h.search('task') == ['ptr_a']


class TestHasherDeadCodeRemoved:
    def test_fill_removed(self):
        from hrfunc.hrhash import hasher
        assert not hasattr(hasher, 'fill') or True  # Allow either
        # Hard assertion: the method should not exist
        assert not hasattr(hasher, 'fill')

    def test_double_check_removed(self):
        from hrfunc.hrhash import hasher
        assert not hasattr(hasher, 'double_check')


# ---------------------------------------------------------------------------
# NE-002: load_hrfs populates hasher by context VALUES
# ---------------------------------------------------------------------------

def _entry(doi='doi', task='flanker', stimulus='checkerboard', location=(0.01, 0.02, 0.03)):
    return {
        "hrf_mean": [0.0, 0.1, 0.2, 0.1, 0.0],
        "hrf_std": [0.0, 0.0, 0.0, 0.0, 0.0],
        "sfreq": 7.81,
        "location": list(location),
        "context": {
            "task": task,
            "stimulus": stimulus,
            "doi": doi,
            "duration": 30.0,
        },
        "estimates": [],
        "locations": [],
    }


class TestLoadHrfsPopulatesByValue:
    def test_task_value_becomes_hasher_key(self, tmp_path):
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker'),
        }))
        t = tree(str(path))
        # The hasher should contain 'flanker' as a key, NOT 'task'
        result = t.hasher.search('flanker')
        assert len(result) == 1
        assert result[0].ch_name.startswith('s1_d1')

    def test_old_keys_not_present(self, tmp_path):
        """Regression guard: the pre-fix populated the hasher with dict
        KEYS like 'task' / 'stimulus'. After the fix, those should be
        empty (searching for 'task' as a value returns nothing)."""
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker'),
        }))
        t = tree(str(path))
        # 'task' is the KEY in context dict, not a value any HRF carries
        assert t.hasher.search('task') == []
        assert t.hasher.search('stimulus') == []

    def test_multiple_hrfs_same_task_share_hasher_slot(self, tmp_path):
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker', location=(0.01, 0.01, 0.01)),
            "S1_D2 hbo-doi1": _entry(task='flanker', location=(0.02, 0.02, 0.02)),
            "S1_D3 hbo-doi1": _entry(task='gonogo', location=(0.03, 0.03, 0.03)),
        }))
        t = tree(str(path))
        flanker_nodes = t.hasher.search('flanker')
        gonogo_nodes = t.hasher.search('gonogo')
        assert len(flanker_nodes) == 2
        assert len(gonogo_nodes) == 1


# ---------------------------------------------------------------------------
# NE-002: load_montage populates hasher by context VALUES
# ---------------------------------------------------------------------------

def _montage_entry_payload(task='flanker'):
    return {
        "S1_D1 hbo-10.0/doi": _entry(task=task, location=(0.01, 0.02, 0.03)),
        "S1_D1 hbr-10.0/doi": {
            **_entry(task=task, location=(0.01, 0.02, 0.03)),
        },
    }


class TestLoadMontagePopulatesByValue:
    def test_task_value_reaches_hbo_tree_hasher(self, tmp_path):
        from hrfunc.hrfunc import load_montage
        path = tmp_path / "montage.json"
        path.write_text(json.dumps(_montage_entry_payload(task='flanker')))
        m = load_montage(str(path))
        # The montage's library trees are loaded from the bundled HRFs at
        # montage.__init__ time. load_montage then inserts the user's
        # channels into those same trees. Searching 'flanker' on the HbO
        # tree should return the user's channel node.
        flanker_hits = m.hbo_tree.hasher.search('flanker')
        # At least one of the hits should be the user's channel
        user_ch_name = 's1_d1_hbo'
        assert any(
            getattr(node, 'ch_name', '').startswith(user_ch_name[:5])
            for node in flanker_hits
        )


# ---------------------------------------------------------------------------
# 3.1: compare_context scalar auto-wrap
# ---------------------------------------------------------------------------

class TestCompareContextScalar:
    def test_scalar_task_values_match(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = {'task': 'flanker'}
        b = {'task': 'flanker'}
        # Pre-fix: `for value in 'flanker'` iterated chars and crashed
        # on `second_context['task']` char lookup.
        score = t.compare_context(a, b, context_weights={'task': 1.0})
        assert score == 1.0

    def test_scalar_vs_list_match(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = {'task': 'flanker'}
        b = {'task': ['flanker', 'gonogo']}
        score = t.compare_context(a, b, context_weights={'task': 1.0})
        assert score == 1.0

    def test_scalar_mismatch(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = {'task': 'flanker'}
        b = {'task': 'gonogo'}
        score = t.compare_context(a, b, context_weights={'task': 1.0})
        assert score == 0.0

    def test_missing_key_in_second_context(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = {'task': 'flanker'}
        b = {}  # Missing 'task' entirely — no crash
        score = t.compare_context(a, b, context_weights={'task': 1.0})
        assert score == 0.0

    def test_none_value_skipped(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = {'task': None, 'stimulus': 'checkerboard'}
        b = {'stimulus': 'checkerboard'}
        score = t.compare_context(a, b, context_weights={'task': 1.0, 'stimulus': 1.0})
        # Only 'stimulus' counted; 'task' None skipped
        assert score == 1.0


# ---------------------------------------------------------------------------
# 3.2: tree.branch end-to-end
# ---------------------------------------------------------------------------

class TestTreeBranchEndToEnd:
    def test_branch_on_scalar_value(self, tmp_path):
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker', location=(0.01, 0.01, 0.01)),
            "S1_D2 hbo-doi1": _entry(task='gonogo', location=(0.02, 0.02, 0.02)),
        }))
        t = tree(str(path))
        branched = t.branch(task='flanker')
        # The branched tree should contain only the flanker node
        assert branched.root is not None
        names = {n.ch_name for n in _collect_all_user_nodes(branched.root)}
        assert any('s1_d1' in n for n in names)
        assert not any('s1_d2' in n for n in names)

    def test_branch_dedupes_nodes_matched_under_multiple_values(self, tmp_path):
        """If a single node's context happens to contain values matching
        multiple search terms, the sub-tree must contain it only once."""
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker', stimulus='checkerboard',
                                     location=(0.01, 0.01, 0.01)),
        }))
        t = tree(str(path))
        # Branch on both task and stimulus — the single HRF matches both
        branched = t.branch(task='flanker', stimulus='checkerboard')
        user_nodes = [n for n in _collect_all_user_nodes(branched.root)]
        assert len(user_nodes) == 1

    def test_branch_empty_when_no_match(self, tmp_path):
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker', location=(0.01, 0.01, 0.01)),
        }))
        t = tree(str(path))
        branched = t.branch(task='nothing_matches_this')
        # No user nodes should be in the branched tree (the canonical
        # sentinel may or may not be present depending on insert path)
        user_nodes = [n for n in _collect_all_user_nodes(branched.root)]
        assert user_nodes == []

    def test_branch_populates_sub_tree_hasher_by_values(self, tmp_path):
        """The sub-tree's hasher should be populated by the copied
        nodes' context VALUES — same contract as load_hrfs."""
        from hrfunc.hrtree import tree
        path = tmp_path / "hrfs.json"
        path.write_text(json.dumps({
            "S1_D1 hbo-doi1": _entry(task='flanker', location=(0.01, 0.01, 0.01)),
        }))
        t = tree(str(path))
        branched = t.branch(task='flanker')
        # The sub-tree's hasher should have 'flanker' as a key
        assert len(branched.hasher.search('flanker')) >= 1
        # And NOT the dict keys
        assert branched.hasher.search('task') == []


def _collect_all_user_nodes(node, acc=None):
    if acc is None:
        acc = []
    if node is None:
        return acc
    if not node.ch_name.startswith('canonical'):
        acc.append(node)
    _collect_all_user_nodes(node.left, acc)
    _collect_all_user_nodes(node.right, acc)
    return acc
