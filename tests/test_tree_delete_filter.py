"""
Targeted unit tests for fix/tree-delete-filter.

Covers:
- KI-009 / H2: tree._delete_recursive signature mismatch + non-existent
  node.hrf_data access. Rewritten to use a _copy_payload helper and to
  pass the replacement min_node (not coordinates) back through the
  recursive call.
- 3.8: tree.gather must return {} on an empty tree (node=None) instead
  of raising AttributeError. Pulled in from fix/tree-edge-cases to
  unblock the xfailed filter remove-path test.
- M6: montage.configure commit-on-success with tree rollback. On a
  failed _merge_montages, newly-inserted tree nodes must be deleted via
  the now-working tree.delete and scalar/list state restored.

No fNIRS data files required.
"""

import inspect
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hrf(ch_name, x, y, z, oxygenation=True):
    from hrfunc.hrtree import HRF
    suffix = 'hbo' if oxygenation else 'hbr'
    return HRF(
        'doi',
        f'{ch_name}_{suffix}',
        30.0,
        7.81,
        np.zeros(234),
        location=[x, y, z],
        estimates=[],
        locations=[],
        context={},
    )


def _collect_nodes(node, acc=None):
    """Walk the kd-tree and return every reachable node reference (including
    the canonical right-child sentinel)."""
    if acc is None:
        acc = []
    if node is None:
        return acc
    acc.append(node)
    _collect_nodes(node.left, acc)
    _collect_nodes(node.right, acc)
    return acc


# ---------------------------------------------------------------------------
# KI-009: delete signature + payload copy
# ---------------------------------------------------------------------------

class TestDeleteLeaf:
    def test_delete_single_user_node_empties_tree(self):
        """After fix/canonical-hrf-sfreq (S4), tree.insert no longer
        creates a canonical sentinel sibling at root.right. Inserting one
        user HRF sets root=user with no children. Deleting that node
        empties the tree (self.root = None)."""
        from hrfunc.hrtree import tree
        t = tree()
        user = _hrf('ch1', 0.0, 0.0, 0.0)
        t.insert(user)
        assert t.root is user
        t.delete(user)
        assert t.root is None

    def test_delete_leaf_node(self):
        """Insert three well-separated HRFs, delete one of the leaves,
        verify the tree still contains the other two."""
        from hrfunc.hrtree import tree
        t = tree()
        a = _hrf('a', 0.0, 0.0, 0.0)
        b = _hrf('b', -1.0, 0.0, 0.0)
        c = _hrf('c', 1.0, 0.0, 0.0)
        t.insert(a)
        t.insert(b)
        t.insert(c)
        t.delete(c)
        remaining = {n.ch_name for n in _collect_nodes(t.root)}
        assert 'a_hbo' in remaining
        assert 'b_hbo' in remaining
        assert 'c_hbo' not in remaining


class TestDeleteWithChildren:
    def test_delete_node_with_right_child(self):
        """Delete a node that has a right subtree. The payload-copy path
        replaces the node in-place with the axis-min from its right subtree."""
        from hrfunc.hrtree import tree
        t = tree()
        a = _hrf('a', 0.0, 0.0, 0.0)
        b = _hrf('b', 1.0, 0.0, 0.0)
        c = _hrf('c', 2.0, 0.0, 0.0)
        t.insert(a)
        t.insert(b)
        t.insert(c)
        t.delete(a)
        remaining = {n.ch_name for n in _collect_nodes(t.root)}
        assert 'b_hbo' in remaining
        assert 'c_hbo' in remaining
        assert 'a_hbo' not in remaining

    def test_delete_root_with_both_subtrees(self):
        from hrfunc.hrtree import tree
        t = tree()
        root = _hrf('root', 5.0, 5.0, 5.0)
        left = _hrf('left', 2.0, 2.0, 2.0)
        right = _hrf('right', 8.0, 8.0, 8.0)
        t.insert(root)
        t.insert(left)
        t.insert(right)
        t.delete(root)
        remaining = {n.ch_name for n in _collect_nodes(t.root)}
        assert 'left_hbo' in remaining
        assert 'right_hbo' in remaining
        assert 'root_hbo' not in remaining

    def test_delete_no_longer_references_hrf_data(self):
        """Regression guard for the pre-fix crash: old _delete_recursive
        touched a non-existent node.hrf_data attribute. The rewrite must
        not re-introduce that reference."""
        from hrfunc.hrtree import tree
        src = inspect.getsource(tree._delete_recursive)
        assert 'hrf_data' not in src


class TestDeleteRecursiveSignature:
    def test_signature_is_three_args(self):
        """Recursive calls must match the 3-arg signature (node, hrf, depth).
        The pre-fix code passed 5 args which would TypeError at runtime."""
        from hrfunc.hrtree import tree
        sig = inspect.signature(tree._delete_recursive)
        # self + node + hrf + depth = 4 parameters
        assert list(sig.parameters.keys()) == ['self', 'node', 'hrf', 'depth']

    def test_copy_payload_helper_exists(self):
        from hrfunc.hrtree import tree
        assert hasattr(tree, '_copy_payload')


class TestCopyPayloadCorrectness:
    """Data-integrity tests for _copy_payload — the helper the delete
    algorithm relies on. Missing fields or aliased arrays would silently
    corrupt kept nodes after a delete cascade."""

    def test_copy_payload_transfers_built_flag(self):
        from hrfunc.hrtree import tree, HRF
        t = tree()
        src = _hrf('src', 0.0, 0.0, 0.0)
        dst = _hrf('dst', 1.0, 1.0, 1.0)
        src.built = True
        dst.built = False
        t._copy_payload(src, dst)
        assert dst.built is True

    def test_copy_payload_does_not_alias_trace(self):
        """dst.trace must be an independent numpy array so in-place
        mutations don't leak across nodes."""
        from hrfunc.hrtree import tree
        t = tree()
        src = _hrf('src', 0.0, 0.0, 0.0)
        dst = _hrf('dst', 1.0, 1.0, 1.0)
        src.trace = np.array([1.0, 2.0, 3.0])
        t._copy_payload(src, dst)
        dst.trace[0] = 999.0
        assert src.trace[0] == 1.0  # src unchanged

    def test_copy_payload_does_not_alias_context(self):
        from hrfunc.hrtree import tree
        t = tree()
        src = _hrf('src', 0.0, 0.0, 0.0)
        dst = _hrf('dst', 1.0, 1.0, 1.0)
        src.context = {'task': 'flanker'}
        t._copy_payload(src, dst)
        dst.context['task'] = 'gonogo'
        assert src.context['task'] == 'flanker'

    def test_copy_payload_leaves_process_lists_on_dst(self):
        """hrf_processes contains bound methods pointing at the owning
        HRF. Copying them cross-references src into dst, which would
        corrupt dst.build() execution. dst must keep its own defaults."""
        from hrfunc.hrtree import tree
        t = tree()
        src = _hrf('src', 0.0, 0.0, 0.0)
        dst = _hrf('dst', 1.0, 1.0, 1.0)
        original_dst_processes = dst.hrf_processes
        t._copy_payload(src, dst)
        assert dst.hrf_processes is original_dst_processes
        # Bound method's owner is still dst
        if dst.hrf_processes:
            bound = dst.hrf_processes[0]
            if hasattr(bound, '__self__'):
                assert bound.__self__ is dst

    def test_copy_payload_leaves_tree_pointers_untouched(self):
        from hrfunc.hrtree import tree
        t = tree()
        src = _hrf('src', 0.0, 0.0, 0.0)
        dst = _hrf('dst', 1.0, 1.0, 1.0)
        sentinel_left = object()
        sentinel_right = object()
        dst.left = sentinel_left
        dst.right = sentinel_right
        t._copy_payload(src, dst)
        assert dst.left is sentinel_left
        assert dst.right is sentinel_right


# ---------------------------------------------------------------------------
# 3.8: gather on empty tree
# ---------------------------------------------------------------------------

class TestGatherEmptyTree:
    def test_gather_on_none_returns_empty_dict(self):
        from hrfunc.hrtree import tree
        t = tree()
        assert t.gather(None) == {}

    def test_gather_after_filter_empties_tree(self, monkeypatch):
        """End-to-end: the previously-xfailed filter remove-path test
        lives in tests/test_phase1a.py; this is a parallel sanity check
        that filter + gather survives on a fully-emptied tree."""
        from hrfunc.hrtree import tree
        t = tree()
        t.insert(_hrf('a', 0.0, 0.0, 0.0))
        t.branched = True
        monkeypatch.setattr(t, 'compare_context', lambda *a, **k: 0.5)
        t.filter(similarity_threshold=0.95)
        assert t.gather(t.root) == {}


# ---------------------------------------------------------------------------
# M6: configure commit-on-success with tree rollback
# ---------------------------------------------------------------------------

class _FakeInfo(dict):
    pass


class _FakeRaw:
    """Minimal MNE Raw stand-in for configure tests."""

    def __init__(self, ch_names, sfreq=7.81, raise_on_chs=False):
        self.ch_names = ch_names
        chs = [
            {'ch_name': ch, 'loc': np.array([0.01 * i, 0.02, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
            for i, ch in enumerate(ch_names)
        ]
        info = _FakeInfo()
        info['sfreq'] = sfreq
        if raise_on_chs:
            class _ExplodingList(list):
                def __iter__(self_inner):
                    raise RuntimeError("boom from fake _merge_montages")
            info['chs'] = _ExplodingList(chs)
        else:
            info['chs'] = chs
        self.info = info


class _PartialMergeRaw:
    """A fake raw whose info['chs'] yields N successful channels then
    raises, so _merge_montages partially inserts into the tree before
    failing. Used to exercise the M6 tree-rollback path."""

    def __init__(self, good_ch_names, sfreq=7.81):
        self.ch_names = good_ch_names + ['BOOM hbo']
        self._good_count = len(good_ch_names)
        good_chs = [
            {'ch_name': ch, 'loc': np.array([0.01 * (i + 1), 0.02, 0.03, 0, 0, 0, 0, 0, 0, 0, 0, 0])}
            for i, ch in enumerate(good_ch_names)
        ]
        parent = self

        class _FaultyChs(list):
            def __iter__(self_inner):
                for ch in good_chs:
                    yield ch
                raise RuntimeError(
                    f"simulated _merge_montages failure after "
                    f"{parent._good_count} channels"
                )

        info = _FakeInfo()
        info['sfreq'] = sfreq
        info['chs'] = _FaultyChs()
        self.info = info


class TestConfigureCommitOnSuccess:
    def test_configure_success_path(self):
        from hrfunc.hrfunc import montage
        m = montage()
        raw = _FakeRaw(['S1_D1 hbo', 'S1_D1 hbr'])
        m.configure(raw)
        assert m.configured is True
        assert m.sfreq == 7.81
        assert len(m.channels) == 2

    def test_rollback_before_any_tree_mutation(self):
        """Failure during the ch_name list comprehension (raise_on_chs=True
        simulates _merge_montages raising at the very start of iteration)
        leaves self untouched."""
        from hrfunc.hrfunc import montage
        m = montage()
        raw = _FakeRaw(['S1_D1 hbo', 'S1_D1 hbr'], raise_on_chs=True)
        with pytest.raises(RuntimeError, match="boom"):
            m.configure(raw)
        assert m.configured is False
        assert not hasattr(m, 'sfreq')
        assert not hasattr(m, 'hbo_channels')
        assert not hasattr(m, 'hbr_channels')

    def test_rollback_after_partial_tree_mutation(self):
        """_merge_montages inserts two channels into the tree before
        raising. M6 must delete those inserted nodes via tree.delete so
        the tree goes back to its pre-call state (empty, since this is a
        first-time configure)."""
        from hrfunc.hrfunc import montage
        m = montage()
        raw = _PartialMergeRaw(['S1_D1 hbo', 'S1_D2 hbo'])
        with pytest.raises(RuntimeError, match="simulated"):
            m.configure(raw)
        assert m.configured is False
        assert not hasattr(m, 'sfreq')
        assert len(m.channels) == 0
        # tree.root should be back to None (no user nodes, so nothing
        # should remain on the montage's own kd-tree root)
        assert m.root is None

    def test_rollback_preserves_prior_configured_state(self):
        """Configure once successfully, then re-configure with a fake
        raw that fails mid-merge. The first configure's state must
        survive: sfreq, channel counts, and the tree's user nodes all
        match the post-first-configure snapshot."""
        from hrfunc.hrfunc import montage
        m = montage()
        raw1 = _FakeRaw(['S1_D1 hbo', 'S1_D1 hbr'], sfreq=5.0)
        m.configure(raw1)
        assert m.configured is True
        prev_sfreq = m.sfreq
        prev_channels_snapshot = dict(m.channels)
        prev_node_ids = {id(n) for n in m.channels.values()}

        raw2 = _PartialMergeRaw(['S2_D2 hbo', 'S2_D2 hbr'], sfreq=10.0)
        with pytest.raises(RuntimeError, match="simulated"):
            m.configure(raw2)

        assert m.configured is True
        assert m.sfreq == prev_sfreq
        assert set(m.channels.keys()) == set(prev_channels_snapshot.keys())
        # Every retained channel points to the same node object as before
        for k, v in m.channels.items():
            assert v is prev_channels_snapshot[k]
        # No stray inserted nodes linger in the tree
        remaining_ids = {id(n) for n in _collect_nodes(m.root) if n.ch_name[:9] != 'canonical'}
        assert remaining_ids == prev_node_ids
