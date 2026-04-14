"""
Targeted unit tests for fix/tree-edge-cases.

Scope:
- **NE-006**: tree.merge now inserts node.copy() so the merged tree's
  node structure is fully independent of the source tree. Recursion
  still walks the SOURCE's children so the full subtree is transplanted.
  Includes an empty-source early return.
- **NE-007**: tree.nearest_neighbor has an explicit early return when
  self.root is None, in addition to the recursive base case handling
  that already existed. Keeps the control flow obvious and avoids any
  reliance on falling through to the recursive terminal.

Note: 3.8 (gather None guard) was pulled into fix/tree-delete-filter.

Fast, no fNIRS data files.
"""

import pytest
import numpy as np


def _hrf(ch_name, x, y, z, oxygenation=True):
    from hrfunc.hrtree import HRF
    suffix = 'hbo' if oxygenation else 'hbr'
    return HRF(
        'doi',
        f'{ch_name}_{suffix}',
        30.0,
        7.81,
        np.ones(10),
        location=[x, y, z],
        estimates=[],
        locations=[],
    )


def _collect_nodes(node, acc=None):
    if acc is None:
        acc = []
    if node is None:
        return acc
    acc.append(node)
    _collect_nodes(node.left, acc)
    _collect_nodes(node.right, acc)
    return acc


# ---------------------------------------------------------------------------
# NE-006: tree.merge
# ---------------------------------------------------------------------------

class TestMergeIndependence:
    def test_merge_copies_nodes_so_source_and_dest_are_independent(self):
        from hrfunc.hrtree import tree
        source = tree()
        dest = tree()
        a = _hrf('a', 0.0, 0.0, 0.0)
        b = _hrf('b', 1.0, 0.0, 0.0)
        c = _hrf('c', -1.0, 0.0, 0.0)
        source.insert(a)
        source.insert(b)
        source.insert(c)

        dest.merge(source)

        # Dest should have three nodes with the same ch_names
        dest_names = {n.ch_name for n in _collect_nodes(dest.root)}
        assert 'a_hbo' in dest_names
        assert 'b_hbo' in dest_names
        assert 'c_hbo' in dest_names

        # Mutating the source must NOT affect the dest
        a.trace = np.zeros(10)
        a.trace[5] = 999.0
        # Find the 'a_hbo' node in dest
        dest_nodes = [n for n in _collect_nodes(dest.root) if n.ch_name == 'a_hbo']
        assert len(dest_nodes) == 1
        assert dest_nodes[0].trace[5] != 999.0  # independent copy

    def test_merge_dest_nodes_are_different_objects(self):
        from hrfunc.hrtree import tree
        source = tree()
        dest = tree()
        a = _hrf('a', 0.5, 0.5, 0.5)
        source.insert(a)

        dest.merge(source)

        dest_a = [n for n in _collect_nodes(dest.root) if n.ch_name == 'a_hbo']
        assert len(dest_a) == 1
        assert dest_a[0] is not a  # different Python object

    def test_merge_empty_source_is_noop(self):
        from hrfunc.hrtree import tree
        source = tree()
        dest = tree()
        dest.insert(_hrf('existing', 0.0, 0.0, 0.0))
        assert dest.root is not None

        dest.merge(source)  # source is empty

        # Dest unchanged
        dest_names = {n.ch_name for n in _collect_nodes(dest.root)}
        assert 'existing_hbo' in dest_names
        assert len(dest_names) == 1

    def test_merge_walks_full_source_subtree(self):
        """Ensures the recursion happens on the source's children, not
        the empty-children copy. Five-node tree must transplant all
        five nodes into dest."""
        from hrfunc.hrtree import tree
        source = tree()
        positions = [
            ('a', 0.0, 0.0, 0.0),
            ('b', 1.0, 0.0, 0.0),
            ('c', -1.0, 0.0, 0.0),
            ('d', 0.5, 0.5, 0.5),
            ('e', -0.5, -0.5, -0.5),
        ]
        for name, x, y, z in positions:
            source.insert(_hrf(name, x, y, z))

        dest = tree()
        dest.merge(source)

        dest_names = {n.ch_name for n in _collect_nodes(dest.root)}
        for name, _, _, _ in positions:
            assert f'{name}_hbo' in dest_names


# ---------------------------------------------------------------------------
# Cross-branch audit fix: load_montage must pass channel['context'] through
# to the HRF constructor so compare_context / filter can match on real values
# ---------------------------------------------------------------------------

class TestLoadMontageContextRoundTrip:
    def test_loaded_hrf_carries_channel_context(self, tmp_path):
        """Pre-fix: load_montage constructed HRFs without passing
        channel['context'], so every loaded node fell back to the
        default template context. The hasher was populated correctly
        but the node itself carried no task/stimulus metadata, so
        downstream compare_context / branch / filter would silently
        fail to match on real values."""
        import json
        from hrfunc.hrfunc import load_montage
        entry = {
            "hrf_mean": [0.0, 0.1, 0.2, 0.1, 0.0],
            "hrf_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "sfreq": 7.81,
            "location": [0.01, 0.02, 0.03],
            "context": {
                "task": "flanker",
                "stimulus": "checkerboard",
                "doi": "10.1000/test",
                "duration": 30.0,
                "study": "my_study",
            },
            "estimates": [],
            "locations": [],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.1000/test": entry}))
        m = load_montage(str(path))

        # The inserted HRF must carry the channel's real context, not
        # the default template. Grab the loaded node via self.channels.
        assert len(m.channels) == 1
        node = next(iter(m.channels.values()))
        assert node.context.get('task') == 'flanker'
        assert node.context.get('stimulus') == 'checkerboard'
        assert node.context.get('study') == 'my_study'

    def test_branch_matches_loaded_context(self, tmp_path):
        """End-to-end: after load_montage, calling tree.branch on the
        tree should find the loaded node by its real context value."""
        import json
        from hrfunc.hrfunc import load_montage
        entry = {
            "hrf_mean": [0.0, 0.1, 0.2, 0.1, 0.0],
            "hrf_std": [0.0, 0.0, 0.0, 0.0, 0.0],
            "sfreq": 7.81,
            "location": [0.01, 0.02, 0.03],
            "context": {
                "task": "flanker",
                "duration": 30.0,
            },
            "estimates": [],
            "locations": [],
        }
        path = tmp_path / "m.json"
        path.write_text(json.dumps({"S1_D1 hbo-10.1000/test": entry}))
        m = load_montage(str(path))

        # Branching the HbO tree on task='flanker' should match the
        # loaded node (hasher populated by NE-002 + context now carried
        # on the node itself).
        branched = m.hbo_tree.branch(task='flanker')
        user_nodes = [
            n for n in _collect_nodes(branched.root)
            if not n.ch_name.startswith('canonical')
        ]
        assert len(user_nodes) >= 1


# ---------------------------------------------------------------------------
# NE-007: tree.nearest_neighbor empty tree
# ---------------------------------------------------------------------------

class TestNearestNeighborEmptyTree:
    def test_empty_tree_returns_none_inf(self):
        from hrfunc.hrtree import tree
        t = tree()
        probe = _hrf('probe', 0.0, 0.0, 0.0)
        result, distance = t.nearest_neighbor(probe, max_distance=0.01)
        assert result is None
        assert distance == float('inf')

    def test_empty_tree_early_return_does_not_touch_self_root(self):
        """Regression: the empty-tree path must not try to dereference
        self.root (which is None). The early-return guard makes this
        explicit."""
        from hrfunc.hrtree import tree
        t = tree()
        probe = _hrf('probe', 0.0, 0.0, 0.0)
        # Just confirm it doesn't raise
        t.nearest_neighbor(probe, max_distance=0.01)

    def test_single_node_tree_far_probe_returns_none(self):
        from hrfunc.hrtree import tree
        t = tree()
        t.insert(_hrf('a', 0.0, 0.0, 0.0))
        far_probe = _hrf('probe', 100.0, 100.0, 100.0)
        result, distance = t.nearest_neighbor(far_probe, max_distance=0.01)
        assert result is None

    def test_single_node_tree_near_probe_returns_node(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = _hrf('a', 0.0, 0.0, 0.0)
        t.insert(a)
        near_probe = _hrf('probe', 1e-6, 1e-6, 1e-6)
        result, distance = t.nearest_neighbor(near_probe, max_distance=0.01)
        assert result is a
        assert distance < 0.01
