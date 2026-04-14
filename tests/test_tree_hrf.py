"""
Targeted unit tests for fix/tree-hrf-correctness.

Scope:
- **NE-001**: tree.insert labels the CANONICAL SENTINEL node with
  context['method']='canonical', not the first user-inserted root.
- **3.5**: tree.insert jitter branch mutates the HRF's x/y/z directly
  rather than a loop variable that was immediately discarded.
- **NE-003**: HRF.__init__ initializes process_options to [None] so the
  build() zip produces one iteration and spline_interp actually runs.
  spline_interp now accepts an optional trace argument so build() can
  call it through the `process(self.trace)` pipeline pattern.
- **NE-004**: HRF.smooth uses scipy.ndimage.gaussian_filter1d (imported
  at module level) instead of the non-existent self.gaussian_filter1d.
- **3.7**: HRF.__init__ mutable default args (estimates=[], locations=[],
  context=[]) replaced with None sentinels + per-instance materialization.
- **3.10**: HRF.build derives hrf_type from self.oxygenation instead of
  reading the never-set self.type.

Fast, no fNIRS data files.
"""

import pytest
import numpy as np


def _hrf(ch_name, x, y, z, oxygenation=True, trace_len=30, **kwargs):
    from hrfunc.hrtree import HRF
    suffix = 'hbo' if oxygenation else 'hbr'
    return HRF(
        'doi',
        f'{ch_name}_{suffix}',
        30.0,
        7.81,
        np.ones(trace_len),
        location=[x, y, z],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# NE-001: canonical context does not clobber the first real node
# ---------------------------------------------------------------------------

class TestCanonicalContextAssignment:
    def test_first_real_node_context_method_preserved(self):
        from hrfunc.hrtree import tree
        t = tree()
        hrf = _hrf('s1_d1', 0.0, 0.0, 0.0)
        # The HRF arrives with method='toeplitz' from its default context
        assert hrf.context.get('method') == 'toeplitz'
        t.insert(hrf)
        # After insert, the USER's method should still be 'toeplitz'
        # (pre-fix this was clobbered to 'canonical')
        assert t.root.context['method'] == 'toeplitz'

    def test_canonical_sentinel_labeled_canonical(self):
        from hrfunc.hrtree import tree
        t = tree()
        hrf = _hrf('s1_d1', 0.0, 0.0, 0.0)
        t.insert(hrf)
        # root.right is the canonical sentinel inserted by tree.insert
        assert t.root.right is not None
        assert t.root.right.ch_name.startswith('canonical')
        assert t.root.right.context['method'] == 'canonical'


# ---------------------------------------------------------------------------
# 3.5: jitter actually moves the HRF's coordinates
# ---------------------------------------------------------------------------

class TestJitterCoordinates:
    def test_duplicate_insert_jitters_coordinates(self, capsys):
        from hrfunc.hrtree import tree
        t = tree()
        a = _hrf('a', 0.5, 0.5, 0.5)
        b = _hrf('b', 0.5, 0.5, 0.5)
        t.insert(a)
        t.insert(b)
        # Post-fix: b's coordinates should have been jittered by 1e-10
        assert b.x != 0.5 or b.y != 0.5 or b.z != 0.5
        captured = capsys.readouterr()
        assert 'Jittering location' in captured.out

    def test_jitter_preserves_first_node_coordinates(self):
        from hrfunc.hrtree import tree
        t = tree()
        a = _hrf('a', 0.5, 0.5, 0.5)
        b = _hrf('b', 0.5, 0.5, 0.5)
        t.insert(a)
        orig_a_x, orig_a_y, orig_a_z = a.x, a.y, a.z
        t.insert(b)
        # The first-inserted node should NOT be jittered
        assert a.x == orig_a_x
        assert a.y == orig_a_y
        assert a.z == orig_a_z


# ---------------------------------------------------------------------------
# NE-003: HRF.build actually runs spline_interp and resamples
# ---------------------------------------------------------------------------

class TestBuildResamples:
    def test_build_resamples_trace_to_target_length(self):
        from hrfunc.hrtree import HRF
        h = HRF(
            'doi',
            's1_d1_hbo',
            duration=10.0,
            sfreq=5.0,
            trace=np.linspace(0.0, 1.0, 50),
            location=[0, 0, 0],
        )
        # Target: new_sfreq=10 × duration=10 = 100 samples
        h.build(new_sfreq=10.0)
        assert len(h.trace) == 100
        assert h.built is True

    def test_build_resamples_downward(self):
        from hrfunc.hrtree import HRF
        h = HRF(
            'doi',
            's1_d1_hbo',
            duration=10.0,
            sfreq=20.0,
            trace=np.linspace(0.0, 1.0, 200),
            location=[0, 0, 0],
        )
        # Target: 5 × 10 = 50 samples
        h.build(new_sfreq=5.0)
        assert len(h.trace) == 50

    def test_process_options_matches_hrf_processes_length(self):
        from hrfunc.hrtree import HRF
        h = HRF('doi', 's1_d1_hbo', 30.0, 7.81, np.zeros(10), location=[0, 0, 0])
        assert len(h.process_options) == len(h.hrf_processes)
        assert len(h.process_options) == len(h.process_names)


# ---------------------------------------------------------------------------
# NE-004: HRF.smooth actually runs without AttributeError
# ---------------------------------------------------------------------------

class TestSmooth:
    def test_smooth_runs_without_attributeerror(self):
        from hrfunc.hrtree import HRF
        h = HRF('doi', 's1_d1_hbo', 30.0, 7.81, np.ones(100), location=[0, 0, 0])
        # Pre-fix: called self.gaussian_filter1d which was never imported
        h.smooth(2.0)
        assert len(h.trace) == 100

    def test_smooth_actually_smooths(self):
        from hrfunc.hrtree import HRF
        # Sharp step function
        trace = np.concatenate([np.zeros(50), np.ones(50)])
        h = HRF('doi', 's1_d1_hbo', 30.0, 7.81, trace, location=[0, 0, 0])
        h.smooth(5.0)
        # After smoothing, the midpoint should be less extreme than raw
        mid = h.trace[49:51]
        assert 0.0 < mid[0] < 1.0
        assert 0.0 < mid[1] < 1.0


# ---------------------------------------------------------------------------
# 3.7: mutable default args are not shared across instances
# ---------------------------------------------------------------------------

class TestMutableDefaults:
    def test_two_hrfs_have_independent_estimates(self):
        from hrfunc.hrtree import HRF
        a = HRF('doi_a', 'a_hbo', 30.0, 7.81, np.zeros(10), location=[0, 0, 0])
        b = HRF('doi_b', 'b_hbo', 30.0, 7.81, np.zeros(10), location=[1, 1, 1])
        a.estimates.append('from_a')
        assert 'from_a' not in b.estimates

    def test_two_hrfs_have_independent_locations(self):
        from hrfunc.hrtree import HRF
        a = HRF('doi_a', 'a_hbo', 30.0, 7.81, np.zeros(10), location=[0, 0, 0])
        b = HRF('doi_b', 'b_hbo', 30.0, 7.81, np.zeros(10), location=[1, 1, 1])
        a.locations.append([99, 99, 99])
        assert [99, 99, 99] not in b.locations

    def test_two_hrfs_have_independent_contexts(self):
        from hrfunc.hrtree import HRF
        a = HRF('doi_a', 'a_hbo', 30.0, 7.81, np.zeros(10), location=[0, 0, 0])
        b = HRF('doi_b', 'b_hbo', 30.0, 7.81, np.zeros(10), location=[1, 1, 1])
        a.context['task'] = 'flanker'
        assert b.context.get('task') is None


# ---------------------------------------------------------------------------
# 3.10: build's plot path uses oxygenation-derived hrf_type
# ---------------------------------------------------------------------------

class TestBuildTypeField:
    def test_build_no_longer_references_self_type(self):
        """The build method must not read self.type (which was never set).
        Only allowed occurrence is inside a comment explaining the fix."""
        import inspect
        from hrfunc.hrtree import HRF
        src = inspect.getsource(HRF.build)
        # Strip comment lines before searching for the attribute access
        code_lines = [
            line for line in src.splitlines()
            if not line.lstrip().startswith('#')
        ]
        code_only = '\n'.join(code_lines)
        assert 'self.type' not in code_only

    def test_build_uses_hrf_type_helper(self):
        import inspect
        from hrfunc.hrtree import HRF
        src = inspect.getsource(HRF.build)
        assert 'hrf_type' in src
