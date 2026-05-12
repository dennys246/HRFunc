"""Targeted unit tests for feat/gui-library-browser (v1.3.0 Sprint 4.2-4.4).

Covers:

- AppState additions (library_hbo / library_hbr / library_filter /
  library_selected_hrf) — defaults, reset behavior.
- ``apply_filter`` — empty filter passes through, substring match,
  case-insensitivity, list-valued context fields, missing-key exclusion.
- ``gather_library_hrfs`` — combines HbO + HbR, handles None tree.
- ``build_plotly_figure`` — produces HbO + HbR traces, customdata is
  the HRF key (for click handling), missing/short locations are skipped.
- ``_extract_clicked_hrf_key`` — pulls key from plotly event payload.
- /library page render — toolbar visible, filter inputs present,
  empty-data fallback (no library loaded).
"""

from __future__ import annotations

import contextlib
import io as _io
from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.pages import library  # noqa: E402
from hrfunc.gui.state import AppState, state as global_state  # noqa: E402

gui_app._register_pages()


def _silent(fn, *args, **kwargs):
    """Run a callable while swallowing its stdout chatter (tree.gather is loud)."""
    with contextlib.redirect_stdout(_io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# State lifecycle
# ---------------------------------------------------------------------------


class TestStateLibraryFields:
    def test_defaults(self):
        s = AppState()
        assert s.library_hbo is None
        assert s.library_hbr is None
        assert s.library_filter == {}
        assert s.library_selected_hrf is None

    def test_reset_clears_filter_and_selection(self):
        s = AppState()
        s.library_filter["task"] = "flanker"
        s.library_selected_hrf = {"_key": "x"}
        s.reset()
        assert s.library_filter == {}
        assert s.library_selected_hrf is None

    def test_reset_does_not_clear_loaded_trees(self):
        """Bundled HRF trees are immutable on-disk data; re-loading on
        every dataset switch would burn 100+ ms with no benefit."""
        s = AppState()
        sentinel = object()
        s.library_hbo = sentinel
        s.library_hbr = sentinel
        s.reset()
        assert s.library_hbo is sentinel
        assert s.library_hbr is sentinel


# ---------------------------------------------------------------------------
# apply_filter — context substring matching
# ---------------------------------------------------------------------------


class TestApplyFilter:
    def _fake_hrfs(self):
        return {
            "h1": {
                "context": {"task": "flanker", "doi": "doi/A", "demographics": "children"},
            },
            "h2": {
                "context": {"task": "rest", "doi": "doi/B", "demographics": "adults"},
            },
            "h3": {
                "context": {"task": "flanker", "doi": "doi/C", "demographics": None},
            },
        }

    def test_empty_filter_returns_all(self):
        hrfs = self._fake_hrfs()
        assert library.apply_filter(hrfs, {}) == hrfs

    def test_match_one_field(self):
        hrfs = self._fake_hrfs()
        result = library.apply_filter(hrfs, {"task": "flanker"})
        assert set(result.keys()) == {"h1", "h3"}

    def test_case_insensitive(self):
        hrfs = self._fake_hrfs()
        assert (
            set(library.apply_filter(hrfs, {"task": "FLANKER"}).keys())
            == set(library.apply_filter(hrfs, {"task": "flanker"}).keys())
        )

    def test_substring_match(self):
        hrfs = self._fake_hrfs()
        # "doi/A" should be found by "doi/" but also by just "A"
        result = library.apply_filter(hrfs, {"doi": "doi/A"})
        assert set(result.keys()) == {"h1"}

    def test_and_across_keys(self):
        hrfs = self._fake_hrfs()
        result = library.apply_filter(
            hrfs, {"task": "flanker", "demographics": "children"}
        )
        # h1 matches both; h3 has flanker but demographics=None → excluded
        assert set(result.keys()) == {"h1"}

    def test_missing_context_key_excludes(self):
        hrfs = {"h1": {"context": {"task": "flanker"}}}
        result = library.apply_filter(hrfs, {"doi": "X"})
        assert result == {}

    def test_list_context_value_any_match(self):
        hrfs = {
            "h1": {"context": {"conditions": ["congruent", "incongruent"]}},
            "h2": {"context": {"conditions": ["rest"]}},
        }
        result = library.apply_filter(hrfs, {"conditions": "congruent"})
        assert set(result.keys()) == {"h1"}


# ---------------------------------------------------------------------------
# gather_library_hrfs
# ---------------------------------------------------------------------------


class TestGatherLibraryHrfs:
    def test_empty_when_trees_none(self):
        s = AppState()
        assert library.gather_library_hrfs(s) == {}

    def test_combines_real_bundled_trees(self):
        s = AppState()
        _silent(library._load_trees, s)
        all_hrfs = _silent(library.gather_library_hrfs, s)
        # Sanity: the bundled databases ship with HRFs; expect at least 1 HbO + 1 HbR
        assert len(all_hrfs) > 0
        # Confirm at least one HbO and one HbR
        oxys = {hrf.get("oxygenation") for hrf in all_hrfs.values()}
        assert True in oxys
        assert False in oxys

    def test_excludes_global_sentinel_entries(self):
        """Regression: globals at sentinel location ~[360, 360, 360] were
        dragging plotly's aspectmode=data axis range out 5000x and
        compressing the real 0.07m HRF cluster to a single invisible
        pixel. The user reported 'I only see 2 globals on the screen'.
        gather_library_hrfs now skips any entry whose tree key starts
        with 'global_'."""
        s = AppState()
        fake_hbo = type("FakeTree", (), {
            "root": True,
            "gather": lambda self, root: {
                "s1_d1_hbo-temp": {"oxygenation": True, "location": [0.05, 0.05, 0.05]},
                "global_hbo-temp": {"oxygenation": True, "location": [360, 360, 360]},
            },
        })()
        s.library_hbo = fake_hbo
        s.library_hbr = None

        result = library.gather_library_hrfs(s)
        keys = list(result.keys())
        assert any("s1_d1_hbo" in k for k in keys), "real HRF should be present"
        assert not any("global_" in k for k in keys), \
            "global sentinel entries must be filtered out"
        # And no entries with sentinel-scale locations
        for hrf in result.values():
            loc = hrf.get("location") or []
            if len(loc) >= 3:
                assert max(abs(c) for c in loc[:3]) < 1.0, \
                    f"location {loc} looks like a sentinel; should have been filtered"

    def test_namespaces_prefix_preserves_cross_file_collisions(self):
        """Regression: the bundled HbO and HbR JSONs share at least one key
        (e.g. ``s8_d4_hbr-temp`` appears in both). The previous plain
        dict.update silently dropped one of the duplicates. Re-keying
        with ``hbo:`` / ``hbr:`` prefixes preserves both."""
        s = AppState()
        shared = {"oxygenation": True, "location": [0.01, 0.02, 0.03]}
        fake_hbo = type("FakeTree", (), {
            "root": True,
            "gather": lambda self, root: {"shared-key": shared},
        })()
        fake_hbr = type("FakeTree", (), {
            "root": True,
            "gather": lambda self, root: {
                "shared-key": {**shared, "oxygenation": False},
            },
        })()
        s.library_hbo = fake_hbo
        s.library_hbr = fake_hbr

        result = library.gather_library_hrfs(s)
        # Both copies should survive, distinguished by namespace prefix
        assert "hbo:shared-key" in result
        assert "hbr:shared-key" in result
        assert result["hbo:shared-key"]["oxygenation"] is True
        assert result["hbr:shared-key"]["oxygenation"] is False

    def test_bundled_library_no_sentinel_locations_reach_viz(self):
        """End-to-end: after gather_library_hrfs filters globals, every
        surviving HRF has an MNI-scale location (< 1 meter magnitude).
        plotly's aspectmode=data won't blow up the axis range."""
        s = AppState()
        _silent(library._load_trees, s)
        all_hrfs = _silent(library.gather_library_hrfs, s)
        for key, hrf in all_hrfs.items():
            loc = hrf.get("location") or []
            if len(loc) >= 3:
                max_coord = max(abs(c) for c in loc[:3])
                assert max_coord < 1.0, (
                    f"HRF {key} has location {loc} with coord {max_coord} m — "
                    f"sentinel locations should have been filtered, real "
                    f"optode locations should be on head scale (~0.1 m)."
                )


# ---------------------------------------------------------------------------
# build_plotly_figure
# ---------------------------------------------------------------------------


class TestBuildPlotlyFigure:
    def test_two_traces_for_mixed_oxygenation(self):
        hrfs = {
            "h1": {"location": [1, 2, 3], "oxygenation": True, "context": {}},
            "h2": {"location": [4, 5, 6], "oxygenation": False, "context": {}},
        }
        fig = library.build_plotly_figure(hrfs)
        assert len(fig.data) == 2
        names = {trace.name for trace in fig.data}
        assert names == {"HbO", "HbR"}

    def test_customdata_carries_keys_for_click(self):
        hrfs = {
            "alpha": {"location": [0, 0, 0], "oxygenation": True, "context": {}},
            "beta": {"location": [1, 1, 1], "oxygenation": True, "context": {}},
        }
        fig = library.build_plotly_figure(hrfs)
        hbo_trace = next(t for t in fig.data if t.name == "HbO")
        assert list(hbo_trace.customdata) == ["alpha", "beta"]

    def test_skips_missing_location(self):
        hrfs = {
            "good": {"location": [0, 1, 2], "oxygenation": True, "context": {}},
            "no_loc": {"oxygenation": True, "context": {}},
            "short": {"location": [0, 1], "oxygenation": True, "context": {}},
        }
        fig = library.build_plotly_figure(hrfs)
        hbo_trace = next(t for t in fig.data if t.name == "HbO")
        assert list(hbo_trace.customdata) == ["good"]

    def test_empty_input_produces_zero_traces(self):
        fig = library.build_plotly_figure({})
        assert len(fig.data) == 0


# ---------------------------------------------------------------------------
# MNI brain overlay
# ---------------------------------------------------------------------------


class TestBrainMeshLoader:
    """The bundled fsaverage pial mesh ships in `hrfunc.assets` as a .npz so
    no fsaverage download is required at runtime. Verify the asset loads,
    contains the expected arrays in the right shapes, and is cached on
    second call."""

    def test_load_returns_vertices_and_faces_arrays(self):
        # Clear any cached state from prior tests
        library._BRAIN_MESH = None
        result = library.load_brain_mesh()
        assert result is not None
        verts, faces = result
        # decimate_surface(quadric, n_triangles=2500 per hemisphere) → ~2500
        # verts + 5000 faces total after stitching the two hemispheres.
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        # Reasonable bounds — should be tens of thousands of triangles max.
        assert 1_000 < verts.shape[0] < 50_000
        assert 1_000 < faces.shape[0] < 100_000

    def test_vertex_coords_in_mni_meter_scale(self):
        """Bundled mesh is converted from mm → m during the build script so
        it overlays directly on bundled HRF locations (which are in meters
        too). If a future build accidentally ships mm-scale verts, plotly's
        aspectmode=data would blow up the axis range — exactly the bug that
        had us showing only 2 globals in PR #38."""
        library._BRAIN_MESH = None
        verts, _ = library.load_brain_mesh()
        # Human head fits in a ~0.2m cube; mm-scale would blow this past 100.
        max_abs = float(abs(verts).max())
        assert max_abs < 1.0, f"verts max-abs={max_abs} looks like mm scale"

    def test_load_is_cached_on_second_call(self):
        library._BRAIN_MESH = None
        a = library.load_brain_mesh()
        b = library.load_brain_mesh()
        assert a is b  # same tuple object reused


class TestBrainOverlayInFigure:
    """``build_plotly_figure(hrfs, show_brain=...)`` adds a Mesh3d trace
    only when the toggle is on, and adds it as the FIRST trace so the
    HRF scatter renders on top of the brain surface."""

    def test_no_brain_trace_when_toggle_off(self):
        hrfs = {
            "hbo:a": {"location": [0.01, 0.02, 0.03], "oxygenation": True, "context": {}},
        }
        fig = library.build_plotly_figure(hrfs, show_brain=False)
        # Only HbO trace, no Mesh3d
        assert len(fig.data) == 1
        assert fig.data[0].type == "scatter3d"

    def test_brain_trace_added_first_when_toggle_on(self):
        hrfs = {
            "hbo:a": {"location": [0.01, 0.02, 0.03], "oxygenation": True, "context": {}},
            "hbr:a": {"location": [-0.01, 0.02, 0.03], "oxygenation": False, "context": {}},
        }
        library._BRAIN_MESH = None  # force a fresh load
        fig = library.build_plotly_figure(hrfs, show_brain=True)
        # Mesh first so HRF scatter renders on top of the brain.
        assert len(fig.data) == 3
        assert fig.data[0].type == "mesh3d"
        assert fig.data[0].name == "MNI brain"
        assert fig.data[1].type == "scatter3d"  # HbO
        assert fig.data[2].type == "scatter3d"  # HbR

    def test_brain_trace_hides_from_legend_and_hover(self):
        hrfs = {"hbo:a": {"location": [0.0, 0.0, 0.0], "oxygenation": True, "context": {}}}
        fig = library.build_plotly_figure(hrfs, show_brain=True)
        brain = fig.data[0]
        assert brain.showlegend is False
        assert brain.hoverinfo == "skip"


class TestStateLibraryShowBrain:
    def test_default_off(self):
        s = AppState()
        assert s.library_show_brain is False

    def test_reset_clears_toggle(self):
        s = AppState()
        s.library_show_brain = True
        s.reset()
        assert s.library_show_brain is False


# ---------------------------------------------------------------------------
# Click extraction
# ---------------------------------------------------------------------------


class TestExtractClickedHrfKey:
    def test_returns_customdata_from_first_point(self):
        class _Event:
            args = {"points": [{"customdata": "the_key"}]}
        assert library._extract_clicked_hrf_key(_Event()) == "the_key"

    def test_returns_none_for_empty_points(self):
        class _Event:
            args = {"points": []}
        assert library._extract_clicked_hrf_key(_Event()) is None

    def test_returns_none_for_malformed_event(self):
        class _Event:
            args = None
        assert library._extract_clicked_hrf_key(_Event()) is None


# ---------------------------------------------------------------------------
# /library page render — User fixture
# ---------------------------------------------------------------------------


async def test_library_page_renders_toolbar(user: User):
    global_state.reset()
    # Force-empty trees so we don't load 22 HRFs in the test render
    global_state.library_hbo = type("FakeTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    global_state.library_hbr = global_state.library_hbo
    await user.open("/library")
    await user.should_see("HRF Library")
    await user.should_see("Back to welcome")


async def test_library_page_shows_filter_pane(user: User):
    global_state.reset()
    global_state.library_hbo = type("FakeTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    global_state.library_hbr = global_state.library_hbo
    await user.open("/library")
    await user.should_see("Filter")
    await user.should_see("Narrow the visible HRFs")


async def test_library_page_shows_empty_state_when_no_data(user: User):
    """If both trees yield zero HRFs (or load failed), the center pane
    surfaces a graceful message rather than rendering an empty plot."""
    global_state.reset()
    global_state.library_hbo = type("FakeTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    global_state.library_hbr = global_state.library_hbo
    await user.open("/library")
    await user.should_see("Library trees not loaded")


async def test_library_page_renders_with_real_data(user: User):
    """End-to-end: real bundled HRFs load + filter + viz."""
    global_state.reset()
    _silent(library._load_trees, global_state)
    await user.open("/library")
    # The match-count label below the filter form should report the totals
    await user.should_see("HRFs match")
    # And the HRtree header in the center pane
    await user.should_see("HRtree")


async def test_library_detail_pane_prompt_when_no_selection(user: User):
    global_state.reset()
    global_state.library_hbo = type("FakeTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    global_state.library_hbr = global_state.library_hbo
    await user.open("/library")
    await user.should_see("Click an HRF in the viz to inspect")


async def test_library_render_clears_subscribers(user: User):
    """Repeated page renders should not accumulate dead refreshable
    handles. _render clears state.subscribers at the top, matching
    workspace's pattern."""
    global_state.reset()
    global_state.library_hbo = type("FakeTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    global_state.library_hbr = global_state.library_hbo
    # Pre-load a junk subscriber that would survive a non-clearing render
    leaked_calls = []
    global_state.subscribe(
        "library_filter_changed", lambda _p: leaked_calls.append(1)
    )
    await user.open("/library")
    # After render, the pre-render subscriber should be gone (replaced
    # by the library page's own subscribers).
    global_state.publish("library_filter_changed", {})
    assert leaked_calls == []
    # And the library page's own subscribers are present
    assert "library_filter_changed" in global_state.subscribers
    assert len(global_state.subscribers["library_filter_changed"]) >= 1


async def test_filter_count_annotates_missing_location_hrfs(user: User):
    """The match-count label should make it clear when some matched
    HRFs are excluded from the viz for lacking a 3D location — so the
    user isn't confused by '5 / 22 match' while seeing only 3 points."""
    global_state.reset()

    class _FakeTree:
        root = "non_none"

        def gather(self, root):
            return {
                "with_loc": {
                    "location": [0, 0, 0],
                    "oxygenation": True,
                    "context": {"task": "flanker"},
                },
                "no_loc": {
                    "location": None,
                    "oxygenation": True,
                    "context": {"task": "flanker"},
                },
            }

    global_state.library_hbo = _FakeTree()
    global_state.library_hbr = type("EmptyTree", (), {
        "root": None,
        "gather": lambda self, root: {},
    })()
    await user.open("/library")
    # Both HRFs match the (empty) filter; one lacks location.
    await user.should_see("not visualizable: missing location")
