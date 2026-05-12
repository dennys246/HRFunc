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


class TestMeshLoader:
    """The bundled fsaverage meshes (pial cortical + outer-skin scalp)
    ship in ``hrfunc.assets`` as .npz files so no fsaverage download is
    required at runtime. Both layers are independently togglable."""

    def test_load_pial_returns_arrays(self):
        library._MESH_CACHE.clear()
        result = library.load_mesh("pial")
        assert result is not None
        verts, faces = result
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert 1_000 < verts.shape[0] < 50_000

    def test_load_scalp_returns_arrays(self):
        library._MESH_CACHE.clear()
        result = library.load_mesh("scalp")
        assert result is not None
        verts, faces = result
        assert verts.shape[1] == 3
        assert faces.shape[1] == 3
        assert 1_000 < verts.shape[0] < 50_000

    def test_unknown_layer_returns_none(self):
        library._MESH_CACHE.clear()
        result = library.load_mesh("not-a-real-layer")
        assert result is None

    def test_both_layers_in_mni_meter_scale(self):
        """Same defensive bound that caught the 360-meter globals bug —
        if a future mesh rebuild ships mm-scale verts by mistake,
        plotly's aspectmode=data would blow up the visible range and
        compress the HRF cluster to a single pixel."""
        library._MESH_CACHE.clear()
        for layer in ("pial", "scalp"):
            verts, _ = library.load_mesh(layer)
            max_abs = float(abs(verts).max())
            assert max_abs < 1.0, (
                f"{layer}: verts max-abs={max_abs} looks like mm scale"
            )

    def test_per_layer_caching(self):
        library._MESH_CACHE.clear()
        a = library.load_mesh("pial")
        b = library.load_mesh("pial")
        assert a is b
        scalp = library.load_mesh("scalp")
        assert scalp is not a  # different layer, different object

    def test_load_brain_mesh_alias_points_at_scalp(self):
        """Back-compat shim: callers that still import ``load_brain_mesh``
        now get the scalp layer (which is the user-visible default)."""
        library._MESH_CACHE.clear()
        result = library.load_brain_mesh()
        scalp = library.load_mesh("scalp")
        assert result is scalp


class TestOverlaysInFigure:
    """``build_plotly_figure`` has two independent overlay toggles —
    ``show_brain`` (cortical pial) and ``show_scalp`` (outer-skin
    head). Each adds a Mesh3d trace; the order is scalp first, brain
    second, HRF scatter on top, so the painter-order produces nested
    anatomy with the markers visible above everything."""

    def _hrfs(self):
        return {
            "hbo:a": {"location": [0.01, 0.02, 0.03], "oxygenation": True, "context": {}},
            "hbr:a": {"location": [-0.01, 0.02, 0.03], "oxygenation": False, "context": {}},
        }

    def test_no_overlays_when_both_off(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), show_brain=False, show_scalp=False
        )
        assert len(fig.data) == 2
        assert {t.type for t in fig.data} == {"scatter3d"}

    def test_brain_only(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), show_brain=True, show_scalp=False
        )
        names = [t.name for t in fig.data]
        assert names == ["MNI brain", "HbO", "HbR"]

    def test_scalp_only(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), show_brain=False, show_scalp=True
        )
        names = [t.name for t in fig.data]
        assert names == ["MNI head", "HbO", "HbR"]

    def test_both_overlays_correct_painter_order(self):
        """Scalp drawn first so it's the outermost in painter order;
        brain nests inside; HRF scatter renders on top of both."""
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), show_brain=True, show_scalp=True
        )
        names = [t.name for t in fig.data]
        assert names == ["MNI head", "MNI brain", "HbO", "HbR"]

    def test_overlays_hidden_from_legend_and_hover(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), show_brain=True, show_scalp=True
        )
        for mesh in fig.data[:2]:
            assert mesh.type == "mesh3d"
            assert mesh.showlegend is False
            assert mesh.hoverinfo == "skip"


class TestStateLibraryOverlays:
    def test_both_default_on(self):
        s = AppState()
        assert s.library_show_brain is True
        assert s.library_show_scalp is True

    def test_reset_restores_both_to_default_on(self):
        s = AppState()
        s.library_show_brain = False
        s.library_show_scalp = False
        s.reset()
        assert s.library_show_brain is True
        assert s.library_show_scalp is True


# ---------------------------------------------------------------------------
# Region-of-Interest selection
# ---------------------------------------------------------------------------


class TestComputeRoiKeys:
    """``compute_roi_keys`` is the membership rule for the ROI: the
    anchor's same-oxygenation neighbours within ``radius_m`` plus any
    keys manually added via shift-hover paint."""

    def _hrfs(self):
        return {
            "hbo:a": {"location": [0.00, 0.00, 0.00], "oxygenation": True},
            "hbo:b": {"location": [0.01, 0.00, 0.00], "oxygenation": True},  # 1cm
            "hbo:c": {"location": [0.05, 0.00, 0.00], "oxygenation": True},  # 5cm
            "hbr:d": {"location": [0.00, 0.00, 0.00], "oxygenation": False},  # same loc, wrong oxy
            "hbo:loc_less": {"oxygenation": True},  # no location
        }

    def test_no_anchor_no_painted_returns_empty(self):
        hrfs = self._hrfs()
        keys = library.compute_roi_keys(hrfs, None, 0.02, set())
        assert keys == set()

    def test_anchor_only_within_radius_same_oxygenation(self):
        hrfs = self._hrfs()
        anchor = {**hrfs["hbo:a"], "_key": "hbo:a"}
        keys = library.compute_roi_keys(hrfs, anchor, 0.02, set())
        # 2 cm radius → anchor + 1cm-away `b`. `c` at 5cm is excluded.
        assert keys == {"hbo:a", "hbo:b"}

    def test_excludes_different_oxygenation(self):
        """``hbr:d`` sits at the same xyz as anchor but is HbR — must be
        excluded even though it's distance-0. Averaging HbO with HbR is
        scientifically wrong."""
        hrfs = self._hrfs()
        anchor = {**hrfs["hbo:a"], "_key": "hbo:a"}
        keys = library.compute_roi_keys(hrfs, anchor, 0.02, set())
        assert "hbr:d" not in keys

    def test_widening_radius_picks_up_more(self):
        hrfs = self._hrfs()
        anchor = {**hrfs["hbo:a"], "_key": "hbo:a"}
        # 10 cm radius → all HbO with locations (a, b, c) — not loc_less.
        keys = library.compute_roi_keys(hrfs, anchor, 0.10, set())
        assert keys == {"hbo:a", "hbo:b", "hbo:c"}

    def test_painted_set_unions_into_roi(self):
        hrfs = self._hrfs()
        anchor = {**hrfs["hbo:a"], "_key": "hbo:a"}
        # 0.5cm radius would leave only the anchor; painted "c" widens
        # the ROI manually.
        keys = library.compute_roi_keys(hrfs, anchor, 0.005, {"hbo:c"})
        assert "hbo:c" in keys
        assert "hbo:a" in keys

    def test_painted_filtered_by_anchor_oxygenation(self):
        """Even if the user paints an HbR HRF, anchor=HbO drops it from
        the ROI so the average stays mono-haemoglobin."""
        hrfs = self._hrfs()
        anchor = {**hrfs["hbo:a"], "_key": "hbo:a"}
        keys = library.compute_roi_keys(hrfs, anchor, 0.005, {"hbr:d"})
        assert "hbr:d" not in keys


class TestComputeRoiAverage:
    def test_returns_none_with_fewer_than_two_traces(self):
        hrfs = {
            "a": {"hrf_mean": [1.0, 2.0, 3.0]},
        }
        assert library.compute_roi_average(hrfs, {"a"}) is None

    def test_averages_same_length_traces(self):
        import numpy as np
        hrfs = {
            "a": {"hrf_mean": [1.0, 2.0, 3.0]},
            "b": {"hrf_mean": [3.0, 4.0, 5.0]},
        }
        result = library.compute_roi_average(hrfs, {"a", "b"})
        assert result is not None
        mean, std, n = result
        assert n == 2
        np.testing.assert_array_almost_equal(mean, [2.0, 3.0, 4.0])

    def test_skips_mismatched_length_traces(self):
        hrfs = {
            "a": {"hrf_mean": [1.0, 2.0, 3.0]},
            "b": {"hrf_mean": [1.0, 2.0]},  # different length → skip
            "c": {"hrf_mean": [5.0, 6.0, 7.0]},
        }
        result = library.compute_roi_average(hrfs, {"a", "b", "c"})
        assert result is not None
        _, _, n = result
        assert n == 2  # b was skipped

    def test_skips_missing_hrfs_silently(self):
        hrfs = {
            "a": {"hrf_mean": [1.0, 2.0]},
            "b": {"hrf_mean": [3.0, 4.0]},
        }
        # nonexistent-key in the ROI set should be tolerated
        result = library.compute_roi_average(hrfs, {"a", "b", "ghost"})
        assert result is not None
        _, _, n = result
        assert n == 2


class TestStateLibraryROI:
    def test_radius_default_is_2cm(self):
        s = AppState()
        assert s.library_roi_radius_m == 0.02

    def test_painted_set_defaults_empty(self):
        s = AppState()
        assert s.library_roi_painted == set()

    def test_reset_clears_painted_and_restores_radius(self):
        s = AppState()
        s.library_roi_radius_m = 0.07
        s.library_roi_painted.add("some-key")
        s.reset()
        assert s.library_roi_radius_m == 0.02
        assert s.library_roi_painted == set()


class TestBuildFigureWithRoi:
    def _hrfs(self):
        return {
            "hbo:a": {"location": [0.0, 0.0, 0.0], "oxygenation": True, "context": {}},
            "hbo:b": {"location": [0.01, 0.0, 0.0], "oxygenation": True, "context": {}},
            "hbr:c": {"location": [-0.01, 0.0, 0.0], "oxygenation": False, "context": {}},
        }

    def test_no_roi_keys_adds_no_roi_trace(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(self._hrfs(), roi_keys=None)
        names = [t.name for t in fig.data]
        assert all(not n.startswith("ROI") for n in names)

    def test_roi_trace_added_with_count(self):
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), roi_keys={"hbo:a", "hbo:b"}
        )
        names = [t.name for t in fig.data]
        assert any(n.startswith("ROI") for n in names)
        roi_trace = next(t for t in fig.data if t.name.startswith("ROI"))
        assert len(roi_trace.x) == 2

    def test_roi_trace_is_last_drawn(self):
        """ROI highlight should sit on top of the regular HbO/HbR markers."""
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(
            self._hrfs(), roi_keys={"hbo:a"}
        )
        # ROI trace must be the final element so it paints over scatter.
        assert fig.data[-1].name.startswith("ROI")

    def test_hbo_hbr_distinct_symbols(self):
        """Regression for the co-located HbO+HbR occlusion bug — both
        markers need distinct plotly symbols so they remain visible
        when at the same 3D coordinates."""
        library._MESH_CACHE.clear()
        fig = library.build_plotly_figure(self._hrfs())
        hbo = next(t for t in fig.data if t.name == "HbO")
        hbr = next(t for t in fig.data if t.name == "HbR")
        assert hbo.marker.symbol != hbr.marker.symbol


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
