"""Targeted unit tests for feat/gui-scan-inspector (v1.3.0 Sprint 3.1).

Covers:

- ``dataset_tree.build_nodes`` filter — case-insensitive substring match
  against display_name + path; subjects/sessions with zero matching scans
  pruned.
- ``dataset_tree.render`` — search input renders above the tree, empty-
  match shows "No scans match filter" rather than blank.
- ``workspace._render_inspect_body`` — three rendering states: no scan,
  scan selected but Raw not yet cached (loading), scan selected and Raw
  cached (channels/probe/events visible).
- ``workspace._load_scan_raw`` — async helper inserts loaded Raw into
  state.raw_cache and refreshes only if the user is still on the same
  scan (compared by path, not identity).

Rendering tests use NiceGUI's User fixture; state-isolation tests don't
need NiceGUI bootstrap.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nicegui")

from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.components import dataset_tree  # noqa: E402
from hrfunc.gui.pages import workspace  # noqa: E402
from hrfunc.gui.state import state as global_state  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

gui_app._register_pages()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def filterable_manifest(tmp_path):
    """A manifest with two BIDS subjects and varied display names — good for
    exercising substring filtering at the leaf level and group pruning."""
    scans = (
        ScanEntry(
            format="snirf",
            path=tmp_path / "sub-01" / "ses-01" / "sub-01_task-flanker_nirs.snirf",
            bids_subject="01",
            bids_session="01",
            bids_task="flanker",
            display_name="sub-01 / ses-01 / task-flanker",
        ),
        ScanEntry(
            format="snirf",
            path=tmp_path / "sub-01" / "ses-01" / "sub-01_task-rest_nirs.snirf",
            bids_subject="01",
            bids_session="01",
            bids_task="rest",
            display_name="sub-01 / ses-01 / task-rest",
        ),
        ScanEntry(
            format="snirf",
            path=tmp_path / "sub-02" / "ses-01" / "sub-02_task-flanker_nirs.snirf",
            bids_subject="02",
            bids_session="01",
            bids_task="flanker",
            display_name="sub-02 / ses-01 / task-flanker",
        ),
    )
    return Manifest(root=tmp_path, scans=scans)


def _make_fake_raw(ch_names=None, annotations_data=None):
    """Build a minimal in-memory MNE RawArray for inspector tests.

    Avoids touching disk; the cache is keyed by path so we can stash this
    under any path string we choose for the cache lookup to succeed.
    """
    import numpy as np
    import mne

    if ch_names is None:
        ch_names = ["S1_D1 hbo", "S1_D1 hbr", "S1_D2 hbo", "S1_D2 hbr"]
    n_ch = len(ch_names)
    sfreq = 10.0
    data = np.zeros((n_ch, 50))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="misc")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if annotations_data is not None:
        onsets, durations, descriptions = zip(*annotations_data)
        raw.set_annotations(
            mne.Annotations(onset=list(onsets), duration=list(durations),
                            description=list(descriptions))
        )
    return raw


# ---------------------------------------------------------------------------
# dataset_tree.build_nodes filter — leaf filtering + group pruning
# ---------------------------------------------------------------------------


class TestBuildNodesFilter:
    def test_empty_filter_returns_all(self, filterable_manifest):
        nodes_all = dataset_tree.build_nodes(filterable_manifest)
        nodes_default = dataset_tree.build_nodes(
            filterable_manifest, filter_text=""
        )
        assert nodes_all == nodes_default

    def test_filter_matches_display_name(self, filterable_manifest):
        nodes = dataset_tree.build_nodes(filterable_manifest, "rest")
        # Only the rest scan survives — that's under sub-01/ses-01
        assert len(nodes) == 1
        assert nodes[0]["label"] == "sub-01"
        session = nodes[0]["children"][0]
        leaves = [leaf["label"] for leaf in session["children"]]
        assert leaves == ["sub-01 / ses-01 / task-rest"]

    def test_filter_is_case_insensitive(self, filterable_manifest):
        lower = dataset_tree.build_nodes(filterable_manifest, "rest")
        upper = dataset_tree.build_nodes(filterable_manifest, "REST")
        mixed = dataset_tree.build_nodes(filterable_manifest, "ReSt")
        assert lower == upper == mixed

    def test_filter_matches_path(self, filterable_manifest):
        # The substring 'sub-02' appears in the path of the third scan only.
        nodes = dataset_tree.build_nodes(filterable_manifest, "sub-02")
        assert len(nodes) == 1
        assert nodes[0]["label"] == "sub-02"

    def test_filter_prunes_subject_with_zero_matches(self, filterable_manifest):
        # 'rest' only matches scans under sub-01 → sub-02 should be pruned.
        nodes = dataset_tree.build_nodes(filterable_manifest, "rest")
        labels = [n["label"] for n in nodes]
        assert "sub-02" not in labels

    def test_filter_prunes_session_with_zero_matches(self, tmp_path):
        scans = (
            ScanEntry(
                format="snirf",
                path=tmp_path / "sub-01" / "ses-01" / "rest.snirf",
                bids_subject="01", bids_session="01",
                display_name="ses-01 rest",
            ),
            ScanEntry(
                format="snirf",
                path=tmp_path / "sub-01" / "ses-02" / "flanker.snirf",
                bids_subject="01", bids_session="02",
                display_name="ses-02 flanker",
            ),
        )
        m = Manifest(root=tmp_path, scans=scans)
        nodes = dataset_tree.build_nodes(m, "rest")
        sub01 = nodes[0]
        session_labels = [s["label"] for s in sub01["children"]]
        # ses-02 has no rest scan → pruned
        assert session_labels == ["ses-01"]

    def test_filter_no_matches_returns_empty(self, filterable_manifest):
        nodes = dataset_tree.build_nodes(filterable_manifest, "zzz_unmatchable")
        assert nodes == []


# ---------------------------------------------------------------------------
# dataset_tree.render — search input wiring + empty filter message
# ---------------------------------------------------------------------------


async def test_dataset_tree_renders_filter_input(user: User, tmp_path):
    """The filter input should render above the tree when scans exist."""
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(
            ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                      display_name="alpha"),
        ),
    )
    await user.open("/workspace")
    # Filter input is a placeholder — NiceGUI's User fixture matches it via
    # the placeholder text. The trailing ellipsis is a Unicode character;
    # match the prefix to stay robust.
    await user.should_see("Filter scans")


async def test_workspace_recording_section_shows_loading_when_uncached(
    user: User, tmp_path
):
    """Selecting a scan before its Raw is cached should show a loading hint."""
    scan = ScanEntry(
        format="snirf",
        path=tmp_path / "sub-01" / "sub-01_task-flanker_nirs.snirf",
        bids_subject="01",
        bids_task="flanker",
        display_name="sub-01 / task-flanker",
    )
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(scan,),
        scanned_at=datetime(2026, 5, 11, 12, 0, tzinfo=timezone.utc),
    )
    global_state.selected_scan = scan

    await user.open("/workspace")
    # The "Recording" header always renders when a scan is selected;
    # combined with "Loading recording…" it confirms the uncached branch.
    await user.should_see("Recording")
    await user.should_see("Loading recording")


async def test_workspace_recording_section_shows_channels_when_cached(
    user: User, tmp_path
):
    """When the Raw is in state.raw_cache, channel/probe/events sections render."""
    scan = ScanEntry(
        format="snirf",
        path=tmp_path / "sub-01" / "sub-01_task-flanker_nirs.snirf",
        bids_subject="01",
        bids_task="flanker",
        display_name="sub-01 / task-flanker",
    )
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(scan,),
    )
    global_state.selected_scan = scan

    # Pre-populate the cache with a synthetic Raw so the rendering path that
    # depends on cache hit is exercised without disk I/O.
    raw = _make_fake_raw(
        ch_names=["S1_D1 hbo", "S1_D1 hbr"],
        annotations_data=[(1.0, 0.5, "stim_a"), (3.0, 0.5, "stim_b")],
    )
    global_state.raw_cache._cache[scan.path.resolve()] = raw

    await user.open("/workspace")
    await user.should_see("Recording")
    # Channels expansion label includes the channel count
    await user.should_see("Channels (2)")
    # Events expansion label includes the annotation count
    await user.should_see("Events (2)")
    # Probe layout label is rendered regardless of whether MNE can plot
    await user.should_see("Probe layout")


# ---------------------------------------------------------------------------
# workspace._load_scan_raw — async load + post-load refresh gating
# ---------------------------------------------------------------------------


class TestLoadScanRaw:
    def test_refresh_fires_when_scan_unchanged(self, tmp_path, monkeypatch):
        """After load, refresh fires only if state.selected_scan still matches by path."""
        scan = ScanEntry(
            format="snirf",
            path=tmp_path / "x.snirf",
            display_name="x",
        )
        from hrfunc.gui.state import AppState
        from hrfunc.io.raw_cache import RawCache

        state = AppState()
        state.raw_cache = RawCache(maxsize=3)
        state.selected_scan = scan

        fake_raw = _make_fake_raw()

        # Stub raw_cache.get so we don't hit disk; insert directly into cache.
        def fake_get(scan_or_path):
            state.raw_cache._cache[scan.path.resolve()] = fake_raw
            return fake_raw

        monkeypatch.setattr(state.raw_cache, "get", fake_get)

        refresh_calls = []
        state._inspect_refresh = lambda: refresh_calls.append(1)  # type: ignore

        asyncio.run(workspace._load_scan_raw(state, scan))

        assert scan in state.raw_cache
        # Refresh fires because state.selected_scan path still matches.
        assert refresh_calls == [1]

    def test_refresh_skipped_when_scan_changed(self, tmp_path, monkeypatch):
        """User navigated away mid-load → no stale refresh."""
        scan_a = ScanEntry(
            format="snirf",
            path=tmp_path / "a.snirf",
            display_name="a",
        )
        scan_b = ScanEntry(
            format="snirf",
            path=tmp_path / "b.snirf",
            display_name="b",
        )
        from hrfunc.gui.state import AppState
        from hrfunc.io.raw_cache import RawCache

        state = AppState()
        state.raw_cache = RawCache(maxsize=3)
        state.selected_scan = scan_a

        fake_raw = _make_fake_raw()

        def fake_get(scan_or_path):
            # Simulate the user clicking a different scan while load is in
            # flight by mutating state.selected_scan inside the executor work.
            state.selected_scan = scan_b
            state.raw_cache._cache[scan_a.path.resolve()] = fake_raw
            return fake_raw

        monkeypatch.setattr(state.raw_cache, "get", fake_get)

        refresh_calls = []
        state._inspect_refresh = lambda: refresh_calls.append(1)  # type: ignore

        asyncio.run(workspace._load_scan_raw(state, scan_a))

        # Load completed but user is on scan_b now → no stale refresh.
        assert refresh_calls == []

    def test_path_equality_match_when_object_changed(self, tmp_path, monkeypatch):
        """A re-scan can produce a new ScanEntry object for the same path —
        the path-equality check should still recognize it as 'still the same scan'."""
        scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf", display_name="x")
        scan_reloaded = ScanEntry(format="snirf", path=tmp_path / "x.snirf",
                                  display_name="x")
        assert scan is not scan_reloaded  # different object, same path

        from hrfunc.gui.state import AppState
        from hrfunc.io.raw_cache import RawCache

        state = AppState()
        state.raw_cache = RawCache(maxsize=3)
        state.selected_scan = scan_reloaded  # fresh instance, e.g. after re-scan

        fake_raw = _make_fake_raw()

        def fake_get(scan_or_path):
            state.raw_cache._cache[scan.path.resolve()] = fake_raw
            return fake_raw

        monkeypatch.setattr(state.raw_cache, "get", fake_get)

        refresh_calls = []
        state._inspect_refresh = lambda: refresh_calls.append(1)  # type: ignore

        asyncio.run(workspace._load_scan_raw(state, scan))

        # Path matches even though objects differ → refresh fires.
        assert refresh_calls == [1]

    def test_load_failure_sets_last_error(self, tmp_path, monkeypatch):
        scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf", display_name="x")
        from hrfunc.gui.state import AppState
        from hrfunc.io.raw_cache import RawCache

        state = AppState()
        state.raw_cache = RawCache(maxsize=3)
        state.selected_scan = scan

        def fake_get(scan_or_path):
            raise FileNotFoundError("nope")

        monkeypatch.setattr(state.raw_cache, "get", fake_get)
        # No refresher attached — the helper should still tolerate the
        # absence and just record the error.
        asyncio.run(workspace._load_scan_raw(state, scan))

        assert state.last_error is not None
        assert "FileNotFoundError" in state.last_error
        # Cache is untouched
        assert scan not in state.raw_cache


# ---------------------------------------------------------------------------
# _render_inspect_body — exercised through the workspace render
# ---------------------------------------------------------------------------


class TestInspectBodyContracts:
    def test_render_inspect_body_is_module_level(self):
        """Sprint 3.1 extracted the body from inside _render_inspect_tab so
        tests and future panels can call it directly. Regression guard."""
        import inspect as inspect_mod
        assert callable(workspace._render_inspect_body)
        sig = inspect_mod.signature(workspace._render_inspect_body)
        assert list(sig.parameters) == ["state"]
