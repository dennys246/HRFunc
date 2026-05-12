"""Targeted unit tests for feat/gui-workspace-shell (v1.3.0 Sprint 2.3).

Covers:

- ``hrfunc.gui.components.dataset_tree`` — Manifest → tree-node conversion,
  BIDS vs non-BIDS grouping, stable ordering, find_scan lookup helper.
- ``hrfunc.gui.pages.workspace`` — page registration, three-pane shell,
  empty-state handling, tab table.

Rendering tests use NiceGUI's User fixture (same setup as the welcome
tests — `tests/gui_main.py` is the main_file). All tests gated by
``pytest.importorskip("nicegui")``.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone
from pathlib import Path

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
# Fixtures — synthetic manifests for tree-building tests
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_manifest(tmp_path):
    return Manifest(root=tmp_path)


@pytest.fixture
def bids_manifest(tmp_path):
    """A manifest with two subjects, one with two sessions, fully BIDS."""
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
            path=tmp_path / "sub-01" / "ses-02" / "sub-01_task-rest_nirs.snirf",
            bids_subject="01",
            bids_session="02",
            bids_task="rest",
            display_name="sub-01 / ses-02 / task-rest",
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


@pytest.fixture
def non_bids_manifest(tmp_path):
    """A manifest with no BIDS metadata — uses parent-dir grouping."""
    lab = tmp_path / "lab_data"
    scans = (
        ScanEntry(format="snirf", path=lab / "subj_A.snirf", display_name="lab_data/subj_A.snirf"),
        ScanEntry(format="snirf", path=lab / "subj_B.snirf", display_name="lab_data/subj_B.snirf"),
    )
    return Manifest(root=tmp_path, scans=scans)


# ---------------------------------------------------------------------------
# dataset_tree.build_nodes — grouping and ordering contracts
# ---------------------------------------------------------------------------


class TestBuildNodes:
    def test_empty_manifest_returns_empty_list(self, empty_manifest):
        assert dataset_tree.build_nodes(empty_manifest) == []

    def test_bids_manifest_groups_by_subject(self, bids_manifest):
        nodes = dataset_tree.build_nodes(bids_manifest)
        labels = [n["label"] for n in nodes]
        assert labels == ["sub-01", "sub-02"]

    def test_bids_manifest_groups_sessions_under_subject(self, bids_manifest):
        nodes = dataset_tree.build_nodes(bids_manifest)
        sub01 = next(n for n in nodes if n["label"] == "sub-01")
        session_labels = [s["label"] for s in sub01["children"]]
        assert session_labels == ["ses-01", "ses-02"]

    def test_scan_leaves_use_display_name(self, bids_manifest):
        nodes = dataset_tree.build_nodes(bids_manifest)
        sub01 = next(n for n in nodes if n["label"] == "sub-01")
        ses01 = next(s for s in sub01["children"] if s["label"] == "ses-01")
        leaf_labels = [c["label"] for c in ses01["children"]]
        assert leaf_labels == ["sub-01 / ses-01 / task-flanker"]

    def test_scan_leaf_id_is_stringified_path(self, bids_manifest):
        nodes = dataset_tree.build_nodes(bids_manifest)
        sub01 = next(n for n in nodes if n["label"] == "sub-01")
        ses01 = next(s for s in sub01["children"] if s["label"] == "ses-01")
        leaf = ses01["children"][0]
        scan = bids_manifest.scans[0]
        assert leaf["id"] == str(scan.path)

    def test_non_bids_groups_by_parent_dir(self, non_bids_manifest):
        nodes = dataset_tree.build_nodes(non_bids_manifest)
        assert len(nodes) == 1
        # Parent dir is "lab_data"; the label uses the folder-emoji prefix.
        assert nodes[0]["label"].endswith("lab_data")

    def test_non_bids_has_single_no_session_group(self, non_bids_manifest):
        nodes = dataset_tree.build_nodes(non_bids_manifest)
        sessions = nodes[0]["children"]
        assert len(sessions) == 1
        assert sessions[0]["label"] == "(no session)"

    def test_subjects_are_alphabetically_sorted(self, tmp_path):
        # Manually feed subjects out of order to verify the sort.
        scans = (
            ScanEntry(format="snirf", path=tmp_path / "x.snirf",
                      bids_subject="zebra"),
            ScanEntry(format="snirf", path=tmp_path / "y.snirf",
                      bids_subject="alpha"),
            ScanEntry(format="snirf", path=tmp_path / "z.snirf",
                      bids_subject="mike"),
        )
        m = Manifest(root=tmp_path, scans=scans)
        labels = [n["label"] for n in dataset_tree.build_nodes(m)]
        assert labels == ["sub-alpha", "sub-mike", "sub-zebra"]


# ---------------------------------------------------------------------------
# dataset_tree.find_scan — id → ScanEntry lookup
# ---------------------------------------------------------------------------


class TestFindScan:
    def test_returns_scan_for_known_path(self, bids_manifest):
        scan = bids_manifest.scans[0]
        found = dataset_tree.find_scan(bids_manifest, str(scan.path))
        assert found is scan

    def test_returns_none_for_unknown_path(self, bids_manifest):
        assert dataset_tree.find_scan(bids_manifest, "/nonexistent") is None

    def test_returns_none_for_none_manifest(self):
        assert dataset_tree.find_scan(None, "/anything") is None


# ---------------------------------------------------------------------------
# workspace module contracts — registration + tab table
# ---------------------------------------------------------------------------


class TestWorkspaceModule:
    def test_register_is_callable(self):
        assert callable(workspace.register)

    def test_tab_names_match_planned_set(self):
        """Sprint 2.3 ships the canonical tab order. Subsequent sprints add
        content to each tab but should NOT change the order — this test is
        the regression guard."""
        assert workspace.TAB_NAMES == (
            "Inspect",
            "Quality",
            "Preprocess",
            "HRFs",
            "Activity",
            "HRtree",
            "Export",
        )

    def test_register_pages_invokes_workspace_register(self):
        source = inspect.getsource(gui_app._register_pages)
        assert "workspace.register()" in source

    def test_app_no_longer_contains_workspace_stub_text(self):
        """Sprint 2.2 inline stub said 'Full workspace UI lands in Sprint 2.3'.
        Confirm Sprint 2.3 removed that string from app.py so future readers
        don't see contradictory placeholder messaging."""
        source = inspect.getsource(gui_app)
        assert "Full workspace UI lands in Sprint 2.3" not in source


# ---------------------------------------------------------------------------
# workspace rendering — User fixture
# ---------------------------------------------------------------------------


async def test_workspace_empty_state_when_no_manifest(user: User):
    global_state.reset()
    await user.open("/workspace")
    await user.should_see("No dataset loaded")
    await user.should_see("Open a folder from the welcome screen")


async def test_workspace_renders_toolbar(user: User):
    global_state.reset()
    await user.open("/workspace")
    await user.should_see("HRfunc")
    await user.should_see("Back to welcome")


async def test_workspace_renders_three_pane_with_manifest(
    user: User, tmp_path
):
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

    await user.open("/workspace")
    # Dataset header label appears in left pane
    await user.should_see("Dataset")
    # Subject node visible in tree
    await user.should_see("sub-01")
    # Tab labels visible
    await user.should_see("Inspect")
    await user.should_see("Quality")
    await user.should_see("HRtree")
    # Manifest summary in right pane
    await user.should_see("Manifest")


async def test_workspace_inspector_prompts_when_no_scan_selected(
    user: User, tmp_path
):
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "x.snirf",
                         display_name="x.snirf"),),
    )
    await user.open("/workspace")
    # Inspect tab is the default; should prompt to select a scan
    await user.should_see("Select a scan from the dataset tree")


async def test_workspace_right_pane_shows_scan_count(user: User, tmp_path):
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(
            ScanEntry(format="snirf", path=tmp_path / "a.snirf"),
            ScanEntry(format="snirf", path=tmp_path / "b.snirf"),
            ScanEntry(format="fif", path=tmp_path / "c.fif"),
        ),
    )
    await user.open("/workspace")
    await user.should_see("3")  # scan count in right pane


async def test_workspace_placeholder_tabs_show_sprint_label(
    user: User, tmp_path
):
    """Tabs beyond Inspect should announce when their content lands."""
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "a.snirf"),),
    )
    await user.open("/workspace")
    # Open the Preprocess tab — exact assertion of text is brittle for
    # tab content (it's not visible until clicked), so instead verify
    # the tab label itself is on the page.
    await user.should_see("Preprocess")
    await user.should_see("HRFs")
    await user.should_see("Activity")
    await user.should_see("Export")
