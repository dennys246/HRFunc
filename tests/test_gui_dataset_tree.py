"""Targeted unit tests for the dataset-tree component.

Covers ``hrfunc.gui.components.dataset_tree`` — the Manifest → tree-node
conversion that renders the left-pane scan picker, plus the
``find_scan`` lookup helper:

- BIDS manifests group by ``sub-XX`` → ``ses-YY`` → scan-leaf.
- Non-BIDS manifests group by parent directory with a single
  ``(no session)`` group.
- Subject ordering is alphabetic (regression guard for set-iteration
  non-determinism).
- ``find_scan`` resolves leaf-id strings (stringified paths) to
  ScanEntry instances; unknown ids and None manifests return None.

These tests were originally in ``test_gui_workspace.py`` (alongside the
workspace page render tests). Moved into a dedicated file when v1.4
Phase 5 deleted ``pages/workspace.py``; the dataset_tree component
itself survived the rework and is reused inside every data-tab in the
shell, so the coverage stays relevant.

All tests are sync — no NiceGUI rendering required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from hrfunc.gui.components import dataset_tree  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — synthetic manifests
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
# build_nodes
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
# find_scan
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
