"""Dataset tree — left-pane navigation of a Manifest.

Renders a Manifest's scans as a hierarchical tree grouped by BIDS subject /
session when the metadata is available, or by parent directory for non-BIDS
folders. Clicking a leaf node sets ``state.selected_scan``, which the
workspace's center and right panes react to.

Grouping rules:
- ``bids_subject`` present  → group under ``sub-<id>``
- ``bids_subject`` absent   → group under ``📁 <parent-dir-name>``
- ``bids_session`` present  → sub-group ``ses-<id>``
- ``bids_session`` absent   → sub-group ``(no session)``

The leaf node label uses the ScanEntry's ``display_name`` (which itself
combines BIDS components when available — see ``scan._make_display_name``).
Node IDs are the absolute scan path stringified — this makes the click
lookup O(1) given a dict from path → ScanEntry.

Search filter (Sprint 3.1):
``build_nodes`` accepts an optional ``filter_text``. When non-empty, scans
are kept only if the filter (case-insensitive) appears in ``display_name``
or the stringified path. Subjects and sessions with zero surviving scans
are pruned, so the tree only shows the relevant subtree. ``render`` wires
a ``ui.input`` above the tree that refreshes the body on change.

Public API:
    build_nodes(manifest, filter_text="") -> list[dict]   - tree node structure
    render(state, on_select_scan=None)                    - render the tree + wire selection
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

from nicegui import ui

from ..state import AppState
from ...io.manifest import Manifest, ScanEntry

logger = logging.getLogger(__name__)

_NO_SESSION_LABEL = "(no session)"

OnScanSelect = Callable[[Optional[ScanEntry]], None]


def _scan_matches_filter(scan: ScanEntry, filter_text: str) -> bool:
    """Case-insensitive substring match against display_name + path."""
    if not filter_text:
        return True
    needle = filter_text.lower()
    if scan.display_name and needle in scan.display_name.lower():
        return True
    return needle in str(scan.path).lower()


def build_nodes(manifest: Manifest, filter_text: str = "") -> List[dict]:
    """Convert a Manifest into the dict structure ``ui.tree`` expects.

    Returns a list of subject-level nodes. Each subject node has children
    (sessions); each session has children (scans). The scan-level nodes use
    the stringified absolute path as the ``id`` so click handlers can do a
    direct dict lookup back to the ScanEntry.

    When ``filter_text`` is non-empty, scans are filtered by case-insensitive
    substring match against their display_name and path. Subjects/sessions
    with zero surviving scans are pruned.

    Stable ordering: subjects sorted alphabetically, sessions within a
    subject sorted alphabetically, scans within a session sorted by
    display_name. Without sorting, the tree would jitter between scans
    depending on filesystem walk order.
    """
    # subject_key -> session_key -> list[ScanEntry]
    subjects: Dict[str, Dict[str, List[ScanEntry]]] = {}

    for scan in manifest.scans:
        if not _scan_matches_filter(scan, filter_text):
            continue
        sub_key = (
            f"sub-{scan.bids_subject}"
            if scan.bids_subject
            else f"📁 {scan.path.parent.name}"
        )
        ses_key = (
            f"ses-{scan.bids_session}"
            if scan.bids_session
            else _NO_SESSION_LABEL
        )
        subjects.setdefault(sub_key, {}).setdefault(ses_key, []).append(scan)

    nodes: List[dict] = []
    for sub_key in sorted(subjects):
        sub_node = {
            "id": f"subject::{sub_key}",
            "label": sub_key,
            "children": [],
        }
        for ses_key in sorted(subjects[sub_key]):
            ses_node = {
                "id": f"session::{sub_key}::{ses_key}",
                "label": ses_key,
                "children": [
                    {
                        "id": str(scan.path),
                        "label": scan.display_name or scan.path.name,
                    }
                    for scan in sorted(
                        subjects[sub_key][ses_key],
                        key=lambda s: s.display_name or str(s.path),
                    )
                ],
            }
            sub_node["children"].append(ses_node)
        nodes.append(sub_node)
    return nodes


def render(
    state: AppState,
    on_select_scan: Optional[OnScanSelect] = None,
) -> None:
    """Render the dataset tree inside the current NiceGUI context.

    Reads ``state.manifest`` to build nodes and wires the click handler to
    update ``state.selected_scan``. If ``on_select_scan`` is provided, it
    is called after the state update with the resolved ScanEntry (or
    ``None`` if the user clicked a group node) — typically used by the
    workspace to refresh the inspector panel.

    A search input above the tree filters scans by case-insensitive
    substring match against display_name or path. Filter state lives in a
    closure dict so the refreshable body always reads the latest value.

    Caller is responsible for placing this inside the desired layout
    (typically the left pane of a splitter).
    """
    if state.manifest is None or not state.manifest.scans:
        ui.label("No dataset loaded.").classes("opacity-60 text-sm p-4")
        return

    # Closure-held filter state. A mutable dict (rather than a nonlocal
    # string) lets the on_change handler write without scope gymnastics.
    filter_state: Dict[str, str] = {"text": ""}

    # Path string -> ScanEntry, for O(1) lookup in the click handler.
    # Built once over the full manifest — filtering only affects which
    # nodes are *rendered*, not which paths can be resolved on click.
    path_to_scan: Dict[str, ScanEntry] = {
        str(s.path): s for s in state.manifest.scans
    }

    def _on_select(event) -> None:
        node_id = event.value
        scan = path_to_scan.get(node_id)
        # Group nodes (subject/session) have no entry in path_to_scan, so
        # scan is None — clear selection to revert the inspector to empty.
        state.selected_scan = scan
        if scan is not None:
            logger.debug("Selected scan: %s", scan.path)
        if on_select_scan is not None:
            on_select_scan(scan)

    @ui.refreshable
    def _tree_body() -> None:
        nodes = build_nodes(state.manifest, filter_state["text"])
        if not nodes:
            ui.label("No scans match filter.").classes(
                "opacity-60 text-xs p-2"
            )
            return
        ui.tree(nodes, on_select=_on_select).classes("w-full")

    def _on_filter_change(event) -> None:
        filter_state["text"] = event.value or ""
        _tree_body.refresh()

    ui.input(
        placeholder="Filter scans…",
        on_change=_on_filter_change,
    ).props("dense clearable").classes("w-full")
    _tree_body()


def find_scan(manifest: Optional[Manifest], path_id: str) -> Optional[ScanEntry]:
    """Look up a ScanEntry by its stringified absolute path.

    Convenience for tests and components that receive a node id from a
    tree event and need the corresponding ScanEntry.
    """
    if manifest is None:
        return None
    for scan in manifest.scans:
        if str(scan.path) == path_id:
            return scan
    return None
