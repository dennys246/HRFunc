"""Workspace page — the main GUI surface once a dataset is loaded.

Three-pane layout, all wrapped in resizable splitters so users can rearrange
the available real estate:

  ┌─────────────┬──────────────────────────────┬──────────────┐
  │  Dataset    │  [Inspect][Quality][Preprocess]              │
  │  tree       │  [HRFs][Activity][HRtree][Export]            │
  │  (subjects, │  ───────────────────────────  │  Scan        │
  │   sessions, │   active tab content here     │  inspector   │
  │   scans)    │                                │              │
  └─────────────┴──────────────────────────────┴──────────────┘

Sprint 2.3 ships the shell: dataset tree (real), tab structure (real), and
inspector showing selected ScanEntry metadata (real). Tab content beyond
``Inspect`` is placeholder — Sprint 3+ replaces it with the estimation,
quality, and visualization panels.

Routing:
- Welcome → workspace via ``ui.navigate.to("/workspace")`` after a
  successful folder scan.
- Workspace → welcome via the "Back to welcome" button in the toolbar.

State:
- Reads ``state.manifest`` (set by the welcome page's open-folder flow).
- Reads/writes ``state.selected_scan`` (driven by dataset-tree clicks).
- Tab change does not modify state — the visible tab is local to the
  workspace render.
"""

from __future__ import annotations

import logging

from nicegui import ui

from ..components import dataset_tree
from ..state import AppState, state as global_state
from ..theme import apply_theme
from ...io.manifest import ScanEntry

logger = logging.getLogger(__name__)

# Tabs in the order researchers typically traverse them.
TAB_NAMES = (
    "Inspect",
    "Quality",
    "Preprocess",
    "HRFs",
    "Activity",
    "HRtree",
    "Export",
)


def register() -> None:
    """Register the workspace page handler at ``/workspace``.

    Called by ``app._register_pages()``. Replaces the Sprint 2.2 inline
    stub that just showed a scan count.
    """

    @ui.page("/workspace")
    def workspace_page() -> None:
        _render(global_state)


def _render(state: AppState) -> None:
    """Render the workspace against the given AppState.

    Split from the page handler so tests can call it with a synthetic
    state without going through NiceGUI's routing layer.
    """
    apply_theme()
    _render_toolbar()

    if state.manifest is None or not state.manifest.scans:
        _render_empty_state()
        return

    _render_three_pane(state)


# ---------------------------------------------------------------------------
# Toolbar — top of every workspace render
# ---------------------------------------------------------------------------


def _render_toolbar() -> None:
    with ui.row().classes(
        "w-full items-center justify-between px-6 py-3 border-b border-slate-800"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("psychology", size="2rem").classes("text-primary")
            ui.label("HRFunc").classes("text-2xl font-semibold")
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Back to welcome",
                on_click=lambda: ui.navigate.to("/"),
            ).props("flat color=primary")


# ---------------------------------------------------------------------------
# Empty state — manifest is None or has zero scans
# ---------------------------------------------------------------------------


def _render_empty_state() -> None:
    with ui.column().classes(
        "w-full items-center justify-center mt-24 gap-3"
    ):
        ui.icon("folder_off", size="4rem").classes("opacity-40")
        ui.label("No dataset loaded.").classes("text-xl opacity-80")
        ui.label(
            "Open a folder from the welcome screen to start exploring scans."
        ).classes("text-sm opacity-60")
        ui.button(
            "Back to welcome",
            on_click=lambda: ui.navigate.to("/"),
        ).props("color=primary")


# ---------------------------------------------------------------------------
# Three-pane layout — splitters for left | (center | right)
# ---------------------------------------------------------------------------


def _render_three_pane(state: AppState) -> None:
    # Outer splitter: dataset tree on the left, everything else on the right.
    with ui.splitter(value=20, limits=(10, 40)).classes(
        "w-full h-screen"
    ) as outer:
        with outer.before:
            _render_left_pane(state)
        with outer.after:
            # Inner splitter: center tabs (taking most space), right inspector.
            with ui.splitter(value=72, limits=(50, 90)).classes(
                "w-full h-full"
            ) as inner:
                with inner.before:
                    _render_center_pane(state)
                with inner.after:
                    _render_right_pane(state)


# ---------------------------------------------------------------------------
# Left pane — dataset tree
# ---------------------------------------------------------------------------


def _render_left_pane(state: AppState) -> None:
    with ui.column().classes("w-full h-full p-3 gap-2 overflow-auto"):
        ui.label("Dataset").classes("text-xs uppercase opacity-60 tracking-wide")
        if state.manifest is not None:
            ui.label(str(state.manifest.root)).classes(
                "text-xs font-mono opacity-70 break-all"
            )
        dataset_tree.render(state, on_select_scan=lambda _scan: _refresh_inspector(state))


def _refresh_inspector(state: AppState) -> None:
    """Refresh the Inspect tab body after a dataset-tree selection change.

    The Inspect tab's body is wrapped in ``@ui.refreshable`` and stashes its
    refresh callable on ``state._inspect_refresh`` at render time. Calling
    that re-runs the body against the latest ``state.selected_scan`` without
    rebuilding the whole workspace.
    """
    refresh = getattr(state, "_inspect_refresh", None)
    if refresh is not None:
        refresh()


# ---------------------------------------------------------------------------
# Center pane — tabs
# ---------------------------------------------------------------------------


def _render_center_pane(state: AppState) -> None:
    with ui.column().classes("w-full h-full"):
        with ui.tabs().classes("w-full") as tabs:
            for name in TAB_NAMES:
                ui.tab(name)
        with ui.tab_panels(tabs, value=TAB_NAMES[0]).classes(
            "w-full flex-1 overflow-auto"
        ):
            for name in TAB_NAMES:
                with ui.tab_panel(name):
                    _render_tab_panel(name, state)


def _render_tab_panel(name: str, state: AppState) -> None:
    """Render one tab's body.

    ``Inspect`` is real and shows the selected scan's metadata. Every other
    tab is a "Coming in Sprint N" placeholder for now — Sprint 3+ replaces
    them.
    """
    if name == "Inspect":
        _render_inspect_tab(state)
        return

    # Map each placeholder tab to the sprint that fills it in.
    next_sprint = {
        "Quality": "Sprint 4 (lens-module wrapper)",
        "Preprocess": "Sprint 3 (preprocess panel)",
        "HRFs": "Sprint 3 (estimate panel + HRF gallery)",
        "Activity": "Sprint 3 (estimate-activity panel)",
        "HRtree": "Sprint 4 (plotly 3D explorer)",
        "Export": "Sprint 5 (export panel)",
    }.get(name, "a future sprint")

    with ui.column().classes("p-6 gap-2"):
        ui.label(name).classes("text-2xl font-semibold")
        ui.label(f"Coming in {next_sprint}.").classes(
            "text-sm opacity-60"
        )


def _render_inspect_tab(state: AppState) -> None:
    """Inspect tab content — selected scan's basic metadata.

    Renders against ``state.selected_scan``. When no scan is selected, shows
    a prompt to pick one. The render is wrapped in ``ui.refreshable`` so
    dataset-tree clicks update this panel without a full page rebuild.
    """

    @ui.refreshable
    def _body() -> None:
        scan = state.selected_scan
        if scan is None:
            with ui.column().classes("p-6 gap-2"):
                ui.label("Inspect").classes("text-2xl font-semibold")
                ui.label("Select a scan from the dataset tree.").classes(
                    "text-sm opacity-60"
                )
            return
        with ui.column().classes("p-6 gap-3"):
            ui.label(scan.display_name or scan.path.name).classes(
                "text-2xl font-semibold"
            )
            _kv_row("Format", scan.format)
            _kv_row("Path", str(scan.path))
            if scan.n_channels is not None:
                _kv_row("Channels", str(scan.n_channels))
            if scan.sfreq is not None:
                _kv_row("Sampling rate", f"{scan.sfreq:.4g} Hz")
            if scan.bids_subject:
                _kv_row("BIDS subject", scan.bids_subject)
            if scan.bids_session:
                _kv_row("BIDS session", scan.bids_session)
            if scan.bids_task:
                _kv_row("BIDS task", scan.bids_task)
            if scan.bids_run:
                _kv_row("BIDS run", scan.bids_run)

    _body()
    # Stash the refresher on the state so dataset_tree.render's on_select
    # can call it after updating state.selected_scan. Sprint 3+ panels will
    # subscribe to the same channel via a more structured event bus, but
    # the direct ref keeps Sprint 2.3 scope tight.
    state._inspect_refresh = _body.refresh  # type: ignore[attr-defined]


def _kv_row(key: str, value: str) -> None:
    with ui.row().classes("w-full gap-4"):
        ui.label(key).classes("text-xs uppercase opacity-60 w-32")
        ui.label(value).classes("text-sm break-all")


# ---------------------------------------------------------------------------
# Right pane — minimal scan inspector for Sprint 2.3
# ---------------------------------------------------------------------------


def _render_right_pane(state: AppState) -> None:
    """Right-pane inspector.

    Sprint 2.3 shows manifest-level summary (root, scan count, scan-by-
    format histogram). Sprint 3 replaces this with parameter controls
    (preprocess toggles, estimate sliders) when an estimation tab is
    active, and falls back to this summary otherwise.
    """
    with ui.column().classes("w-full h-full p-4 gap-3 overflow-auto"):
        ui.label("Manifest").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        if state.manifest is None:
            ui.label("No manifest").classes("text-sm opacity-60")
            return

        manifest = state.manifest
        format_counts = _count_by_format(manifest.scans)

        _kv_row("Scans", str(len(manifest.scans)))
        _kv_row("Scanned", manifest.scanned_at.strftime("%Y-%m-%d %H:%M UTC"))
        if manifest.errors:
            _kv_row("Errors", str(len(manifest.errors)))

        if format_counts:
            ui.separator()
            ui.label("By format").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            for fmt, count in sorted(format_counts.items()):
                _kv_row(fmt, str(count))


def _count_by_format(scans) -> dict:
    counts: dict = {}
    for scan in scans:
        counts[scan.format] = counts.get(scan.format, 0) + 1
    return counts
