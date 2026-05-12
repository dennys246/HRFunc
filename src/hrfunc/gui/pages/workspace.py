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

Sprint 2.3 shipped the shell: dataset tree (real), tab structure (real), and
inspector showing selected ScanEntry metadata (real). Sprint 3.1 enriches
the Inspect tab to be Raw-aware: when a scan is selected, the underlying
MNE Raw is loaded in the background and the channel list, 2D probe layout,
and event annotations appear in the Inspect tab.

Routing:
- Welcome → workspace via ``ui.navigate.to("/workspace")`` after a
  successful folder scan.
- Workspace → welcome via the "Back to welcome" button in the toolbar.

State:
- Reads ``state.manifest`` (set by the welcome page's open-folder flow).
- Reads/writes ``state.selected_scan`` (driven by dataset-tree clicks).
- Reads/writes ``state.raw_cache`` (Sprint 3.1 — populated by background
  Raw loads kicked off when a scan is selected).
- Tab change does not modify state — the visible tab is local to the
  workspace render.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from typing import TYPE_CHECKING, Optional

from nicegui import background_tasks, ui

from ..components import (
    activity_panel,
    dataset_tree,
    export_panel,
    hrf_panel,
    preprocess_panel,
    quality_panel,
)
from ..state import AppState, state as global_state
from ..theme import apply_theme
from ...io.manifest import ScanEntry

if TYPE_CHECKING:
    import mne

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

    Event-bus subscriptions are scoped to a single workspace render. We
    clear ``state.subscribers`` here so that navigating away and back
    doesn't accumulate dead refreshable handles pointing at the previous
    DOM. Each tab's render method re-subscribes during this render. If
    future sprints add subscribers from outside the workspace page, this
    cleanup needs a more selective approach.
    """
    apply_theme()
    state.subscribers.clear()
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
        dataset_tree.render(
            state, on_select_scan=lambda scan: _on_scan_selected(state, scan)
        )


def _on_scan_selected(state: AppState, scan: Optional[ScanEntry]) -> None:
    """Publish ``scan_selected`` and kick off a background Raw load if needed.

    Dataset-tree clicks reach the workspace via this single function, which
    has two responsibilities:

    1. Fan the selection out to subscribers (Inspect tab, Preprocess tab,
       and any future panel that reacts to selection) by publishing
       ``"scan_selected"``.
    2. If a scan is selected and its Raw is not yet cached, schedule a
       background load. When that load completes, ``_load_scan_raw``
       publishes ``"scan_loaded"`` so panels can react to the now-available
       Raw — typically by re-rendering the section that depends on it.

    The path-equality check in ``_load_scan_raw`` (not done here) ensures
    that if the user clicks A then B before A loads, A's late completion
    does NOT publish a stale ``scan_loaded`` for A.
    """
    state.publish("scan_selected", scan)

    if scan is None or scan in state.raw_cache:
        return
    background_tasks.create(_load_scan_raw(state, scan))


async def _load_scan_raw(state: AppState, scan: ScanEntry) -> None:
    """Load ``scan`` into ``state.raw_cache`` off the event loop.

    Bypasses ``workers.run_in_background`` deliberately: that helper gates on
    ``state.busy`` (reserved for the long estimation tasks Sprint 3.3/3.4
    wire up), and rapid scan navigation should not be blocked by that gate
    or affect the estimation progress indicator.

    After the load completes (or fails), publishes ``"scan_loaded"`` so
    subscribers can react — but only if the user is still inspecting the
    same scan. Equality is by path, not object identity, so a fresh
    ScanEntry from a re-scan still matches.
    """
    loop = asyncio.get_event_loop()
    try:
        await loop.run_in_executor(None, state.raw_cache.get, scan)
    except Exception as exc:  # noqa: BLE001 — surface to UI via last_error
        state.last_error = (
            f"Failed to load scan: {type(exc).__name__}: {exc}"
        )
        logger.exception("Failed to load scan %s", scan.path)
        return

    current = state.selected_scan
    if current is not None and current.path == scan.path:
        state.publish("scan_loaded", scan)


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

    ``Inspect`` is real and shows the selected scan's metadata plus
    Raw-derived recording info (Sprint 3.1). Every other tab is a "Coming
    in Sprint N" placeholder — Sprint 3.2+ replaces them.
    """
    if name == "Inspect":
        _render_inspect_tab(state)
        return
    if name == "Preprocess":
        preprocess_panel.render(state)
        return
    if name == "HRFs":
        hrf_panel.render(state)
        return
    if name == "Activity":
        activity_panel.render(state)
        return
    if name == "Quality":
        quality_panel.render(state)
        return
    if name == "Export":
        export_panel.render(state)
        return

    # Map each placeholder tab to the sprint that fills it in.
    next_sprint = {
        "HRtree": "Sprint 4 (plotly 3D explorer at /library)",
    }.get(name, "a future sprint")

    with ui.column().classes("p-6 gap-2"):
        ui.label(name).classes("text-2xl font-semibold")
        ui.label(f"Coming in {next_sprint}.").classes(
            "text-sm opacity-60"
        )


def _render_inspect_tab(state: AppState) -> None:
    """Inspect tab — Metadata + Recording sections, refreshable on scan change.

    Wraps ``_render_inspect_body`` in ``ui.refreshable`` and subscribes the
    resulting refresher to the event bus so dataset-tree clicks
    (``scan_selected``) and background-load completions (``scan_loaded``)
    drive a re-render without rebuilding the whole workspace.
    """

    @ui.refreshable
    def _body() -> None:
        _render_inspect_body(state)

    _body()

    # Subscribe the refreshable to scan-state events. ``_render`` clears the
    # subscriber list at the top of each workspace render, so this is safe to
    # call once per render without accumulating duplicates.
    def _refresh(_payload=None):
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)


def _render_inspect_body(state: AppState) -> None:
    """Render the Inspect tab body against the current ``state.selected_scan``.

    Two sections: Metadata (always available from ScanEntry) and Recording
    (renders once the MNE Raw is in ``state.raw_cache``). If no scan is
    selected, shows a prompt; if a scan is selected but not yet cached,
    shows a "Loading recording…" skeleton so users see the load is in
    flight.

    Extracted as a top-level function so tests can call it inside a
    synthetic NiceGUI context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    if scan is None:
        with ui.column().classes("p-6 gap-2"):
            ui.label("Inspect").classes("text-2xl font-semibold")
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
        return

    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label(scan.display_name or scan.path.name).classes(
            "text-2xl font-semibold"
        )

        # ── Metadata section (always renders from ScanEntry)
        ui.label("Metadata").classes(
            "text-xs uppercase opacity-60 tracking-wide"
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

        ui.separator()

        # ── Recording section (loaded MNE Raw)
        ui.label("Recording").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        if scan in state.raw_cache:
            try:
                raw = state.raw_cache.get(scan)
            except Exception as exc:  # noqa: BLE001
                # Cache hit but reload failed (defensive — get() shouldn't
                # error on hit, but the cache could be cleared between the
                # __contains__ check and the get call by another callback).
                logger.warning("Inspect: cache.get raised: %s", exc)
                ui.label(
                    "Recording unavailable — cache entry was cleared."
                ).classes("text-sm opacity-60")
                return
            _render_recording_sections(raw)
        else:
            with ui.row().classes("items-center gap-3"):
                ui.spinner(size="sm")
                ui.label("Loading recording…").classes(
                    "text-sm opacity-70"
                )


def _render_recording_sections(raw: "mne.io.BaseRaw") -> None:
    """Render channel list + probe layout + events for a loaded Raw."""
    ch_names = list(raw.ch_names)
    annotations = raw.annotations if raw.annotations is not None else []

    # ── Channel list (collapsed by default; channel counts dominate the
    # vertical real estate on dense montages otherwise).
    with ui.expansion(
        f"Channels ({len(ch_names)})",
        icon="sensors",
    ).classes("w-full"):
        with ui.column().classes(
            "max-h-64 overflow-auto gap-1 text-xs font-mono"
        ):
            for name in ch_names:
                ui.label(name).classes("opacity-80")

    # ── 2D probe layout — matplotlib via base64 PNG. ui.matplotlib would
    # also work, but PNG keeps the snapshot purely declarative and avoids
    # holding a Figure across the NiceGUI re-render cycle.
    with ui.expansion("Probe layout", icon="scatter_plot").classes(
        "w-full"
    ):
        probe_html = _render_probe_png(raw)
        if probe_html is None:
            ui.label("Probe layout unavailable for this scan.").classes(
                "text-sm opacity-60"
            )
        else:
            ui.image(probe_html).classes("max-w-md")

    # ── Events / annotations
    n_events = len(annotations)
    with ui.expansion(
        f"Events ({n_events})", icon="event"
    ).classes("w-full"):
        if n_events == 0:
            ui.label("No events recorded in this scan.").classes(
                "text-sm opacity-60"
            )
        else:
            rows = [
                {
                    "description": str(ann["description"]),
                    "onset": f"{float(ann['onset']):.3f}",
                    "duration": f"{float(ann['duration']):.3f}",
                }
                for ann in annotations
            ]
            ui.table(
                columns=[
                    {
                        "name": "description",
                        "label": "Description",
                        "field": "description",
                        "align": "left",
                    },
                    {
                        "name": "onset",
                        "label": "Onset (s)",
                        "field": "onset",
                        "align": "right",
                    },
                    {
                        "name": "duration",
                        "label": "Duration (s)",
                        "field": "duration",
                        "align": "right",
                    },
                ],
                rows=rows,
                row_key="onset",
            ).classes("w-full")


def _render_probe_png(raw: "mne.io.BaseRaw") -> Optional[str]:
    """Render the probe layout to a base64-encoded PNG data URL.

    Returns None if MNE refuses to plot (no montage, no sensor positions,
    or any matplotlib failure). The caller is expected to fall back to a
    placeholder label in that case.
    """
    try:
        # Lazy imports — matplotlib import time is noticeable at GUI startup,
        # and this function is only called on a successful Raw load.
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for probe layout: %s", exc)
        return None

    fig = None
    try:
        fig = raw.plot_sensors(show=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("raw.plot_sensors failed: %s", exc)
        if fig is not None:
            plt.close(fig)
        return None

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("probe layout encode failed: %s", exc)
        return None
    finally:
        plt.close(fig)


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
    format histogram). Sprint 3+ replaces this with parameter controls
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
