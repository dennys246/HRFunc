"""HRF Library page — browse the bundled literature-derived HRF databases.

The /library route is the entry path for the **Browser** persona — a
researcher who has no scan of their own but wants to explore the
literature HRFs HRFunc bundles (``hbo_hrfs.json`` / ``hbr_hrfs.json``)
and the community-contributed entries that get merged in over time.

Sprint 4.2 + 4.3 + 4.4 ship together as one combined branch because the
three pieces are tightly coupled on shared state:

- **Library Browser (4.4)** — the three-pane ``/library`` page scaffold:
  Context Filter on the left, plotly 3D HRtree explorer in the center,
  HRF detail card on the right.
- **HRtree explorer (4.2)** — the plotly 3D scatter showing HRF nodes
  positioned by their (x, y, z) location, colored by oxygenation,
  hoverable for context preview, clickable to select an HRF.
- **Context Filter (4.3)** — the left-sidebar form for filtering the
  visible HRFs by context fields (task, doi, demographics, etc.).

Loading: both trees are read from disk once on the first /library
visit and stashed on ``state.library_hbo`` / ``state.library_hbr``.
Subsequent visits reuse the cached trees. The trees themselves never
change (the bundled files are read-only data), so we don't bother
with cache invalidation.

Filtering: applied non-destructively in ``_apply_filter`` rather than
via ``tree.branch()`` — the GUI's filter is a view, not a permanent
sub-tree, and we want users to be able to toggle filters freely.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from nicegui import ui

from ..state import AppState, state as global_state
from ..theme import apply_theme, page_container

if TYPE_CHECKING:
    from ...hrtree import tree as TreeType

logger = logging.getLogger(__name__)


# Subset of context fields exposed as filter controls. The library tree
# context has ~10 fields; researchers most commonly filter on the first
# few (task, doi, study, demographics). Less-used fields (intensity,
# protocol) stay accessible via the data but don't get a control.
FILTER_FIELDS = (
    "task",
    "doi",
    "study",
    "demographics",
    "stimulus",
    "conditions",
)


def register() -> None:
    """Register the /library page handler.

    Called by ``app._register_pages()``. Replaces the Sprint 2.2 inline
    stub.
    """

    @ui.page("/library")
    def library_page() -> None:
        _render(global_state)


def _render(state: AppState) -> None:
    """Render the Library page against the given AppState.

    Split from the page handler so tests can call it with a synthetic
    state without going through NiceGUI's routing layer.

    Event-bus subscriptions are page-scoped (see ``workspace._render`` for
    the same pattern): clear the subscriber list at the top so repeat
    visits don't accumulate dead refreshable handles from prior page
    DOMs. Each pane's render function re-subscribes during this render.
    """
    apply_theme()
    state.subscribers.clear()
    _render_toolbar()

    if state.library_hbo is None or state.library_hbr is None:
        _load_trees(state)

    _render_three_pane(state)


def _load_trees(state: AppState) -> None:
    """Read the bundled HRF databases into memory once.

    The trees stay on state for the lifetime of the process. Failures are
    surfaced to ``state.last_error`` but the page still renders so users
    can see what went wrong rather than getting a blank screen.
    """
    try:
        # Lazy-import to keep the GUI import graph minimal at module load.
        from ...hrtree import tree as Tree
        from ... import __file__ as hrfunc_file

        import os
        lib_dir = os.path.join(os.path.dirname(hrfunc_file), "hrfs")
        hbo_path = os.path.join(lib_dir, "hbo_hrfs.json")
        hbr_path = os.path.join(lib_dir, "hbr_hrfs.json")
        state.library_hbo = Tree(hbo_path)
        state.library_hbr = Tree(hbr_path)
        logger.info(
            "Loaded library trees: HbO=%d nodes, HbR=%d nodes",
            len(state.library_hbo.gather(state.library_hbo.root)),
            len(state.library_hbr.gather(state.library_hbr.root)),
        )
    except Exception as exc:  # noqa: BLE001
        state.last_error = (
            f"Failed to load bundled HRF library: {type(exc).__name__}: {exc}"
        )
        logger.exception("library load failed: %s", exc)


# ---------------------------------------------------------------------------
# Toolbar + empty state
# ---------------------------------------------------------------------------


def _render_toolbar() -> None:
    with ui.row().classes(
        "w-full items-center justify-between px-6 py-3 border-b border-slate-800"
    ):
        with ui.row().classes("items-center gap-3"):
            ui.icon("library_books", size="2rem").classes("text-primary")
            ui.label("HRF Library").classes("text-2xl font-semibold")
        with ui.row().classes("items-center gap-2"):
            ui.button(
                "Back to welcome",
                on_click=lambda: ui.navigate.to("/"),
            ).props("flat color=primary")


# ---------------------------------------------------------------------------
# Three-pane layout — splitters for left | (center | right)
# ---------------------------------------------------------------------------


def _render_three_pane(state: AppState) -> None:
    """Render filter | viz | detail with two splitters, matching workspace."""
    with ui.splitter(value=20, limits=(10, 35)).classes(
        "w-full h-screen"
    ) as outer:
        with outer.before:
            _render_filter_pane(state)
        with outer.after:
            with ui.splitter(value=70, limits=(40, 90)).classes(
                "w-full h-full"
            ) as inner:
                with inner.before:
                    _render_viz_pane(state)
                with inner.after:
                    _render_detail_pane(state)


# ---------------------------------------------------------------------------
# Left pane: Context Filter
# ---------------------------------------------------------------------------


def _render_filter_pane(state: AppState) -> None:
    """The Context Filter sidebar.

    Each FILTER_FIELDS entry becomes a text input bound to
    ``state.library_filter[field]``. Empty string = field not filtered
    (the entry is removed from the dict). An "Apply" button refreshes
    the dependent viz + detail panes.
    """
    with ui.column().classes("w-full h-full p-3 gap-3 overflow-auto"):
        ui.label("Filter").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        ui.label(
            "Narrow the visible HRFs by context. Case-insensitive substring "
            "match against each field; leave blank to ignore."
        ).classes("text-xs opacity-60")

        # One ui.input per filter field.
        inputs: Dict[str, Any] = {}
        for field in FILTER_FIELDS:
            initial = str(state.library_filter.get(field, ""))
            inputs[field] = ui.input(
                label=field,
                value=initial,
            ).props("dense clearable").classes("w-full")

        def _apply() -> None:
            new_filter: Dict[str, Any] = {}
            for field, widget in inputs.items():
                value = (widget.value or "").strip()
                if value:
                    new_filter[field] = value
            state.library_filter = new_filter
            # Trigger a re-render of dependent panes via the event bus.
            state.publish("library_filter_changed", new_filter)

        def _reset() -> None:
            for widget in inputs.values():
                widget.value = ""
            state.library_filter = {}
            state.publish("library_filter_changed", {})

        with ui.row().classes("w-full gap-2"):
            ui.button("Apply", on_click=_apply).props("color=primary")
            ui.button("Reset", on_click=_reset).props("flat")

        # Live count of matching HRFs, refreshable so Apply / Reset
        # updates it. The count also annotates how many filtered HRFs
        # are NOT visualizable in the 3D viz because they lack a
        # location — without this, a user can see "5 / 22 match" while
        # the viz shows only 3 points and be confused.
        @ui.refreshable
        def _count_label() -> None:
            all_hrfs = gather_library_hrfs(state)
            matched = apply_filter(all_hrfs, state.library_filter)
            visualizable = sum(
                1
                for hrf in matched.values()
                if hrf.get("location") is not None
                and len(hrf.get("location") or []) >= 3
            )
            text = f"{len(matched)} / {len(all_hrfs)} HRFs match"
            if visualizable < len(matched):
                missing = len(matched) - visualizable
                text += f" ({missing} not visualizable: missing location)"
            ui.label(text).classes("text-xs opacity-70")

        _count_label()
        state.subscribe(
            "library_filter_changed", lambda _p=None: _count_label.refresh()
        )


# ---------------------------------------------------------------------------
# Center pane: HRtree 3D viz
# ---------------------------------------------------------------------------


def _render_viz_pane(state: AppState) -> None:
    """The plotly 3D scatter of HRF locations.

    Refreshable so the Apply button can re-render against the filter.
    """

    @ui.refreshable
    def _viz_body() -> None:
        all_hrfs = gather_library_hrfs(state)
        matched = apply_filter(all_hrfs, state.library_filter)

        if not all_hrfs:
            with ui.column().classes("p-6 gap-2"):
                ui.label("HRtree").classes("text-2xl font-semibold")
                if state.last_error:
                    ui.label(state.last_error).classes(
                        "text-sm text-red-400"
                    )
                else:
                    ui.label(
                        "Library trees not loaded. Returning to the welcome "
                        "screen and re-opening /library may help."
                    ).classes("text-sm opacity-60")
            return

        with ui.column().classes("w-full h-full p-3 gap-2"):
            ui.label(
                f"HRtree — {len(matched)} HRFs shown"
            ).classes("text-sm opacity-70")
            fig = build_plotly_figure(matched)
            plot = ui.plotly(fig).classes("w-full h-full")

            def _on_click(event) -> None:
                # NiceGUI's plotly click event delivers an args dict with a
                # 'points' list. Each point has a 'customdata' field if we
                # set it on the trace, which we use to store the HRF key.
                hrf_key = _extract_clicked_hrf_key(event)
                if hrf_key is None:
                    return
                hrf = matched.get(hrf_key) or all_hrfs.get(hrf_key)
                if hrf is None:
                    return
                # Stash the key on the dict so the detail pane can show it.
                state.library_selected_hrf = {**hrf, "_key": hrf_key}
                state.publish("library_selection_changed", hrf_key)

            plot.on("plotly_click", _on_click)

    _viz_body()

    # Re-render the viz on filter change.
    def _refresh_viz(_payload=None) -> None:
        _viz_body.refresh()
    state.subscribe("library_filter_changed", _refresh_viz)


def _extract_clicked_hrf_key(event) -> Optional[str]:
    """Pull the clicked HRF's key from a plotly click event payload.

    NiceGUI delivers ``event.args`` as the raw plotly JSON; the clicked
    point has a ``customdata`` field which we populated with the HRF key
    when building the figure. Returns None on any malformed payload,
    logging at warning level so a future plotly-API change surfaces
    instead of leaving clicks mysteriously silent.
    """
    try:
        args = getattr(event, "args", None) or {}
        points = args.get("points") or []
        if not points:
            return None
        first = points[0]
        return first.get("customdata")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "library: failed to extract HRF key from click event: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Right pane: HRF detail
# ---------------------------------------------------------------------------


def _render_detail_pane(state: AppState) -> None:
    """The selected-HRF detail card.

    Shows context metadata + a matplotlib trace plot for the picked HRF.
    """

    @ui.refreshable
    def _detail_body() -> None:
        hrf = state.library_selected_hrf
        with ui.column().classes("w-full h-full p-4 gap-3 overflow-auto"):
            ui.label("Detail").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            if hrf is None:
                ui.label("Click an HRF in the viz to inspect.").classes(
                    "text-sm opacity-60"
                )
                return

            key = hrf.get("_key", "")
            ui.label(key or "(no key)").classes(
                "text-lg font-mono break-all"
            )

            _kv("oxygenation", "HbO" if hrf.get("oxygenation") else "HbR")
            _kv("sfreq", f"{float(hrf.get('sfreq', 0)):.4g} Hz")
            loc = hrf.get("location") or [0, 0, 0]
            _kv(
                "location",
                f"x={loc[0]:.3f}  y={loc[1]:.3f}  z={loc[2]:.3f}",
            )
            trace = hrf.get("hrf_mean") or []
            _kv("trace length", str(len(trace)))

            context = hrf.get("context") or {}
            if context:
                ui.separator()
                ui.label("Context").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )
                for ctx_key, value in context.items():
                    if value is None:
                        continue
                    _kv(ctx_key, str(value))

            if trace:
                ui.separator()
                ui.label("Trace").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )
                png = _render_trace_png(hrf)
                if png is not None:
                    ui.image(png).classes("max-w-md")

    _detail_body()

    def _refresh_detail(_payload=None) -> None:
        _detail_body.refresh()
    state.subscribe("library_selection_changed", _refresh_detail)


def _kv(key: str, value: str) -> None:
    with ui.row().classes("w-full gap-4"):
        ui.label(key).classes("text-xs uppercase opacity-60 w-32")
        ui.label(value).classes("text-sm break-all")


# ---------------------------------------------------------------------------
# Data helpers (module-level so tests can call them)
# ---------------------------------------------------------------------------


def gather_library_hrfs(state: AppState) -> Dict[str, Dict[str, Any]]:
    """Combine the HbO + HbR trees into a single name → HRF-dict map.

    Returns empty dict if the trees aren't loaded. Each value is the
    gathered-form dict produced by ``tree.gather``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tree_obj in (state.library_hbo, state.library_hbr):
        if tree_obj is None:
            continue
        hrfs = tree_obj.gather(tree_obj.root)
        if hrfs:
            out.update(hrfs)
    return out


def apply_filter(
    hrfs: Dict[str, Dict[str, Any]],
    filter_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Filter an HRF dict by case-insensitive substring match on context.

    Each key in ``filter_kwargs`` must appear in the HRF's context with
    a value whose string representation contains the filter value (case-
    insensitive). HRFs missing a filtered key are excluded. Empty filter
    returns the input unchanged.
    """
    if not filter_kwargs:
        return dict(hrfs)

    out: Dict[str, Dict[str, Any]] = {}
    for key, hrf in hrfs.items():
        context = hrf.get("context") or {}
        if _hrf_matches_filter(context, filter_kwargs):
            out[key] = hrf
    return out


def _hrf_matches_filter(
    context: Dict[str, Any],
    filter_kwargs: Dict[str, Any],
) -> bool:
    """True if every (key, value) in the filter is reflected in context."""
    for field, needle in filter_kwargs.items():
        if needle is None or needle == "":
            continue
        haystack = context.get(field)
        if haystack is None:
            return False
        # Stringify and substring-match case-insensitively for primitive
        # context values. For list/tuple values, match if any entry matches.
        if isinstance(haystack, (list, tuple)):
            if not any(_str_match(item, needle) for item in haystack):
                return False
        else:
            if not _str_match(haystack, needle):
                return False
    return True


def _str_match(value: Any, needle: str) -> bool:
    if value is None:
        return False
    return str(needle).lower() in str(value).lower()


# ---------------------------------------------------------------------------
# Plotly figure builder
# ---------------------------------------------------------------------------


def build_plotly_figure(hrfs: Dict[str, Dict[str, Any]]):
    """Build the 3D scatter figure for the given HRF dict.

    Two traces: HbO (red) and HbR (blue). Each point's ``customdata`` is
    the HRF key so click handlers can look it up. ``hovertext`` carries a
    short context summary.
    """
    import plotly.graph_objects as go

    hbo_x, hbo_y, hbo_z, hbo_keys, hbo_hover = [], [], [], [], []
    hbr_x, hbr_y, hbr_z, hbr_keys, hbr_hover = [], [], [], [], []

    for key, hrf in hrfs.items():
        loc = hrf.get("location")
        # Skip HRFs without a real 3D location rather than fabricating
        # (0,0,0) — clustering location-less nodes at the origin would be
        # visually misleading and the GUI's spatial story (kd-tree) only
        # makes sense for HRFs with measured coordinates.
        if loc is None or len(loc) < 3:
            continue
        is_hbo = bool(hrf.get("oxygenation"))
        hover = _hover_text_for(key, hrf)
        if is_hbo:
            hbo_x.append(loc[0])
            hbo_y.append(loc[1])
            hbo_z.append(loc[2])
            hbo_keys.append(key)
            hbo_hover.append(hover)
        else:
            hbr_x.append(loc[0])
            hbr_y.append(loc[1])
            hbr_z.append(loc[2])
            hbr_keys.append(key)
            hbr_hover.append(hover)

    traces = []
    if hbo_x:
        traces.append(
            go.Scatter3d(
                x=hbo_x, y=hbo_y, z=hbo_z,
                mode="markers",
                marker=dict(size=4, color="#fb7185", opacity=0.85),
                name="HbO",
                customdata=hbo_keys,
                hovertext=hbo_hover,
                hoverinfo="text",
            )
        )
    if hbr_x:
        traces.append(
            go.Scatter3d(
                x=hbr_x, y=hbr_y, z=hbr_z,
                mode="markers",
                marker=dict(size=4, color="#38bdf8", opacity=0.85),
                name="HbR",
                customdata=hbr_keys,
                hovertext=hbr_hover,
                hoverinfo="text",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            aspectmode="data",
        ),
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _hover_text_for(key: str, hrf: Dict[str, Any]) -> str:
    """Short multi-line hover-text summary for one HRF."""
    context = hrf.get("context") or {}
    bits = [key]
    for field in ("task", "doi", "study", "demographics"):
        value = context.get(field)
        if value:
            bits.append(f"{field}: {value}")
    return "<br>".join(bits)


# ---------------------------------------------------------------------------
# Trace plot PNG
# ---------------------------------------------------------------------------


def _render_trace_png(hrf: Dict[str, Any]) -> Optional[str]:
    """Render the HRF trace as a base64 PNG line plot."""
    try:
        import base64
        import io as _io
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for library trace: %s", exc)
        return None

    trace = hrf.get("hrf_mean") or []
    if not trace:
        return None
    sfreq = float(hrf.get("sfreq") or 1.0)
    if sfreq <= 0:
        sfreq = 1.0

    fig = None
    try:
        t = np.arange(len(trace)) / sfreq
        std = hrf.get("hrf_std") or []
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
        ax.plot(t, trace, lw=1.2, color="#6366f1")
        if std and len(std) == len(trace):
            lower = np.asarray(trace) - np.asarray(std)
            upper = np.asarray(trace) + np.asarray(std)
            ax.fill_between(t, lower, upper, alpha=0.15, color="#6366f1")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (a.u.)")
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("library trace render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
