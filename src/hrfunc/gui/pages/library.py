"""HRF Library page — browse the bundled literature-derived HRF databases.

The /library route is the entry path for the **Browser** persona — a
researcher who has no scan of their own but wants to explore the
literature HRFs HRfunc bundles (``hbo_hrfs.json`` / ``hbr_hrfs.json``)
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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from nicegui import ui

from ..state import AppState, state as global_state
from ..theme import apply_theme, page_container

if TYPE_CHECKING:
    import numpy as np
    from ...hrtree import tree as TreeType

logger = logging.getLogger(__name__)


# Cached MNI head + brain meshes. Both are bundled in the wheel as decimated
# fsaverage surfaces (~2.5k verts / 5k tris each, MNI-meter coords).
# Loaded lazily on first toggle-on and reused for every subsequent render.
#
# Two layers (researchers find both useful and they show different things):
#   - "pial"  → fsaverage lh.pial + rh.pial, stitched. The cortical surface.
#   - "scalp" → fsaverage bem/outer_skin.surf. The head's outer skin —
#               where forehead-mounted optodes physically sit, so HRF
#               points overlay this surface anatomically correctly.
_MESH_CACHE: Dict[str, Optional[Tuple["np.ndarray", "np.ndarray"]]] = {}

_MESH_FILENAMES = {
    "pial": "fsaverage_pial_lowpoly.npz",
    "scalp": "fsaverage_scalp_lowpoly.npz",
}


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
# ROI shift-hover paint — JS injection
# ---------------------------------------------------------------------------


_PAINT_HOOK_HEAD_INJECTED = False


def _ensure_shift_tracker_injected() -> None:
    """Inject a global shift-key tracker into the page head (once per process).

    The tracker writes ``window._hrfShift = True/False`` on Shift down/up,
    plus a safety reset on window blur (so a Shift-down outside the window
    followed by a release-outside doesn't leave a stuck shift state).

    Page-render-scope idempotency: every /library render calls this, but
    a module-level flag ensures we add the ``<script>`` to the document
    head only once per process. Subsequent calls are no-ops.
    """
    global _PAINT_HOOK_HEAD_INJECTED
    if _PAINT_HOOK_HEAD_INJECTED:
        return
    ui.add_head_html(
        """
<script>
  if (window._hrfShiftWired === undefined) {
    window._hrfShift = false;
    window._hrfShiftWired = true;
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Shift') window._hrfShift = true;
    });
    document.addEventListener('keyup', (e) => {
      if (e.key === 'Shift') window._hrfShift = false;
    });
    window.addEventListener('blur', () => { window._hrfShift = false; });
  }
</script>
"""
    )
    _PAINT_HOOK_HEAD_INJECTED = True


def _install_paint_hook(plot_id: int) -> None:
    """Hook the freshly-rendered plotly element's ``plotly_hover`` event.

    Plotly's native hover event includes the points data; what we need
    beyond that is the Shift-key state. We read it from the global
    ``window._hrfShift`` flag wired by :func:`_ensure_shift_tracker_injected`,
    then forward a custom ``roi_paint`` event up to the NiceGUI Python
    handler via the element's ``$emit``.

    A small ``ui.timer`` delay gives plotly's render cycle time to attach
    the underlying div to the DOM before we query it.
    """
    _ensure_shift_tracker_injected()

    js = f"""
const el = getElement({plot_id}).$el;
if (el && el.on && !el._hrfPaintHooked) {{
  el._hrfPaintHooked = true;
  el.on('plotly_hover', function(data) {{
    if (!window._hrfShift) return;
    if (!data || !data.points || !data.points.length) return;
    const key = data.points[0].customdata;
    if (!key) return;
    getElement({plot_id}).$emit('roi_paint', {{key: key}});
  }});
}}
"""

    ui.timer(0.5, lambda: ui.run_javascript(js), once=True)


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
            matched = filter_by_oxygenation(
                apply_filter(all_hrfs, state.library_filter),
                state.library_oxygenation,
            )
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

        # MNI overlay toggles. Drawn below the filter section because
        # they're viz-only switches, not data filters — grouping them
        # with the filters would confuse the count label semantics.
        # Two independent toggles so users can show either, both, or
        # neither.
        ui.separator()
        ui.label("Overlay").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )

        def _publish_filter_change() -> None:
            # The viz re-renders on library_filter_changed; reuse the
            # event rather than adding a separate one — payload is the
            # current filter, which the viz already handles correctly.
            state.publish("library_filter_changed", state.library_filter)

        def _on_brain_toggle(event) -> None:
            state.library_show_brain = bool(event.value)
            _publish_filter_change()

        def _on_scalp_toggle(event) -> None:
            state.library_show_scalp = bool(event.value)
            _publish_filter_change()

        ui.switch(
            "Show MNI brain",
            value=state.library_show_brain,
            on_change=_on_brain_toggle,
        ).tooltip(
            "Translucent fsaverage pial cortical surface beneath the "
            "HRF scatter — where the neural activity originates."
        )
        ui.switch(
            "Show MNI head",
            value=state.library_show_scalp,
            on_change=_on_scalp_toggle,
        ).tooltip(
            "Translucent fsaverage scalp (outer-skin) surface — where "
            "forehead/head-mounted fNIRS optodes physically sit."
        )

        # ── Oxygenation filter. Radio rather than a switch because
        # there are three meaningful states (both / HbO only / HbR
        # only) and a switch wouldn't surface the third.
        ui.label("Show oxygenation").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )

        def _on_oxygenation_change(event) -> None:
            state.library_oxygenation = event.value or "both"
            state.publish("library_filter_changed", state.library_filter)

        ui.radio(
            {"both": "Both", "hbo": "HbO only", "hbr": "HbR only"},
            value=state.library_oxygenation,
            on_change=_on_oxygenation_change,
        ).props("inline dense")

        # ── Region-of-Interest controls. Anchor is set by clicking an
        # HRF in the viz; the radius slider sweeps neighbours into the
        # ROI; Shift+hover adds individual HRFs (paint mode) for
        # non-spherical selections. The averaged ROI trace renders in
        # the detail pane (right side of the page).
        ui.separator()
        ui.label("Region of interest").classes(
            "text-xs uppercase opacity-60 tracking-wide"
        )
        ui.label(
            "Click an HRF to anchor. Hold Shift + hover to paint extra "
            "same-oxygenation HRFs into the selection. The detail pane "
            "shows the averaged trace."
        ).classes("text-xs opacity-60")

        radius_label = ui.label(
            f"Radius: {state.library_roi_radius_m * 100:.1f} cm"
        ).classes("text-xs font-mono opacity-80")

        def _on_radius_change(event) -> None:
            # The slider is in centimetres for human-readable steps; we
            # store metres on state so the geometric compute keeps its
            # MNI-meter native units.
            cm = float(event.value)
            state.library_roi_radius_m = cm / 100.0
            radius_label.set_text(f"Radius: {cm:.1f} cm")
            state.publish("library_filter_changed", state.library_filter)

        ui.slider(
            min=0.5, max=10.0, step=0.1,
            value=state.library_roi_radius_m * 100.0,
            on_change=_on_radius_change,
        ).props("dense")

        def _on_clear_roi() -> None:
            state.library_selected_hrf = None
            state.library_roi_painted.clear()
            state.publish("library_selection_changed", None)
            state.publish("library_filter_changed", state.library_filter)

        ui.button(
            "Clear ROI", on_click=_on_clear_roi
        ).props("flat dense")


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
        matched = filter_by_oxygenation(
            apply_filter(all_hrfs, state.library_filter),
            state.library_oxygenation,
        )

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
            # Compute ROI keys now so the figure draws the highlight
            # halo and the label can report the count.
            anchor = state.library_selected_hrf
            roi_keys = compute_roi_keys(
                matched,
                anchor,
                state.library_roi_radius_m,
                state.library_roi_painted,
            )
            roi_status = ""
            if roi_keys:
                roi_status = f"  •  ROI: {len(roi_keys)} highlighted"
            ui.label(
                f"HRtree — {len(matched)} HRFs shown{roi_status}"
            ).classes("text-sm opacity-70")
            fig = build_plotly_figure(
                matched,
                show_brain=state.library_show_brain,
                show_scalp=state.library_show_scalp,
                roi_keys=roi_keys,
            )
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
                # A plain click also resets the painted set — the user is
                # picking a fresh anchor, so accumulated shift-hover paint
                # from a prior anchor shouldn't carry over.
                state.library_selected_hrf = {**hrf, "_key": hrf_key}
                state.library_roi_painted.clear()
                state.publish("library_selection_changed", hrf_key)

            def _on_paint(event) -> None:
                # Shift+hover fired our custom roi_paint event with a key
                # in event.args. Add to painted set, refresh viz + detail.
                args = getattr(event, "args", None) or {}
                key = args.get("key") if isinstance(args, dict) else None
                if not key or key not in matched:
                    return
                # Only paint HRFs that match the anchor's oxygenation so
                # the average trace doesn't mix HbO + HbR (different
                # physiological signals, scientifically wrong to average).
                anchor_inner = state.library_selected_hrf
                if anchor_inner is not None:
                    if matched[key].get("oxygenation") != anchor_inner.get("oxygenation"):
                        return
                if key in state.library_roi_painted:
                    return  # already painted, no-op
                state.library_roi_painted.add(key)
                state.publish("library_selection_changed", key)

            plot.on("plotly_click", _on_click)
            plot.on("roi_paint", _on_paint)
            # Wire the JS shift-tracker + plotly_hover hook AFTER the
            # plotly element has rendered. Slight delay so the
            # underlying div is queryable in the DOM. once=True so
            # the hook isn't registered repeatedly on each refresh.
            _install_paint_hook(plot.id)

    _viz_body()

    # Re-render the viz on filter change.
    def _refresh_viz(_payload=None) -> None:
        _viz_body.refresh()
    state.subscribe("library_filter_changed", _refresh_viz)


def load_mesh(layer: str) -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Return a bundled MNI anatomical mesh as ``(vertices, faces)``.

    Args:
        layer: ``"pial"`` for the cortical surface (fsaverage lh.pial +
            rh.pial stitched) or ``"scalp"`` for the outer-skin head
            surface (fsaverage bem/outer_skin.surf). Anything else
            returns None.

    Both meshes are pre-decimated to ~2.5k verts / 5k triangles in
    MNI-meter coordinates so they overlay directly on bundled HRF
    locations without any transform. Results are cached per-layer at
    module scope; the first call per layer pays the .npz load cost.

    Returns None if the asset is missing, numpy can't be imported, or
    the requested layer is unknown. Callers fall back to no-overlay
    rendering rather than crashing.

    Why both layers: the pial shows where neural activity comes from
    (the cortical surface). The scalp shows where the optodes sit —
    forehead-mounted optodes are anatomically 5-10 mm beyond the pial,
    so they float visually outside the pial mesh and look "misaligned"
    even though their coordinates are correct. The scalp surface
    contains the optodes naturally. Users can show either, both, or
    neither.

    Mesh source: ``mne.surface.decimate_surface(method="quadric")``
    applied to fsaverage's ``lh.pial`` + ``rh.pial`` for the pial
    layer, and to ``bem/outer_skin.surf`` for the scalp layer. mm → m
    conversion baked in. Bundled in the wheel so no fsaverage download
    is required at runtime.
    """
    if layer in _MESH_CACHE:
        cached = _MESH_CACHE[layer]
        return cached
    filename = _MESH_FILENAMES.get(layer)
    if filename is None:
        logger.warning("load_mesh: unknown layer %r", layer)
        _MESH_CACHE[layer] = None
        return None
    try:
        import numpy as np
        from importlib import resources

        ref = resources.files("hrfunc.assets") / filename
        with resources.as_file(ref) as path:
            data = np.load(path)
            verts = data["vertices"]
            faces = data["faces"]
        _MESH_CACHE[layer] = (verts, faces)
        return _MESH_CACHE[layer]
    except Exception as exc:  # noqa: BLE001
        logger.warning("mesh load failed for layer=%r: %s", layer, exc)
        _MESH_CACHE[layer] = None
        return None


# Back-compat alias for the original Sprint 4 single-mesh loader. Kept so
# external callers (and the test suite) that imported ``load_brain_mesh``
# don't break. Defaults to the scalp layer because that's the new visible
# default in the GUI.
def load_brain_mesh() -> Optional[Tuple["np.ndarray", "np.ndarray"]]:
    """Deprecated alias for ``load_mesh("scalp")``."""
    return load_mesh("scalp")


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

            # ROI average plot (only renders when the ROI has at least
            # 2 averageable same-oxygenation HRFs — fewer than that
            # means there's nothing useful to average).
            _render_roi_average(state)

    _detail_body()

    def _refresh_detail(_payload=None) -> None:
        _detail_body.refresh()
    state.subscribe("library_selection_changed", _refresh_detail)
    # The radius slider publishes ``library_filter_changed`` because the
    # viz already listens there; the detail pane needs to refresh too
    # so the ROI-average plot updates when the user widens the radius.
    state.subscribe("library_filter_changed", _refresh_detail)


def _render_roi_average(state: AppState) -> None:
    """Render the averaged-trace plot for the current ROI.

    Pulls ``compute_roi_keys`` for the same set the viz highlights, then
    feeds those to ``compute_roi_average``. Renders nothing when the ROI
    has fewer than 2 averageable HRFs.
    """
    anchor = state.library_selected_hrf
    if anchor is None:
        return
    all_hrfs = gather_library_hrfs(state)
    matched = filter_by_oxygenation(
        apply_filter(all_hrfs, state.library_filter),
        state.library_oxygenation,
    )
    roi_keys = compute_roi_keys(
        matched,
        anchor,
        state.library_roi_radius_m,
        state.library_roi_painted,
    )
    result = compute_roi_average(matched, roi_keys)
    if result is None:
        return
    mean, std, n = result
    sfreq = float(anchor.get("sfreq") or 1.0)
    if sfreq <= 0:
        sfreq = 1.0
    ui.separator()
    ui.label(
        f"ROI average ({n} HRFs)"
    ).classes("text-xs uppercase opacity-60 tracking-wide")
    png = _render_roi_average_png(mean, std, sfreq, n)
    if png is not None:
        ui.image(png).classes("max-w-md")

    # ── Save-to-workspace button. Writes the averaged trace as JSON in
    # the same schema as hrtree HRF entries (plus ROI provenance fields)
    # so it can be re-loaded into hrfunc.tree if researchers want to
    # treat the ROI average as a derived HRF.
    def _on_save_roi() -> None:
        from ..workspace_io import save_roi_average, workspace_dir

        try:
            out_path = save_roi_average(
                anchor=anchor,
                roi_keys=roi_keys,
                hrf_mean=mean,
                hrf_std=std,
                sfreq=sfreq,
                radius_m=state.library_roi_radius_m,
                library_filter=state.library_filter,
            )
            ui.notify(
                f"Saved ROI average to {out_path.name} "
                f"({workspace_dir()})",
                type="positive",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("save ROI average failed: %s", exc)
            ui.notify(
                f"Save failed: {type(exc).__name__}: {exc}",
                type="negative",
            )

    with ui.row().classes("gap-2 mt-1"):
        ui.button(
            "Save ROI average",
            icon="download",
            on_click=_on_save_roi,
        ).props("color=primary dense")


def _render_roi_average_png(
    mean, std, sfreq: float, n: int
) -> Optional[str]:
    """Plot ROI-averaged trace with ±1 std shading."""
    try:
        import base64
        import io as _io
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for ROI average: %s", exc)
        return None
    fig = None
    try:
        t = np.arange(len(mean)) / sfreq
        fig, ax = plt.subplots(1, 1, figsize=(5, 2.5))
        ax.plot(t, mean, lw=1.4, color="#f59e0b", label=f"mean (n={n})")
        ax.fill_between(
            t, mean - std, mean + std,
            alpha=0.18, color="#f59e0b",
            label="±1 std",
        )
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (a.u.)")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("ROI average render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _kv(key: str, value: str) -> None:
    with ui.row().classes("w-full gap-4"):
        ui.label(key).classes("text-xs uppercase opacity-60 w-32")
        ui.label(value).classes("text-sm break-all")


# ---------------------------------------------------------------------------
# Data helpers (module-level so tests can call them)
# ---------------------------------------------------------------------------


def gather_library_hrfs(state: AppState) -> Dict[str, Dict[str, Any]]:
    """Combine the HbO + HbR trees into a single name → HRF-dict map.

    Returns empty dict if the trees aren't loaded.

    Two filters applied while merging:

    1. **Global sentinels excluded.** ``montage.estimate_hrf`` and friends
       seed every Montage with ``global_hbo`` / ``global_hbr`` placeholder
       entries at the sentinel location ``[~360, ~360, ~360]`` (out-of-
       MNI-range so they don't collide with real optodes — see
       ``montage._merge_montages`` and ``tree.get_canonical_hrf``). Those
       entries leak into the bundled HRF databases when a researcher
       saves their montage. They have no business in the user-facing
       library browser, and at ``[360, 360, 360]`` they dominate
       plotly's ``aspectmode="data"`` axis range, compressing the real
       optode cluster (~0.07 m) to a single invisible pixel. Skip
       anything whose key starts with ``global_``.
    2. **Re-keyed by oxygenation prefix.** The bundled HbO and HbR
       JSONs share at least one key (``s8_d4_hbr-temp`` appears in
       both — community-contributed entries can be duplicated across
       files). A plain ``dict.update`` would silently drop one copy on
       collision. Prefixing with ``hbo:`` / ``hbr:`` preserves both
       oxygenation flavors even when their optode-pair keys match.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tree_obj, prefix in (
        (state.library_hbo, "hbo:"),
        (state.library_hbr, "hbr:"),
    ):
        if tree_obj is None:
            continue
        hrfs = tree_obj.gather(tree_obj.root)
        if not hrfs:
            continue
        for key, hrf in hrfs.items():
            if key.startswith("global_"):
                continue
            out[f"{prefix}{key}"] = hrf
    return out


def filter_by_oxygenation(
    hrfs: Dict[str, Dict[str, Any]],
    mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Filter HRFs by oxygenation channel.

    Args:
        hrfs: name → HRF-dict map (post-context-filter).
        mode: ``"both"`` returns unchanged; ``"hbo"`` keeps only HRFs
            with ``oxygenation is True``; ``"hbr"`` keeps only HRFs
            with ``oxygenation is False``. Unknown mode strings
            fall through as ``"both"`` so a typo doesn't blank the
            entire viz.

    Module-level so tests can hit it without spinning up the GUI.
    """
    if mode == "hbo":
        return {k: v for k, v in hrfs.items() if v.get("oxygenation") is True}
    if mode == "hbr":
        return {k: v for k, v in hrfs.items() if v.get("oxygenation") is False}
    return dict(hrfs)


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


def build_plotly_figure(
    hrfs: Dict[str, Dict[str, Any]],
    *,
    show_brain: bool = False,
    show_scalp: bool = False,
    roi_keys: Optional[Any] = None,
):
    """Build the 3D scatter figure for the given HRF dict.

    Up to five traces, ordered so the ROI highlight renders on top:

    - **Scalp** (``go.Mesh3d``, only when ``show_scalp=True``):
      fsaverage outer-skin surface — anatomically where the optodes
      sit. Drawn first (outermost in 3D-painter order).
    - **Brain** (``go.Mesh3d``, only when ``show_brain=True``):
      fsaverage pial cortical surface — where the neural activity
      originates. Drawn inside the scalp.
    - **HbO** (``go.Scatter3d``, red): oxygenated HRFs.
    - **HbR** (``go.Scatter3d``, blue): deoxygenated HRFs.
    - **ROI** (``go.Scatter3d``, gold, larger): every HRF whose key
      is in ``roi_keys``. Drawn last so the highlight sits above the
      regular markers. Skipped when ``roi_keys`` is None or empty.

    Mesh overlays have ``hoverinfo="skip"`` and ``showlegend=False``
    so they don't clutter legend / hover UX. Each scatter point's
    ``customdata`` is the HRF key for click + shift-hover handlers.
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

    # Overlay meshes first (scalp outside, brain inside, both more
    # transparent than the HRF markers) so the HRF scatter renders on
    # top. Scalp is drawn FIRST so it's the outermost in 3D-painter
    # order — when both are on, the brain visually nests inside the
    # head.
    if show_scalp:
        mesh = load_mesh("scalp")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color="#c4b5a0",  # warm skin tone
                    opacity=0.12,
                    name="MNI head",
                    hoverinfo="skip",
                    showlegend=False,
                    lighting=dict(ambient=0.6, diffuse=0.5),
                )
            )
    if show_brain:
        mesh = load_mesh("pial")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color="#9ca3af",  # cool grey for cortex
                    opacity=0.30,
                    name="MNI brain",
                    hoverinfo="skip",
                    showlegend=False,
                    lighting=dict(ambient=0.5, diffuse=0.6),
                )
            )

    if hbo_x:
        traces.append(
            go.Scatter3d(
                x=hbo_x, y=hbo_y, z=hbo_z,
                mode="markers",
                # HbO and HbR for the same optode pair share the exact 3D
                # location (one source-detector → two measurements at one
                # spot). Plotly Scatter3d draws traces in order, so without
                # distinct symbols the second trace fully occludes the first.
                # Distinct symbols + sizes keep both visible at the same xyz.
                marker=dict(
                    size=6,
                    color="#fb7185",
                    opacity=0.9,
                    symbol="circle",
                    line=dict(width=1, color="#7f1d1d"),
                ),
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
                marker=dict(
                    size=4,
                    color="#38bdf8",
                    opacity=0.85,
                    symbol="diamond",
                    line=dict(width=1, color="#0c4a6e"),
                ),
                name="HbR",
                customdata=hbr_keys,
                hovertext=hbr_hover,
                hoverinfo="text",
            )
        )

    # ROI highlight — drawn LAST so the gold markers visually sit above
    # the regular HbO/HbR scatter (same points still appear in the
    # underlying trace; the ROI layer is an emphasis halo).
    if roi_keys:
        roi_x, roi_y, roi_z, roi_keys_list, roi_hover = [], [], [], [], []
        for key in roi_keys:
            hrf = hrfs.get(key)
            if hrf is None:
                continue
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            roi_x.append(loc[0])
            roi_y.append(loc[1])
            roi_z.append(loc[2])
            roi_keys_list.append(key)
            roi_hover.append(_hover_text_for(key, hrf))
        if roi_x:
            traces.append(
                go.Scatter3d(
                    x=roi_x, y=roi_y, z=roi_z,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="#fbbf24",   # gold
                        opacity=0.9,
                        line=dict(width=1, color="#92400e"),
                    ),
                    name=f"ROI ({len(roi_x)})",
                    customdata=roi_keys_list,
                    hovertext=roi_hover,
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


def compute_roi_keys(
    hrfs: Dict[str, Dict[str, Any]],
    anchor: Optional[Dict[str, Any]],
    radius_m: float,
    painted: Optional[Any] = None,
) -> "set":
    """Return the set of HRF keys that belong to the current ROI.

    Membership rules:
    - If ``anchor`` is set and has a 3-element ``location``, every HRF
      with the SAME ``oxygenation`` as the anchor whose Euclidean
      distance to the anchor is ``<= radius_m`` is included.
    - Every key in ``painted`` is included (filtered to the anchor's
      oxygenation when an anchor is set, so a stray paint on the
      wrong haemoglobin doesn't contaminate the average).

    The anchor's own key is always part of the ROI when it's still in
    ``hrfs`` (otherwise filtering away the anchor's neighbourhood
    would be confusing).

    Module-level so tests can hit it without spinning up the GUI.
    """
    out: set = set()
    if anchor is None and not painted:
        return out

    anchor_loc = None
    anchor_oxy = None
    anchor_key = None
    if anchor is not None:
        anchor_loc = anchor.get("location")
        anchor_oxy = anchor.get("oxygenation")
        anchor_key = anchor.get("_key")
        if anchor_key is not None and anchor_key in hrfs:
            out.add(anchor_key)

    if anchor_loc is not None and len(anchor_loc) >= 3 and radius_m > 0:
        ax, ay, az = float(anchor_loc[0]), float(anchor_loc[1]), float(anchor_loc[2])
        r2 = float(radius_m) * float(radius_m)
        for key, hrf in hrfs.items():
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            if anchor_oxy is not None and hrf.get("oxygenation") != anchor_oxy:
                continue
            dx, dy, dz = float(loc[0]) - ax, float(loc[1]) - ay, float(loc[2]) - az
            if dx * dx + dy * dy + dz * dz <= r2:
                out.add(key)

    if painted:
        for key in painted:
            if key not in hrfs:
                continue
            if anchor_oxy is not None:
                if hrfs[key].get("oxygenation") != anchor_oxy:
                    continue
            out.add(key)

    return out


def compute_roi_average(
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
):
    """Average the ``hrf_mean`` arrays of all HRFs in the ROI.

    Returns ``(mean, std, n_used)`` numpy arrays + a count, or None if
    fewer than 2 traces in the ROI are averageable.

    Skips HRFs with empty / None / mismatched-length ``hrf_mean``.
    Anchor's trace length is the canonical length; HRFs whose traces
    don't match are skipped (we'd otherwise smear different durations
    together silently).

    Module-level so tests can call without a UI.
    """
    import numpy as np
    from collections import Counter

    # First pass: parse every candidate trace into a numpy array.
    candidates: List[np.ndarray] = []
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        trace = hrf.get("hrf_mean")
        if trace is None:
            continue
        try:
            arr = np.asarray(trace, dtype=float)
        except Exception:  # noqa: BLE001
            continue
        if arr.size == 0:
            continue
        candidates.append(arr)

    if len(candidates) < 2:
        return None

    # Pick the MODAL length as the canonical one rather than the first
    # iterated one. ``roi_keys`` is typically a set (no order
    # guarantees), and taking the first-seen length means an outlier
    # length could throw out the majority. Modal length is robust to
    # iteration order and matches what a researcher would expect:
    # "average the traces that share the typical duration; skip the
    # oddball".
    lengths = Counter(arr.shape[0] for arr in candidates)
    canonical_len = lengths.most_common(1)[0][0]
    traces = [arr for arr in candidates if arr.shape[0] == canonical_len]

    if len(traces) < 2:
        return None

    stacked = np.vstack(traces)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=0), len(traces)


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
