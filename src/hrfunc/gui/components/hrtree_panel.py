"""HRtree panel — embeddable HRF-tree explorer.

This module is the Library/HRtree implementation extracted from the
Sprint 4 ``/library`` route so the v1.4 single-shell GUI can mount it as a
tab without dragging route-specific chrome with it. Three deliberate
differences from the legacy ``pages.library`` code it was ported from:

1. **No toolbar.** The shell renders a single brand wordmark + project
   picker; the panel renders only the filter / viz / detail panes.
2. **No ``state.subscribers.clear()``.** The legacy ``/library`` and
   ``/workspace`` route handlers cleared the subscriber list on every
   render so repeat visits didn't accumulate stale refreshable
   handles. In the single-shell model that clear-on-render is a
   footgun — it nukes other tabs' subscriptions. Subscribers are
   instead cleared only on project switch (Phase 3 work).
3. **Event prefix is ``hrtree_*``.** Legacy events were prefixed
   ``library_*``; the rename frees up the namespace for the future
   "Library / Project / Both" data-source toggle where project-side
   events live alongside.

Public API:
    render(state, *, data_source="library")
        Mount the three-pane HRtree explorer. ``data_source`` is
        plumbed for the future toggle but only ``"library"`` is wired
        in Phase 1 — passing other values currently falls back to
        library-tree rendering.

Pure helpers (``gather_library_hrfs``, ``apply_filter``,
``filter_by_oxygenation``, ``compute_roi_keys``, ``compute_roi_average``,
``build_plotly_figure``, ``load_mesh``) stay module-level so tests can
call them without a UI context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple

from nicegui import ui

from ...spatial.atlas import load_harvard_oxford_cortical
from ...spatial.coords import meters_to_mm
from ...spatial.shapes import AtlasRegion, Box, Shape, Sphere
from ...viz.brain_scene import (
    make_box_overlay_trace,
    make_sphere_overlay_trace,
    make_surface_trace,
)
from ...viz.meshes import MESH_CACHE as _MESH_CACHE
from ...viz.meshes import MESH_FILENAMES as _MESH_FILENAMES
from ...viz.meshes import load_brain_mesh, load_mesh
from . import brand
from ..state import AppState

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Mesh loaders were moved to :mod:`hrfunc.viz.meshes` during the v1.3
# spatial/viz compartmentalization refactor. The names are re-exported
# here so existing callers (``library.load_mesh``, ``library.load_brain_mesh``,
# ``library._MESH_CACHE``, ``library._MESH_FILENAMES``) and the v1.3.0
# test suite continue to work without import-path churn. New code should
# import from ``hrfunc.viz.meshes`` directly.
__all__ = (
    "load_mesh",
    "load_brain_mesh",
    "_MESH_CACHE",
    "_MESH_FILENAMES",
)


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


# Available shape modes for the Cluster sub-tab's ROI selector.
SHAPE_SPHERE = "sphere"
SHAPE_BOX = "box"
SHAPE_ATLAS_REGION = "atlas_region"
SHAPE_MODES = (SHAPE_SPHERE, SHAPE_BOX, SHAPE_ATLAS_REGION)


def _resolve_cluster_oxygenation(state: AppState) -> Optional[bool]:
    """Pick the oxygenation filter the Cluster sub-tab should apply.

    Returns ``True`` to keep HbO only, ``False`` to keep HbR only,
    ``None`` to skip oxygenation filtering and let mixed-haemoglobin
    HRFs into the ROI.

    Precedence (matches the v1.2 anchor+radius behaviour and extends
    it for free-floating modes):

    1. If a click-anchor is set, use the anchor's oxygenation.
       Averaging mixed-haemoglobin traces is scientifically wrong --
       same rationale as the original ``compute_roi_keys`` behaviour.
    2. Else if the filter sub-tab's oxygenation is ``"hbo"`` or
       ``"hbr"``, route the binary through.
    3. Else (filter is ``"both"`` with no anchor), return None --
       the user has explicitly opted into mixed visibility, and
       silently filtering would surprise them.
    """
    anchor = state.library_selected_hrf
    if anchor is not None and anchor.get("oxygenation") is not None:
        return bool(anchor.get("oxygenation"))
    if state.library_oxygenation == "hbo":
        return True
    if state.library_oxygenation == "hbr":
        return False
    return None


def _build_current_shape(state: AppState) -> Optional[Shape]:
    """Build the spatial-layer :class:`Shape` for the current Cluster mode.

    Sphere mode:
        Centre = ``(cluster_center_*_mm)``. Radius =
        ``library_roi_radius_m * 1000``. Note that the cluster centre
        is normally seeded from a clicked HRF (see the viz pane's
        click handler), so in the legacy "click and adjust radius"
        workflow the resulting sphere matches the v1.2 anchor-based
        sphere.

    Box mode:
        Centre = ``(cluster_center_*_mm)``. Half-extents =
        ``(cluster_box_half_*_mm)``. Always free-floating.

    Returns None for unknown shape modes (defensive -- a stale state
    value from a future / legacy build shouldn't crash the render).

    PR #54: returns None when ``state.cluster_roi_active`` is False so
    the gold halo + shape overlay + save button stay quiet by default;
    researchers opt into ROI mode explicitly via the toggle at the top
    of the Cluster sub-tab.
    """
    if not state.cluster_roi_active:
        return None
    return _build_shape_unconditional(state)


def _build_shape_unconditional(state: AppState) -> Optional[Shape]:
    """Same as :func:`_build_current_shape` but ignores the ROI-active
    toggle. The Cluster sub-tab UI needs to render shape-specific
    controls (sphere radius, atlas dropdown) regardless of whether
    the toggle is on -- otherwise turning the toggle off would
    collapse the entire UI body and disorient the user."""
    if state.cluster_shape == SHAPE_BOX:
        return Box(
            center_mm=(
                state.cluster_center_x_mm,
                state.cluster_center_y_mm,
                state.cluster_center_z_mm,
            ),
            half_extents_mm=(
                state.cluster_box_half_x_mm,
                state.cluster_box_half_y_mm,
                state.cluster_box_half_z_mm,
            ),
        )
    if state.cluster_shape == SHAPE_ATLAS_REGION:
        # Atlas mode needs both the loaded atlas and a selected region.
        # If either is missing we return None so callers fall back to
        # "no ROI yet" UI (the save button disables, the viz skips the
        # shape-membership filter).
        if not state.cluster_atlas_label:
            return None
        atlas = load_harvard_oxford_cortical()
        if atlas is None:
            return None
        try:
            return AtlasRegion(atlas, state.cluster_atlas_label)
        except ValueError:
            # Label no longer in atlas (e.g. user state from a future
            # version with a richer atlas). Treat as unselected.
            return None
    if state.cluster_shape == SHAPE_SPHERE:
        return Sphere(
            center_mm=(
                state.cluster_center_x_mm,
                state.cluster_center_y_mm,
                state.cluster_center_z_mm,
            ),
            radius_mm=float(meters_to_mm(state.library_roi_radius_m)),
        )
    return None


def _build_atlas_alignment_affine(state: AppState) -> "Optional[np.ndarray]":
    """Compose the full HRF-coord -> MNI mm affine for atlas lookups.

    Returns ``None`` when the alignment is identity (no transform
    needed) -- callers fast-path the lookup.

    The user can provide alignment two ways:

    1. A full 4x4 ``cluster_atlas_alignment_affine`` loaded from a
       JSON or .npy file via the file picker in atlas mode.
    2. Three pure-translation offsets (``cluster_atlas_offset_*_mm``)
       for users without a registered affine.

    Both compose -- offsets translate AFTER the affine. Identity
    affine + zero offsets returns None so callers know they can
    skip the transform.
    """
    import numpy as np

    ox = float(state.cluster_atlas_offset_x_mm)
    oy = float(state.cluster_atlas_offset_y_mm)
    oz = float(state.cluster_atlas_offset_z_mm)
    has_offset = ox != 0.0 or oy != 0.0 or oz != 0.0
    has_affine = state.cluster_atlas_alignment_affine is not None

    if not has_offset and not has_affine:
        return None

    if has_affine:
        affine = np.asarray(
            state.cluster_atlas_alignment_affine, dtype=np.float64
        )
        if affine.shape != (4, 4):
            # Defensive: stale state; ignore and fall back to identity.
            affine = np.eye(4, dtype=np.float64)
    else:
        affine = np.eye(4, dtype=np.float64)

    if has_offset:
        translation = np.eye(4, dtype=np.float64)
        translation[:3, 3] = (ox, oy, oz)
        # Translation applied AFTER the affine ("T @ A @ point").
        affine = translation @ affine

    return affine


def _alignment_for_shape(state: AppState, shape: Optional[Shape]) -> "Optional[np.ndarray]":
    """Return the HRF -> atlas alignment affine when applicable, else None.

    Atlas mode needs the alignment because library HRFs are stored in
    MNE head coords (origin near auditory meatus) while the bundled
    atlas is in MNI mm. Sphere / Box modes don't need alignment --
    their geometry lives in the same frame as the HRF locations.
    """
    if not isinstance(shape, AtlasRegion):
        return None
    return _build_atlas_alignment_affine(state)


DataSource = Literal["library", "project", "both"]


def render(state: AppState, *, data_source: DataSource = "library") -> None:
    """Render the HRtree explorer panel against the given AppState.

    Lazy-loads the bundled HRF trees on first call; subsequent calls
    reuse the cached state. The three-pane layout (filter / viz / detail)
    is the same as the legacy ``/library`` page minus the toolbar.

    :param state: AppState singleton (or a synthetic one in tests).
    :param data_source: ``"library"`` shows bundled literature HRFs;
        ``"project"`` and ``"both"`` are reserved for the future
        project-HRF integration and currently fall back to library
        rendering.
    """
    # The data_source param is plumbed for forward compat; the project /
    # both code paths aren't wired yet (the panel renders library data
    # regardless). Once project HRFs are sourced from state.montage in a
    # future phase, this becomes a true switch.
    _ = data_source

    if state.library_hbo is None or state.library_hbr is None:
        _load_trees(state)

    _render_three_pane(state)


def _load_trees(state: AppState) -> None:
    """Read the bundled HRF databases into memory once.

    The trees stay on state for the lifetime of the process. Failures are
    surfaced to ``state.last_error`` but the panel still renders so users
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
        # ``rich=True`` keeps the per-subject ``estimates`` lists on each
        # HRF node so ROI averaging can pool subject-level traces (the
        # statistically correct grand mean). The default ``rich=False``
        # strips them to save memory; for the GUI we accept the ~1-2 MB
        # cost in exchange for accurate ROI averages.
        state.library_hbo = Tree(hbo_path, rich=True)
        state.library_hbr = Tree(hbr_path, rich=True)
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

    Module-level idempotency flag ensures we add the ``<script>`` to the
    document head only once per process. Subsequent calls are no-ops.
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
# Three-pane layout — splitters for left | center | right
# ---------------------------------------------------------------------------


def _render_three_pane(state: AppState) -> None:
    """Render left | viz | detail with two splitters.

    ``h-full`` (vs the legacy ``h-screen``) sizes the splitter to fill
    its parent container — the shell's tab-panels — instead of the full
    viewport. The shell's toolbar already consumes the top of the
    viewport, so ``h-screen`` here would push the bottom of the panel
    below the fold and cause page-level scrolling.

    The left pane hosts the HR_tree_ wordmark plus Filter / Cluster
    sub-tabs (Phase 6 redesign); the center is the 3D viz at full
    height; the right is the HRF detail card.
    """
    with ui.splitter(value=22, limits=(15, 35)).classes(
        "w-full h-full"
    ) as outer:
        with outer.before:
            _render_left_pane(state)
        with outer.after:
            with ui.splitter(value=68, limits=(40, 90)).classes(
                "w-full h-full"
            ) as inner:
                with inner.before:
                    _render_viz_pane(state)
                with inner.after:
                    _render_detail_pane(state)


# ---------------------------------------------------------------------------
# Left pane: HR_tree_ wordmark + Filter / Cluster sub-tabs
# ---------------------------------------------------------------------------


# Sub-tab labels for the left pane. ``Filter`` holds the context inputs,
# overlay toggles, oxygenation radio, and ROI radius slider. ``Cluster``
# holds actions that commit the current ROI to disk (save averaged
# trace) plus room for future clustering scripts.
SUBTAB_FILTER = "Filter"
SUBTAB_CLUSTER = "Cluster"
SUBTAB_NAMES = (SUBTAB_FILTER, SUBTAB_CLUSTER)


def _render_left_pane(state: AppState) -> None:
    """Wordmark + sub-tabs at the top, active sub-tab content below.

    The HR_tree_ Brand wordmark lives here (not in the shell) so the
    branding sits inside the panel where the user is looking. Sub-tabs
    keep filter and cluster actions docked side-by-side rather than
    competing for vertical space; switching tabs is one click.
    """
    with ui.column().classes("w-full h-full p-3 gap-2 overflow-hidden"):
        # Brand wordmark + one-line subtitle. Compact size_rem so the
        # header doesn't dominate the pane.
        with ui.row().classes("items-center gap-2 w-full"):
            brand.brand("HRtree", italic_suffix="tree", size_rem=1.3)
        ui.label(
            "3D spatial database of literature HRFs."
        ).classes("text-xs opacity-60")

        # Sub-tabs.
        with ui.tabs().props("dense").classes("w-full") as subtabs:
            for name in SUBTAB_NAMES:
                ui.tab(name)
        # ``min-h-0`` is required alongside ``flex-1`` for the
        # ``overflow-hidden`` boundary to actually clip — without it,
        # the Filter sub-tab's content height pushes the column past
        # the splitter pane and the left side ends up taller than the
        # viz on the right (causing the mismatch you'd otherwise see).
        with ui.tab_panels(subtabs, value=SUBTAB_FILTER).classes(
            "w-full flex-1 min-h-0 overflow-hidden"
        ):
            with ui.tab_panel(SUBTAB_FILTER).classes(
                "p-0 h-full overflow-auto"
            ):
                _render_filter_subtab(state)
            with ui.tab_panel(SUBTAB_CLUSTER).classes(
                "p-0 h-full overflow-auto"
            ):
                _render_cluster_subtab(state)


def _render_filter_subtab(state: AppState) -> None:
    """The Filter sub-tab -- oxygenation + context inputs.

    Filter sub-tab owns "what's visible" (oxygenation, context filters);
    the Cluster sub-tab owns "what's in the ROI" (shape, radius, paint,
    clear). PR #49 moved the radius slider + Clear ROI button to the
    Cluster sub-tab so the two sub-tabs have clearly separated
    responsibilities.

    Each FILTER_FIELDS entry becomes a text input bound to
    ``state.library_filter[field]`` -- empty string = field not
    filtered. "Apply" then refreshes the dependent viz + detail panes.
    """
    with ui.column().classes("w-full gap-3"):
        # -- Oxygenation radio (frequently toggled, top placement)

        def _on_oxygenation_change(event) -> None:
            state.library_oxygenation = event.value or "both"
            state.publish("hrtree_filter_changed", state.library_filter)

        ui.radio(
            {"both": "Both", "hbo": "HbO only", "hbr": "HbR only"},
            value=state.library_oxygenation,
            on_change=_on_oxygenation_change,
        ).props("inline dense")

        # ── Context inputs (set-once / refine-slowly)
        ui.separator()
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
            state.publish("hrtree_filter_changed", new_filter)

        def _reset() -> None:
            for widget in inputs.values():
                widget.value = ""
            state.library_filter = {}
            state.publish("hrtree_filter_changed", {})

        with ui.row().classes("w-full gap-2"):
            ui.button("Apply", on_click=_apply).props("color=primary dense")
            ui.button("Reset", on_click=_reset).props("flat dense")

        # Live match count — annotates how many filtered HRFs are
        # invisible in the 3D viz for lacking a location, so users
        # aren't confused by "5 / 22 match" while the viz shows 3 points.
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
            "hrtree_filter_changed", lambda _p=None: _count_label.refresh()
        )


def _render_cluster_subtab(state: AppState) -> None:
    """The Cluster sub-tab -- ROI shape selection, sizing, save action.

    Two shape modes since PR #53:

    - **Sphere** (default): the centre + radius selection from
      PR #49. Centre seeds from clicks in the viz; radius slider
      lives in this sub-tab.
    - **Atlas region**: pick a Harvard-Oxford cortical region from
      a dropdown; the ROI is every HRF whose MNI coordinate lies in
      that region's voxel mask. Region name + "Region at centre"
      readout serve as the methods-section provenance line.

    Box mode is hidden from the UI; the underlying class is still
    available for the v1.4 rotatable-box UI work (PR #52 made it
    orientation-aware).

    Contents:

    - **Shape radio**: Sphere | Atlas region.
    - **Centre inputs**: three MNI-mm number inputs. Visible in
      both modes -- they drive the atlas readout even when not
      driving membership.
    - **Radius slider** (sphere only).
    - **Region dropdown** (atlas only).
    - **Clear ROI button**: drops the anchor + painted set.
    - **MNI readout** + **Region-at-centre readout**: copy-pasteable
      methods-section provenance.
    - **Save ROI average**: writes the averaged trace + shape
      metadata to the workspace folder.
    """

    @ui.refreshable
    def _body() -> None:
        # Load the atlas lazily on first sub-tab render. The loader
        # caches per-process so repeat renders pay nothing; sphere-only
        # users still get the per-render cost (small, ~ms) but in
        # exchange the atlas readout works in sphere mode too.
        atlas = load_harvard_oxford_cortical()

        with ui.column().classes("w-full gap-3"):
            ui.label("Cluster").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )

            # --- ROI active toggle (PR #54) ---------------------------
            # Default off so a fresh page load shows raw HRFs without a
            # mystery gold halo around (0, 0, 0). Researchers explicitly
            # opt in to ROI mode before the shape, centre, radius, etc.
            # contribute to membership.
            def _on_roi_active_change(event) -> None:
                state.cluster_roi_active = bool(event.value)
                state.publish("hrtree_filter_changed", state.library_filter)

            ui.switch(
                "ROI active",
                value=state.cluster_roi_active,
                on_change=_on_roi_active_change,
            ).props("dense").tooltip(
                "Turn the ROI on to highlight matching HRFs in the viz "
                "and enable Save. Off by default so the page loads "
                "without a default ROI applied."
            )

            # --- Shape radio -----------------------------------------
            shape_options = {SHAPE_SPHERE: "Sphere"}
            if atlas is not None:
                shape_options[SHAPE_ATLAS_REGION] = "Atlas region"
            # Defensive: if persisted state has atlas mode but atlas
            # failed to load, fall back to sphere so the user isn't
            # stuck on a dead option.
            current_shape = state.cluster_shape
            if current_shape not in shape_options:
                current_shape = SHAPE_SPHERE
                state.cluster_shape = SHAPE_SPHERE

            def _on_shape_change(event) -> None:
                new_shape = event.value
                if new_shape not in shape_options:
                    return
                state.cluster_shape = new_shape
                state.publish("hrtree_filter_changed", state.library_filter)
                _body.refresh()

            ui.radio(
                shape_options,
                value=current_shape,
                on_change=_on_shape_change,
            ).props("inline dense")

            if state.cluster_shape == SHAPE_SPHERE:
                ui.label(
                    "Place the ROI sphere in MNI mm. Click an HRF in "
                    "the viz to seed the centre, or type coordinates "
                    "directly."
                ).classes("text-xs opacity-60")
            elif state.cluster_shape == SHAPE_ATLAS_REGION:
                ui.label(
                    "Pick a Harvard-Oxford cortical region. The ROI "
                    "includes every HRF whose MNI coordinate falls "
                    "inside the region's voxel mask."
                ).classes("text-xs opacity-60")

            # --- Centre inputs (MNI mm) ------------------------------
            # Always visible: drives the atlas readout in both modes,
            # and is also the sphere centre in sphere mode.
            ui.label("Centre (MNI mm)").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )

            def _make_centre_input(axis: str, attr: str):
                def _on_change(event) -> None:
                    try:
                        value = float(event.value or 0.0)
                    except (TypeError, ValueError):
                        return
                    setattr(state, attr, value)
                    state.publish("hrtree_filter_changed", state.library_filter)

                return ui.number(
                    label=axis,
                    value=getattr(state, attr),
                    step=1.0,
                    format="%.1f",
                    on_change=_on_change,
                ).props("dense").classes("w-20")

            with ui.row().classes("w-full gap-2"):
                _make_centre_input("x", "cluster_center_x_mm")
                _make_centre_input("y", "cluster_center_y_mm")
                _make_centre_input("z", "cluster_center_z_mm")

            # --- Sphere radius (sphere mode only) --------------------
            if state.cluster_shape == SHAPE_SPHERE:
                ui.label("ROI radius").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )
                radius_label = ui.label(
                    f"{state.library_roi_radius_m * 100:.1f} cm"
                ).classes("text-xs font-mono opacity-80")

                def _on_radius_change(event) -> None:
                    cm = float(event.value)
                    state.library_roi_radius_m = cm / 100.0
                    radius_label.set_text(f"{cm:.1f} cm")
                    state.publish(
                        "hrtree_filter_changed", state.library_filter
                    )

                ui.slider(
                    min=0.5, max=10.0, step=0.1,
                    value=state.library_roi_radius_m * 100.0,
                    on_change=_on_radius_change,
                ).props("dense")

            # --- Atlas region dropdown (atlas mode only) -------------
            if state.cluster_shape == SHAPE_ATLAS_REGION and atlas is not None:
                ui.label("Atlas region").classes(
                    "text-xs uppercase opacity-60 tracking-wide"
                )

                def _on_region_change(event) -> None:
                    state.cluster_atlas_label = event.value or None
                    state.publish(
                        "hrtree_filter_changed", state.library_filter
                    )

                ui.select(
                    options=atlas.region_names,
                    value=state.cluster_atlas_label,
                    label="Region",
                    on_change=_on_region_change,
                ).props("dense outlined").classes("w-full")

                # --- Atlas alignment (HRF coords -> MNI mm) ----------
                # Library HRFs are stored in MNE head coords (origin
                # near auditory meatus); the atlas is in MNI mm.
                # Without alignment, every lookup falls outside the
                # atlas volume -> silently empty ROIs. Two UI paths:
                # full 4x4 affine upload (for users with a registered
                # head->MNI transform) or pure-translation offsets
                # (a one-click "shift everything"). They compose.
                _render_atlas_alignment_section(state, _body)

            # --- Clear ROI button ------------------------------------
            def _on_clear_roi() -> None:
                state.library_selected_hrf = None
                state.library_roi_painted.clear()
                state.publish("hrtree_selection_changed", None)
                state.publish("hrtree_filter_changed", state.library_filter)

            ui.button(
                "Clear ROI", on_click=_on_clear_roi
            ).props("flat dense")

            # --- MNI + atlas readouts --------------------------------
            ui.separator()
            ui.label(_format_shape_readout(state)).classes(
                "text-xs font-mono opacity-80 break-all"
            )
            if atlas is not None:
                # Atlas readout is shown in BOTH modes so sphere users
                # can see "my centre sits in: Frontal Pole" without
                # switching modes -- useful navigation aid.
                region_at_centre = atlas.region_at((
                    state.cluster_center_x_mm,
                    state.cluster_center_y_mm,
                    state.cluster_center_z_mm,
                ))
                centre_region_text = (
                    f"Region at centre: {region_at_centre}"
                    if region_at_centre is not None
                    else "Region at centre: (outside atlas / background)"
                )
                ui.label(centre_region_text).classes(
                    "text-xs font-mono opacity-70"
                )

            # --- ROI status + Save button ----------------------------
            all_hrfs = gather_library_hrfs(state)
            matched = filter_by_oxygenation(
                apply_filter(all_hrfs, state.library_filter),
                state.library_oxygenation,
            )
            shape = _build_current_shape(state)
            oxy_filter = _resolve_cluster_oxygenation(state)
            alignment = _alignment_for_shape(state, shape)
            roi_keys = compute_roi_keys_by_shape(
                matched, shape, state.library_roi_painted,
                oxygenation_filter=oxy_filter,
                alignment_affine=alignment,
            )
            roi_result = compute_roi_average(matched, roi_keys)
            excluded_count = compute_roi_excluded_count(matched, roi_keys)
            can_save = roi_result is not None

            if roi_result is None:
                ui.label(
                    "ROI has fewer than 2 averageable subject estimates "
                    "in the current shape. Either widen the shape, paint "
                    "more neighbours, or seed a different centre. (Note: "
                    "HRFs without per-subject estimates are excluded.)"
                ).classes("text-xs opacity-60 italic")
            else:
                _, _, n_subjects, n_channels = roi_result
                ui.label(
                    f"Current ROI: averaging {n_subjects} subject "
                    f"estimates across {n_channels} channel"
                    f"{'s' if n_channels != 1 else ''}."
                ).classes("text-xs opacity-70")
                if excluded_count > 0:
                    ui.label(
                        f"  ({excluded_count} HRF"
                        f"{'s' if excluded_count != 1 else ''} excluded "
                        f"for lacking subject-level estimates.)"
                    ).classes("text-xs opacity-50 italic")

            def _on_save_roi() -> None:
                # Recompute at click time so we save what the user sees
                # right now (state may have changed between mount and
                # click).
                _matched = filter_by_oxygenation(
                    apply_filter(gather_library_hrfs(state),
                                 state.library_filter),
                    state.library_oxygenation,
                )
                _shape = _build_current_shape(state)
                _oxy = _resolve_cluster_oxygenation(state)
                _alignment = _alignment_for_shape(state, _shape)
                _roi_keys = compute_roi_keys_by_shape(
                    _matched, _shape, state.library_roi_painted,
                    oxygenation_filter=_oxy,
                    alignment_affine=_alignment,
                )
                _result = compute_roi_average(_matched, _roi_keys)
                if _result is None:
                    ui.notify(
                        "ROI has fewer than 2 averageable subject "
                        "estimates.",
                        type="warning",
                    )
                    return
                _mean, _std, _n_subjects, _n_channels = _result
                _anchor = state.library_selected_hrf
                _sfreq = _resolve_roi_sfreq(_anchor, _matched, _roi_keys)

                from ..workspace_io import save_roi_average, workspace_dir
                try:
                    out_path = save_roi_average(
                        roi_keys=_roi_keys,
                        hrf_mean=_mean,
                        hrf_std=_std,
                        sfreq=_sfreq,
                        shape=_shape,
                        anchor=_anchor,
                        library_filter=state.library_filter,
                        oxygenation_filter=_oxy,
                    )
                    state.last_saved_roi_path = out_path
                    ui.notify(
                        f"Saved ROI average to {out_path.name} "
                        f"({workspace_dir()})",
                        type="positive",
                    )
                    # Re-render so the persistent "Last saved" label
                    # below picks up the new path immediately.
                    _body.refresh()
                except Exception as exc:  # noqa: BLE001
                    logger.exception("save ROI average failed: %s", exc)
                    ui.notify(
                        f"Save failed: {type(exc).__name__}: {exc}",
                        type="negative",
                    )

            save_btn = ui.button(
                "Save ROI average",
                icon="download",
                on_click=_on_save_roi,
            ).props("color=primary dense")
            if not can_save:
                save_btn.props("disable")

            # --- Persistent "last saved" feedback (PR #54) ----------
            # ``ui.notify`` toasts vanish in seconds; this label stays
            # visible until the next render replaces it, so users can
            # confirm the save happened even if they didn't catch the
            # toast.
            if state.last_saved_roi_path is not None:
                ui.label(
                    f"Last saved: {state.last_saved_roi_path.name}"
                ).classes("text-xs font-mono opacity-60 break-all")

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("hrtree_selection_changed", _refresh)
    state.subscribe("hrtree_filter_changed", _refresh)


def _format_shape_readout(state: AppState) -> str:
    """One-line summary of the current Cluster shape, suitable for copy-paste.

    Branches on ``state.cluster_shape`` so when the box / lasso UIs
    return in v1.4 / PR #54 this helper renders them too without
    touching the call sites. Today the box branch is unreachable
    from the GUI but stays here as the contract for the spatial-
    layer ``cluster_shape`` field.
    """
    cx, cy, cz = (
        state.cluster_center_x_mm,
        state.cluster_center_y_mm,
        state.cluster_center_z_mm,
    )
    centre = f"Centre: ({cx:.1f}, {cy:.1f}, {cz:.1f}) mm"
    if state.cluster_shape == SHAPE_BOX:
        hx = state.cluster_box_half_x_mm
        hy = state.cluster_box_half_y_mm
        hz = state.cluster_box_half_z_mm
        dims = f"Box {hx * 2:.1f}x{hy * 2:.1f}x{hz * 2:.1f} mm"
        return f"{centre}  ·  {dims}"
    if state.cluster_shape == SHAPE_ATLAS_REGION:
        region = state.cluster_atlas_label or "(no region selected)"
        return f"{centre}  ·  Atlas region: {region}"
    radius_mm = state.library_roi_radius_m * 1000.0
    return f"{centre}  ·  Sphere r={radius_mm:.1f} mm"


def _atlas_alignment_status(state: AppState) -> str:
    """Short label describing the current HRF->MNI alignment state."""
    import numpy as np
    has_affine = state.cluster_atlas_alignment_affine is not None
    has_offset = (
        state.cluster_atlas_offset_x_mm != 0.0
        or state.cluster_atlas_offset_y_mm != 0.0
        or state.cluster_atlas_offset_z_mm != 0.0
    )
    if has_affine and has_offset:
        return "Alignment: custom affine + offsets"
    if has_affine:
        affine = np.asarray(state.cluster_atlas_alignment_affine)
        if affine.shape == (4, 4) and np.allclose(affine, np.eye(4), atol=1e-9):
            return "Alignment: identity (no transform)"
        return "Alignment: custom 4x4 affine"
    if has_offset:
        ox = state.cluster_atlas_offset_x_mm
        oy = state.cluster_atlas_offset_y_mm
        oz = state.cluster_atlas_offset_z_mm
        return f"Alignment: offset ({ox:+.1f}, {oy:+.1f}, {oz:+.1f}) mm"
    return "Alignment: identity (no transform)"


def _looks_out_of_mni(hrfs: Dict[str, Dict[str, Any]]) -> bool:
    """Heuristic: sample a few HRF locations and check if they're out of MNI mm bounds.

    MNI Y axis runs roughly -100 to +80 mm. Bundled P-CAT HRFs are
    stored in MNE head coords with origin near the auditory meatus,
    so their Y values land around +60 to +110 mm -- the ``> 100`` mm
    test catches that case while letting properly-MNI HRFs pass.
    Used by the Cluster sub-tab to surface an alignment warning.
    """
    import numpy as np
    # Sample up to 8 HRFs; require >=3 to have Y > 100 mm to flag.
    over_threshold = 0
    sampled = 0
    for hrf in hrfs.values():
        loc = hrf.get("location")
        if loc is None or len(loc) < 3:
            continue
        sampled += 1
        try:
            y_mm = float(loc[1]) * 1000.0
        except (TypeError, ValueError):
            continue
        if abs(y_mm) > 100.0:
            over_threshold += 1
        if sampled >= 8:
            break
    return sampled >= 3 and over_threshold >= 3


def _render_atlas_alignment_section(state: AppState, body_refreshable) -> None:
    """Render the alignment controls in atlas mode.

    Three pieces:

    1. Out-of-MNI warning when HRF locations look like MNE head coords.
       Helps the user understand why atlas mode shows empty ROIs.
    2. Three offset number inputs (x / y / z mm) for users who want
       to dial in a rough translation by eye.
    3. An upload widget for a JSON 4x4 affine matrix. Cleared via a
       "reset" button.
    """
    import json as _json
    import numpy as np

    matched = filter_by_oxygenation(
        apply_filter(gather_library_hrfs(state), state.library_filter),
        state.library_oxygenation,
    )
    if _looks_out_of_mni(matched):
        with ui.row().classes("w-full items-start gap-2"):
            ui.icon("warning").classes("text-amber-500 text-sm")
            ui.label(
                "HRF locations appear to be in MNE head coords (not MNI). "
                "Atlas membership will be inaccurate until you load an "
                "alignment matrix or set offsets below."
            ).classes("text-xs opacity-80")

    ui.label("Atlas alignment (HRF coord -> MNI mm)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )

    # --- Offset inputs ---
    def _make_offset_input(axis: str, attr: str):
        def _on_change(event) -> None:
            try:
                value = float(event.value or 0.0)
            except (TypeError, ValueError):
                return
            setattr(state, attr, value)
            state.publish("hrtree_filter_changed", state.library_filter)
            body_refreshable.refresh()

        return ui.number(
            label=axis,
            value=getattr(state, attr),
            step=1.0,
            format="%.1f",
            on_change=_on_change,
        ).props("dense").classes("w-20")

    with ui.row().classes("w-full gap-2"):
        _make_offset_input("dx", "cluster_atlas_offset_x_mm")
        _make_offset_input("dy", "cluster_atlas_offset_y_mm")
        _make_offset_input("dz", "cluster_atlas_offset_z_mm")

    # --- Affine matrix upload ---
    def _on_upload(event) -> None:
        try:
            content = event.content.read()
            payload = _json.loads(content.decode("utf-8"))
            # Accept either {"affine_mm": [[...], ...]} or a bare
            # nested-list 4x4. Both make sense as user input.
            raw = (
                payload.get("affine_mm")
                if isinstance(payload, dict) and "affine_mm" in payload
                else payload
            )
            affine = np.asarray(raw, dtype=np.float64)
            if affine.shape != (4, 4):
                raise ValueError(
                    f"affine must be 4x4, got shape {affine.shape}"
                )
            state.cluster_atlas_alignment_affine = affine
            ui.notify(
                "Loaded HRF -> MNI alignment matrix.", type="positive"
            )
            state.publish("hrtree_filter_changed", state.library_filter)
            body_refreshable.refresh()
        except Exception as exc:  # noqa: BLE001
            ui.notify(
                f"Failed to load alignment: {type(exc).__name__}: {exc}",
                type="negative",
            )

    def _on_reset_alignment() -> None:
        state.cluster_atlas_alignment_affine = None
        state.cluster_atlas_offset_x_mm = 0.0
        state.cluster_atlas_offset_y_mm = 0.0
        state.cluster_atlas_offset_z_mm = 0.0
        state.publish("hrtree_filter_changed", state.library_filter)
        body_refreshable.refresh()

    with ui.row().classes("w-full gap-2 items-center"):
        ui.upload(
            label="Load alignment .json",
            on_upload=_on_upload,
            auto_upload=True,
            max_files=1,
        ).props("flat dense accept=.json").classes("w-48")
        ui.button(
            "Reset alignment",
            on_click=_on_reset_alignment,
        ).props("flat dense")

    ui.label(_atlas_alignment_status(state)).classes(
        "text-xs font-mono opacity-70"
    )


# ---------------------------------------------------------------------------
# Center pane: HRtree 3D viz
# ---------------------------------------------------------------------------


def _render_viz_pane(state: AppState) -> None:
    """The plotly 3D scatter of HRF locations.

    Refreshable so the Apply button can re-render against the filter.

    The refreshable's container (a NiceGUI ``RefreshableContainer``
    custom element) defaults to inline-ish display, which breaks the
    ``h-full`` chain — the plotly viz would otherwise shrink to its
    content's intrinsic height and leave the bottom half of the
    splitter pane empty. We apply ``w-full h-full`` to the container
    after the first call so the body fills the available height.
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
                        "Library trees not loaded. Re-opening the HRtree tab "
                        "may help."
                    ).classes("text-sm opacity-60")
            return

        # Use a flex column so the plotly viz can claim flex-1 and the
        # overlay-toggles row sits underneath at content-height.
        with ui.column().classes("w-full h-full p-3 gap-2 flex flex-col"):
            # Compute ROI keys now so the figure draws the highlight
            # halo and the label can report the count.
            shape = _build_current_shape(state)
            oxy_filter = _resolve_cluster_oxygenation(state)
            alignment = _alignment_for_shape(state, shape)
            roi_keys = compute_roi_keys_by_shape(
                matched,
                shape,
                state.library_roi_painted,
                oxygenation_filter=oxy_filter,
                alignment_affine=alignment,
            )
            roi_status = ""
            if roi_keys:
                roi_status = f"  •  ROI: {len(roi_keys)} highlighted"
            ui.label(
                f"{len(matched)} HRFs shown{roi_status}"
            ).classes("text-sm opacity-70")
            fig = build_plotly_figure(
                matched,
                show_brain=state.library_show_brain,
                show_scalp=state.library_show_scalp,
                roi_keys=roi_keys,
                roi_shape=shape,
            )
            plot = ui.plotly(fig).classes("w-full flex-1 min-h-0")

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
                # A plain click also resets the painted set -- the user is
                # picking a fresh anchor, so accumulated shift-hover paint
                # from a prior anchor shouldn't carry over.
                state.library_selected_hrf = {**hrf, "_key": hrf_key}
                # Seed the Cluster sub-tab's shape centre from the clicked
                # HRF so sphere mode's behaviour matches the v1.2 "click
                # an HRF, sphere centres on it" workflow even though the
                # spatial layer now drives the centre from state. Box mode
                # uses the same centre so a click also re-centres the box.
                # HRF locations are stored in meters; spatial layer is mm.
                loc = hrf.get("location") or [0, 0, 0]
                if len(loc) >= 3:
                    state.cluster_center_x_mm = float(loc[0]) * 1000.0
                    state.cluster_center_y_mm = float(loc[1]) * 1000.0
                    state.cluster_center_z_mm = float(loc[2]) * 1000.0
                state.library_roi_painted.clear()
                state.publish("hrtree_selection_changed", hrf_key)

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
                state.publish("hrtree_selection_changed", key)

            plot.on("plotly_click", _on_click)
            plot.on("roi_paint", _on_paint)
            # Wire the JS shift-tracker + plotly_hover hook AFTER the
            # plotly element has rendered. Slight delay so the
            # underlying div is queryable in the DOM. once=True so
            # the hook isn't registered repeatedly on each refresh.
            _install_paint_hook(plot.id)

            # MNI overlay toggles under the viz — they control what's
            # rendered above (brain mesh, scalp mesh), so visual
            # adjacency makes them easier to discover than the legacy
            # left-sidebar placement. Independent switches so users can
            # show either, both, or neither.
            def _publish_filter_change() -> None:
                state.publish("hrtree_filter_changed", state.library_filter)

            def _on_brain_toggle(event) -> None:
                state.library_show_brain = bool(event.value)
                _publish_filter_change()

            def _on_scalp_toggle(event) -> None:
                state.library_show_scalp = bool(event.value)
                _publish_filter_change()

            with ui.row().classes(
                "w-full items-center justify-center gap-6 shrink-0 pt-1"
            ):
                ui.switch(
                    "Show MNI brain",
                    value=state.library_show_brain,
                    on_change=_on_brain_toggle,
                ).props("dense").tooltip(
                    "Translucent fsaverage pial cortical surface beneath "
                    "the HRF scatter — where the neural activity originates."
                )
                ui.switch(
                    "Show MNI head",
                    value=state.library_show_scalp,
                    on_change=_on_scalp_toggle,
                ).props("dense").tooltip(
                    "Translucent fsaverage scalp (outer-skin) surface — "
                    "where forehead/head-mounted fNIRS optodes physically sit."
                )

    _viz_body()
    # Note: NiceGUI's ``RefreshableContainer`` template is just a
    # ``<slot>``, which Vue renders as a fragment with no root DOM
    # element. That means classes / styles applied to the container
    # have nowhere to land (Vue prints a "non-prop attribute could not
    # be inherited" warning). The refreshable's children are direct
    # layout children of the splitter slot already, so no wrapper-
    # styling is needed — height propagates through the slot directly.

    # Re-render the viz on both filter and selection change.
    #
    # Selection changes (clicking an HRF, shift-paint adding to the
    # painted set) update both ``state.library_selected_hrf`` AND
    # ``state.cluster_center_*_mm`` (the click handler seeds the
    # cluster centre from the clicked HRF's location). Without the
    # selection_changed subscription, the figure's shape overlay
    # would stay at the old centre until the user happened to toggle
    # something on the Filter sub-tab or the MNI overlay switches --
    # confusing because the detail pane + Cluster sub-tab DO update
    # immediately (both subscribe to selection_changed), so the user
    # sees the readout update while the 3D shape stays put.
    def _refresh_viz(_payload=None) -> None:
        _viz_body.refresh()
    state.subscribe("hrtree_filter_changed", _refresh_viz)
    state.subscribe("hrtree_selection_changed", _refresh_viz)


def _extract_clicked_hrf_key(event) -> Optional[str]:
    """Pull the clicked HRF's key from a plotly click event payload."""
    try:
        args = getattr(event, "args", None) or {}
        points = args.get("points") or []
        if not points:
            return None
        first = points[0]
        return first.get("customdata")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "hrtree: failed to extract HRF key from click event: %s", exc
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
    # (See _render_viz_pane note on refreshable wrappers — no styling
    # needed; RefreshableContainer renders as a Vue fragment with no
    # root DOM element.)

    def _refresh_detail(_payload=None) -> None:
        _detail_body.refresh()
    state.subscribe("hrtree_selection_changed", _refresh_detail)
    # The radius slider publishes ``hrtree_filter_changed`` because the
    # viz already listens there; the detail pane needs to refresh too
    # so the ROI-average plot updates when the user widens the radius.
    state.subscribe("hrtree_filter_changed", _refresh_detail)


def _render_roi_average(state: AppState) -> None:
    """Render the averaged-trace plot for the current ROI.

    Read-only display: shows the averaged trace PNG with the ROI
    member count. The Save-to-workspace action lives in the Cluster
    sub-tab on the left (Phase 6); the detail pane is for viewing,
    not for committing state to disk.

    Uses the Cluster sub-tab's current Shape (box or sphere) plus
    the visible-filter context to compute ROI membership, so this
    panel stays in sync with the cluster shape even when the user
    has no anchor HRF clicked.
    """
    all_hrfs = gather_library_hrfs(state)
    matched = filter_by_oxygenation(
        apply_filter(all_hrfs, state.library_filter),
        state.library_oxygenation,
    )
    shape = _build_current_shape(state)
    oxy_filter = _resolve_cluster_oxygenation(state)
    alignment = _alignment_for_shape(state, shape)
    roi_keys = compute_roi_keys_by_shape(
        matched, shape, state.library_roi_painted,
        oxygenation_filter=oxy_filter,
        alignment_affine=alignment,
    )
    result = compute_roi_average(matched, roi_keys)
    if result is None:
        return
    mean, std, n_subjects, n_channels = result
    # Sfreq: prefer the anchor's when present (legacy behaviour);
    # otherwise default to the first ROI member's sfreq, falling back
    # to 1.0 if even that's missing. The save flow does the same.
    anchor = state.library_selected_hrf
    sfreq = _resolve_roi_sfreq(anchor, matched, roi_keys)
    ui.separator()
    ui.label(
        f"ROI average ({n_subjects} subjects, {n_channels} channels)"
    ).classes("text-xs uppercase opacity-60 tracking-wide")
    png = _render_roi_average_png(mean, std, sfreq, n_subjects)
    if png is not None:
        ui.image(png).classes("max-w-md")


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


def _resolve_roi_sfreq(
    anchor: Optional[Dict[str, Any]],
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
) -> float:
    """Pick the sample rate for an ROI-averaged trace.

    Prefers the click-anchor's ``sfreq`` when there is one (matches the
    v1.2 behaviour). For free-floating ROIs with no anchor, falls back
    to the first ROI member's ``sfreq``. Final fallback is 1.0 Hz so
    the time axis on the plot has *some* scale even when the HRFs are
    missing rate metadata.
    """
    if anchor is not None:
        anchor_sfreq = float(anchor.get("sfreq") or 0.0)
        if anchor_sfreq > 0:
            return anchor_sfreq
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        member_sfreq = float(hrf.get("sfreq") or 0.0)
        if member_sfreq > 0:
            return member_sfreq
    return 1.0


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
    roi_shape: Optional[Shape] = None,
):
    """Build the 3D scatter figure for the given HRF dict.

    Up to six traces, ordered so the ROI highlight renders on top:

    - **Scalp** (``go.Mesh3d``, only when ``show_scalp=True``):
      fsaverage outer-skin surface -- anatomically where the optodes
      sit. Drawn first (outermost in 3D-painter order).
    - **Brain** (``go.Mesh3d``, only when ``show_brain=True``):
      fsaverage pial cortical surface -- where the neural activity
      originates. Drawn inside the scalp.
    - **ROI shape overlay** (``go.Mesh3d``, only when ``roi_shape``
      is a :class:`~hrfunc.spatial.shapes.Box` or
      :class:`~hrfunc.spatial.shapes.Sphere`): a translucent violet
      cuboid / UV-sphere showing where the Cluster sub-tab's ROI
      selector currently sits. The shape's centre/extent state is
      converted from MNI mm to MNE-meter coordinates so it renders
      in the same coordinate frame as the HRF scatter.
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
    # order — when both are on, the brain visually nests inside the head.
    if show_scalp:
        mesh = load_mesh("scalp")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                make_surface_trace(
                    verts, faces,
                    color="#c4b5a0",  # warm skin tone
                    opacity=0.12,
                    name="MNI head",
                    ambient=0.6, diffuse=0.5,
                )
            )
    if show_brain:
        mesh = load_mesh("pial")
        if mesh is not None:
            verts, faces = mesh
            traces.append(
                make_surface_trace(
                    verts, faces,
                    color="#9ca3af",  # cool grey for cortex
                    opacity=0.30,
                    name="MNI brain",
                    ambient=0.5, diffuse=0.6,
                )
            )

    # ROI shape overlay (PR #49 box/sphere). HRF coords are in meters;
    # the spatial-layer shape is in mm, so we down-convert before
    # building the trace -- the resulting Mesh3d is in meters and
    # overlays directly on the HRF scatter.
    if roi_shape is not None:
        if isinstance(roi_shape, Box):
            box_m = Box(
                center_mm=(c / 1000.0 for c in roi_shape.center_mm),
                half_extents_mm=(h / 1000.0 for h in roi_shape.half_extents_mm),
            )
            traces.append(make_box_overlay_trace(box_m))
        elif isinstance(roi_shape, Sphere):
            sphere_m = Sphere(
                center_mm=(c / 1000.0 for c in roi_shape.center_mm),
                radius_mm=roi_shape.radius_mm / 1000.0,
            )
            traces.append(make_sphere_overlay_trace(sphere_m))

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
      oxygenation when an anchor is set, so a stray paint on the wrong
      haemoglobin doesn't contaminate the average).

    The anchor's own key is always part of the ROI when it's still in
    ``hrfs`` (otherwise filtering away the anchor's neighbourhood would
    be confusing).

    The radius check is delegated to :class:`hrfunc.spatial.shapes.Sphere`.
    HRF locations are stored in meters internally; the spatial layer
    works in mm per the v1.3 compartmentalization convention, so this
    function converts both the anchor and each candidate location to mm
    before asking the sphere. The result is bit-identical to the
    pre-refactor Euclidean check (``loc * 1000`` is exact for the
    head-scale magnitudes here in float64).

    Module-level so tests can hit it without spinning up the GUI.
    """
    out: set = set()
    if anchor is None and not painted:
        return out

    anchor_loc = None
    anchor_oxy = None
    anchor_key = None
    sphere: Optional[Sphere] = None
    if anchor is not None:
        anchor_loc = anchor.get("location")
        anchor_oxy = anchor.get("oxygenation")
        anchor_key = anchor.get("_key")
        if anchor_key is not None and anchor_key in hrfs:
            out.add(anchor_key)
        if anchor_loc is not None and len(anchor_loc) >= 3 and radius_m > 0:
            sphere = Sphere(
                center_mm=meters_to_mm(anchor_loc[:3]).tolist(),
                radius_mm=float(meters_to_mm(radius_m)),
            )

    if sphere is not None:
        for key, hrf in hrfs.items():
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            if anchor_oxy is not None and hrf.get("oxygenation") != anchor_oxy:
                continue
            loc_mm = meters_to_mm(loc[:3]).tolist()
            if sphere.contains(loc_mm):
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


def compute_roi_keys_by_shape(
    hrfs: Dict[str, Dict[str, Any]],
    shape: Optional[Any],
    painted: Optional[Any] = None,
    *,
    oxygenation_filter: Optional[bool] = None,
    alignment_affine: "Optional[np.ndarray]" = None,
) -> "set":
    """Shape-based ROI membership for free-floating Box / Sphere modes.

    Companion to :func:`compute_roi_keys`. The original anchor-based
    API is preserved for the legacy click-anchor + radius workflow;
    this function takes a fully-constructed :class:`hrfunc.spatial.Shape`
    (typically a :class:`Box` or a free-floating :class:`Sphere`) plus
    an explicit oxygenation filter and returns the matching keys.

    Membership rules:

    - If ``shape`` is not None, every HRF whose location (converted
      meters->mm, then through the optional ``alignment_affine``)
      is inside ``shape`` is included. HRFs without a location are
      skipped.
    - Every key in ``painted`` is included (filtered by
      ``oxygenation_filter`` when set, so a stray paint on the
      wrong haemoglobin doesn't contaminate the average).
    - When ``oxygenation_filter`` is ``True`` / ``False``, only HRFs
      with matching ``oxygenation`` survive the membership check.
      ``None`` (the default) skips the oxygenation filter -- callers
      that have already filtered upstream (e.g. via
      ``library_oxygenation``) should leave this as None.

    ``alignment_affine`` (PR #54): a 4x4 homogeneous transform applied
    to the HRF coordinate before the shape predicate. Used for atlas
    mode to map MNE-head-coord HRFs into the atlas's MNI-mm frame.
    Pass ``None`` to skip the transform (default; sphere / box modes
    don't need it because the shape itself is in MNE-head space).

    Module-level so tests can hit it without spinning up the GUI.
    """
    import numpy as np

    out: "set" = set()
    if shape is None and not painted:
        return out

    if shape is not None:
        for key, hrf in hrfs.items():
            loc = hrf.get("location")
            if loc is None or len(loc) < 3:
                continue
            if (
                oxygenation_filter is not None
                and bool(hrf.get("oxygenation")) != bool(oxygenation_filter)
            ):
                continue
            loc_mm = meters_to_mm(loc[:3]).tolist()
            if alignment_affine is not None:
                homo = np.array(
                    [loc_mm[0], loc_mm[1], loc_mm[2], 1.0],
                    dtype=np.float64,
                )
                aligned = alignment_affine @ homo
                loc_mm = [
                    float(aligned[0]),
                    float(aligned[1]),
                    float(aligned[2]),
                ]
            if shape.contains(loc_mm):
                out.add(key)

    if painted:
        for key in painted:
            if key not in hrfs:
                continue
            if (
                oxygenation_filter is not None
                and bool(hrfs[key].get("oxygenation")) != bool(oxygenation_filter)
            ):
                continue
            out.add(key)

    return out


def compute_roi_average(
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
):
    """Average the per-subject ``estimates`` of every HRF in the ROI.

    Returns ``(mean, std, n_subjects, n_channels)`` -- the grand mean
    and std across all subject-level estimates pooled from every HRF
    in the ROI, plus the number of subject traces that contributed
    and the number of source channels they came from. Returns ``None``
    if fewer than 2 subject traces are averageable.

    **PR #54 correctness fix:** previously averaged ``hrf_mean`` (the
    per-channel mean), so a 50-subject channel got the same weight as
    a 5-subject channel in the final grand mean. Pooling ``estimates``
    instead gives every subject equal weight, which is what
    researchers report in publications.

    **HRFs without populated ``estimates``** are excluded from the
    average -- :func:`compute_roi_excluded_count` surfaces the count
    so the GUI can warn the user. The bundled library is loaded with
    ``rich=True`` so estimates survive the JSON load; HRFs missing
    estimates are typically those that came from a study where only
    the channel mean was published.

    Skips traces with empty / mismatched length. The modal length
    across the candidate pool is the canonical length; outliers are
    dropped (e.g. a single channel published at a different duration
    doesn't contaminate the average).

    Module-level so tests can call without a UI.
    """
    import numpy as np
    from collections import Counter

    # First pass: parse every subject-level estimate into a numpy array.
    # Track the source-channel count separately so the UI can show
    # "averaged N subjects across M channels".
    candidates: List["np.ndarray"] = []
    contributing_channels = 0
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        estimates = hrf.get("estimates") or []
        if not estimates:
            # No subject-level estimates -> can't contribute to a
            # subject-weighted grand mean. Skip the channel entirely
            # rather than fall back to hrf_mean (would mix two
            # averaging conventions in the same output).
            continue
        added_from_this_channel = False
        for estimate in estimates:
            try:
                arr = np.asarray(estimate, dtype=float)
            except Exception:  # noqa: BLE001
                continue
            if arr.ndim != 1 or arr.size == 0:
                continue
            candidates.append(arr)
            added_from_this_channel = True
        if added_from_this_channel:
            contributing_channels += 1

    if len(candidates) < 2:
        return None

    # Pick the MODAL length as the canonical one rather than the first
    # iterated one. ``roi_keys`` is typically a set (no order guarantees),
    # and taking the first-seen length means an outlier length could
    # throw out the majority. Modal length is robust to iteration order
    # and matches what a researcher would expect: "average the traces
    # that share the typical duration; skip the oddball".
    lengths = Counter(arr.shape[0] for arr in candidates)
    canonical_len = lengths.most_common(1)[0][0]
    traces = [arr for arr in candidates if arr.shape[0] == canonical_len]

    if len(traces) < 2:
        return None

    stacked = np.vstack(traces)
    return (
        stacked.mean(axis=0),
        stacked.std(axis=0, ddof=0),
        len(traces),
        contributing_channels,
    )


def compute_roi_excluded_count(
    hrfs: Dict[str, Dict[str, Any]],
    roi_keys: Any,
) -> int:
    """Count the ROI HRFs that have no usable per-subject estimates.

    Used by the GUI to warn researchers when their ROI mixes channels
    with and without published subject-level data -- the average will
    silently drop the un-publishable channels.
    """
    excluded = 0
    for key in roi_keys:
        hrf = hrfs.get(key)
        if hrf is None:
            continue
        estimates = hrf.get("estimates") or []
        if not estimates:
            excluded += 1
    return excluded


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
        logger.warning("matplotlib unavailable for hrtree trace: %s", exc)
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
        logger.warning("hrtree trace render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
