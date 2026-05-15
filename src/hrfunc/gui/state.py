"""AppState — single source of truth for the running GUI.

NiceGUI page handlers, components, and background workers all read and write
the same `AppState` instance. A module-level `state` singleton is created at
import time so any module can ``from hrfunc.gui.state import state`` without
threading a reference through every function signature.

Lifecycle:
- One AppState per process. The `state` singleton is created on first import.
- Tests can instantiate fresh AppState() instances for isolated unit testing
  (the class itself is just a dataclass).
- The singleton holds mutable fields by design — pages bind their UI elements
  to these fields and re-render on changes.

What lives here:
- `manifest`             - last folder scan result (None until a scan completes)
- `selected_scan`        - currently inspected ScanEntry, or None
- `raw_cache`            - hot-path LRU(3) loader of source MNE Raw objects
- `processed_cache`      - LRU(3) of *preprocessed* Raw objects (Sprint 3.2);
                           HRFs / Activity tabs read from here
- `preload_path`         - CLI arg from `hrfunc <path>`; consumed by welcome
                           page on first render
- `busy`                 - True while a background task is running (drives
                           spinner UI); gate for estimation, NOT for scan loads
- `estimation_progress`  - (current, total, channel_name) tuple from the latest
                           progress_callback fire; None when no estimation in flight
- `last_error`           - last error message surfaced to the user, or None
- `subscribers`          - event-bus dispatch table (Sprint 3.2); see
                           ``subscribe`` / ``publish``
- `montage`              - most recently estimated Montage from the HRFs tab
                           (Sprint 3.3); None until estimate_hrf runs at least
                           once. Cleared on dataset reset; switching the
                           selected scan does NOT clear it, so users can
                           switch tabs and come back without losing results
                           — but a new estimation overwrites the field
                           regardless of which scan it came from.
- `activity_raw`         - most recent deconvolved Raw from the Activity tab
                           (Sprint 3.4); the output of ``estimate_activity``
                           which mutates a copy of the preprocessed Raw and
                           returns it with neural-activity values in place
                           of haemoglobin values. None until run at least
                           once; cleared on reset.

Event bus (Sprint 3.2, extended in 3.3):
The bus replaces the Sprint 2.3-era ``_inspect_refresh`` private attribute.
Panels subscribe to named events and are called when other parts of the GUI
publish. The bus is dict-of-lists, deliberately minimal — no priorities, no
async dispatch, no payload schemas. Defined events:

- ``"scan_selected"``  — payload: ``ScanEntry`` (or None for deselection).
  Published when the dataset tree updates ``state.selected_scan``.
- ``"scan_loaded"``    — payload: ``ScanEntry``. Published after a background
  Raw load completes successfully; subscribers can read the Raw from
  ``state.raw_cache``.
- ``"preprocess_done"`` — payload: ``ScanEntry``. Published after a successful
  preprocess run; subscribers can read the processed Raw from
  ``state.processed_cache``.
- ``"hrf_estimated"``   — payload: ``ScanEntry``. Published after a successful
  ``estimate_hrf`` (or canonical HRF generation); subscribers can read the
  resulting Montage from ``state.montage``.
- ``"activity_estimated"`` — payload: ``ScanEntry``. Published after a
  successful ``estimate_activity`` run; subscribers can read the deconvolved
  Raw from ``state.activity_raw``.
- ``"quality_computed"`` — payload: ``ScanEntry`` or ``None``. Published
  after a Quality-panel metrics computation finishes (per-scan: ScanEntry;
  dataset-wide aggregate: None). Subscribers can read
  ``state.quality_metrics`` for the results.
- ``"project_changed"`` — payload: ``Manifest`` or ``None``. Published by
  ``set_manifest`` when the active project swaps (load new, switch, or
  close). Panels with persistent refreshables subscribe to blank or
  rebuild their views before reading the new manifest.
- ``"busy_changed"`` — payload: ``bool`` (the new busy value). Published
  by ``set_busy`` when a background worker starts (True) or completes
  (False). The project picker subscribes to disable Open / Close while
  busy so a switch can't strand a half-finished run on the new project.

Subscribers are sync callables. Async handlers can dispatch via
``nicegui.background_tasks.create`` from inside their callback.

Fields are added (not removed) as later sprints integrate more state. Keeping
the AppState surface stable across sprints means GUI components written in
earlier sprints don't need updates as later panels land.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..io.manifest import Manifest, ScanEntry
from ..io.raw_cache import RawCache

logger = logging.getLogger(__name__)

EventCallback = Callable[..., None]


@dataclass
class ROISlot:
    """One ROI in the Cluster sub-tab's multi-ROI list (PR #55).

    Holds everything that distinguishes one ROI from another: its shape
    selection, geometry parameters, painted-key set, and the click-
    anchor that seeded it (if any). Atlas alignment is NOT stored
    here -- alignment is a property of the HRF library's coord frame,
    not of any individual ROI, so it lives on AppState as a global
    per scan/dataset (locked decision 2026-05-14).

    The default-constructed slot reproduces the pre-PR-#55 single-ROI
    starting state: sphere mode, free-floating centred at the MNI
    origin with a 20 mm radius. The first slot in ``cluster_rois``
    is therefore safe to leave at defaults if the user never opens
    the Cluster sub-tab.
    """

    # Display name for the ROI list ("ROI 1", "ROI 2", ...). Auto-
    # assigned by ``AppState.add_roi`` but mutable so future iterations
    # can rename. Not included in the saved montage's per-ROI block
    # unless renamed (see ``workspace_io``).
    name: str = "ROI 1"

    # Shape mode: "sphere" | "box" | "atlas_region". Matches the
    # module-level SHAPE_* constants in hrtree_panel; kept as a string
    # here so this module doesn't have to import the panel.
    shape: str = "sphere"

    # Free-floating shape centre, MNI mm. Three separate fields so each
    # binds cleanly to its own ``ui.number`` input. Defaults to MNI
    # origin (0, 0, 0); seeded by clicking an HRF in the viz pane.
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    center_z_mm: float = 0.0

    # Box half-extents, MNI mm. Default 20 mm on each axis = a 40 mm
    # cube, comparable to a 2 cm radius sphere by volume.
    box_half_x_mm: float = 20.0
    box_half_y_mm: float = 20.0
    box_half_z_mm: float = 20.0

    # Sphere radius, MNI mm. Pre-PR-#55 this lived on AppState as
    # ``library_roi_radius_m`` (meters); the per-ROI move converts to
    # mm to match the rest of the spatial-layer convention (MNI mm
    # everywhere from PR #46). Default 20 mm = the legacy 0.02 m.
    radius_mm: float = 20.0

    # Atlas-region label when ``shape == "atlas_region"``. ``None``
    # means "no region picked yet" and the save button disables.
    atlas_label: Optional[str] = None

    # Shift-hover painted keys (the lasso-like accumulation), filtered
    # to the anchor's oxygenation when an anchor is set. Joins the
    # ROI regardless of the shape geometry. Cleared on every new
    # anchor click so paint from a prior anchor doesn't carry over.
    painted: Set[str] = field(default_factory=set)

    # Click-anchor HRF, if any. Same dict shape as
    # ``state.library_selected_hrf`` (gathered HRF + ``_key``).
    # When present, drives the saved JSON's location + oxygenation
    # fields and the sphere's centre seed.
    anchor: Optional[Dict[str, Any]] = None


def _default_rois() -> List[ROISlot]:
    """Factory for AppState.cluster_rois.

    Always seeds with one default ROISlot so the active-index has
    something to point at and the proxy properties never look at an
    empty list. CLEAR ROI deletes the active slot but refuses to drop
    below one entry (it resets the last slot instead).
    """
    return [ROISlot()]


@dataclass
class AppState:
    """Mutable, single-process GUI state.

    All fields default to `None` / empty so a freshly-constructed AppState
    represents "no data loaded yet" — the state shown by the welcome page.
    """

    manifest: Optional[Manifest] = None
    selected_scan: Optional[ScanEntry] = None
    raw_cache: RawCache = field(default_factory=RawCache)
    processed_cache: RawCache = field(default_factory=RawCache)
    preload_path: Optional[Path] = None
    busy: bool = False
    estimation_progress: Optional[Tuple[int, int, str]] = None
    last_error: Optional[str] = None
    subscribers: Dict[str, List[EventCallback]] = field(default_factory=dict)
    # Montage from the most recent HRF estimation (Sprint 3.3). Typed as Any to
    # avoid pulling hrfunc.hrfunc into the GUI import graph at module load —
    # the GUI must stay importable without MNE for tests that disable it.
    montage: Optional[Any] = None
    # Scan that produced the current ``montage`` (Sprint 3.4). The Activity tab
    # uses this to refuse toeplitz-mode estimation when the user has switched
    # to a different scan since estimate_hrf ran — applying scan A's HRFs to
    # scan B's Raw would silently produce wrong results because the library
    # matches by channel name, not by scan identity.
    montage_source_scan: Optional[ScanEntry] = None
    # Deconvolved Raw from the most recent estimate_activity call (Sprint 3.4).
    # Typed Any for the same import-graph reason. The Activity panel reads
    # the data + annotations for the lens-style preproc/deconv overlay plot.
    activity_raw: Optional[Any] = None
    # Per-scan quality metrics (Sprint 4.1). Keyed by ``ScanEntry.path.resolve()``;
    # each value is a dict {"raw": metrics_dict, "preprocessed": metrics_dict,
    # "deconvolved": metrics_dict}. Each metrics_dict contains numeric summaries
    # (snr_mean, skew_mean, kurtosis_mean, sci_mean when applicable). Entries
    # appear as the Quality panel computes them — either per-scan when the user
    # views Quality for the current scan, or in bulk during the dataset-wide
    # aggregate run.
    quality_metrics: Dict[Path, Dict[str, Any]] = field(default_factory=dict)
    # Lazy-loaded bundled HRF databases for the /library page (Sprint 4.2-4.4).
    # Tree objects from ``hrfunc.hrtree.tree``. Populated on first /library
    # visit; never cleared (the data is read-only from disk so re-loading
    # would just re-read the same files). Typed Any to keep the GUI import
    # graph free of hrfunc.hrtree at module load.
    library_hbo: Optional[Any] = None
    library_hbr: Optional[Any] = None
    # Current context filter state for the Library page. Keys are context
    # field names ('task', 'doi', 'demographics', ...). Empty dict = no
    # filter applied. The Library page rebuilds the visible HRF list from
    # the filter every render.
    library_filter: Dict[str, Any] = field(default_factory=dict)
    # Currently-selected HRF on the Library page (from a click in the
    # plotly viz or a manual list selection). Stored as the gathered-form
    # dict (the value type produced by ``tree.gather``), or None.
    library_selected_hrf: Optional[Dict[str, Any]] = None
    # Channel name selected for the HRF gallery's detail-pane focus
    # (Sprint 5.1). Sprint 3.3 rendered all channel HRFs overlaid on one
    # plot; Sprint 5 replaces that with a clickable grid + a per-channel
    # full-detail view. None = no channel focused yet (grid renders, no
    # detail view).
    hrf_selected_channel: Optional[str] = None
    # Toggles for the MNI fsaverage overlay surfaces on the /library
    # plotly viz. The "brain" toggle controls the pial cortical
    # surface (where the neural activity originates); the "scalp"
    # toggle controls the outer-skin head surface (where the optodes
    # physically sit). Both are independent so users can show either,
    # both, or neither. Scalp defaults ON because most fNIRS optodes
    # are forehead/scalp-mounted and the head shape gives the most
    # immediate spatial context; brain defaults ON too because the
    # cortex-relative position is the scientifically meaningful one.
    library_show_brain: bool = True
    library_show_scalp: bool = True
    # Oxygenation filter for the /library viz. ``"both"`` (default)
    # shows HbO + HbR; ``"hbo"`` / ``"hbr"`` hide the other channel.
    # Researchers often want to inspect one haemoglobin at a time;
    # the toggle lets them do that without writing a context filter.
    library_oxygenation: str = "both"
    # ROI selection on the /library viz. Two ways to add HRFs to the
    # ROI, used in combination:
    #
    # 1. **Anchor + radius**: click an HRF and every same-oxygenation
    #    HRF inside the sphere of radius ``library_roi_radius_m``
    #    around it gets included. The clicked HRF is the same dict
    #    as ``library_selected_hrf`` (anchor + detail-pane focus
    #    share state).
    # 2. **Shift-hover painting**: hold Shift and hover over HRFs.
    #    Each hovered key is added to ``library_roi_painted`` and
    #    joins the ROI regardless of radius. Lets researchers trace
    #    a non-spherical region by mouse, like a lasso.
    #
    # The full ROI is the union of (in-radius set) and the painted
    # set, filtered to the anchor's oxygenation. The averaged trace
    # in the detail pane is computed from this union.
    #
    # PR #55: per-ROI cluster state moved into ``cluster_rois``. The
    # ``library_roi_radius_m`` / ``library_roi_painted`` /
    # ``cluster_shape`` / ``cluster_center_*_mm`` / ``cluster_box_half_*_mm``
    # / ``cluster_atlas_label`` names are now ``@property`` proxies
    # onto ``cluster_rois[cluster_active_index]`` so existing panel
    # bindings and tests keep working. See ``ROISlot``.
    #
    # Multi-ROI list (PR #55). Always at least one entry -- CLEAR ROI
    # on the last slot resets it instead of removing it, so the
    # proxy properties never index an empty list. The active index
    # picks which slot the proxies read/write. UI exposes ADD ROI to
    # append, click-to-switch on the list, and CLEAR ROI on the
    # current active slot.
    cluster_rois: List[ROISlot] = field(default_factory=_default_rois)
    cluster_active_index: int = 0
    # PR #54: cluster ROI active toggle (default off). When False, the
    # cluster shape doesn't contribute to ROI membership at all -- the
    # viz pane shows raw HRFs with no halo, the detail-pane ROI-average
    # section stays hidden, and the save button is disabled. Researchers
    # opt into ROI mode explicitly via the toggle at the top of the
    # Cluster sub-tab. Pre-PR-#54 the ROI was always on which made the
    # gold halo around the default-centre (0, 0, 0) appear as if the
    # tool had pre-selected something.
    cluster_roi_active: bool = False
    # PR #54: HRF-coords-to-MNI alignment for atlas membership. Bundled
    # library HRFs are stored in MNE head coordinates (origin near the
    # auditory meatus); the Harvard-Oxford atlas is in MNI mm (origin
    # at the brain centroid). Without alignment, every HRF maps to
    # voxels outside the atlas volume -> atlas mode silently shows
    # empty ROIs. ``cluster_atlas_alignment_affine`` is a 4x4
    # homogeneous transform applied at the membership-check boundary
    # (HRF coord -> MNI coord). Defaults to None (= no transform) so
    # users with already-MNI HRFs aren't affected. Loadable from a
    # JSON or .npy file via the alignment file picker in atlas mode.
    cluster_atlas_alignment_affine: Optional[Any] = None
    # PR #54: human-friendly atlas alignment offsets (mm). These are a
    # shorthand for users without a full 4x4 affine -- they compose
    # with ``cluster_atlas_alignment_affine`` to produce the full
    # transform applied at lookup. Pure-translation corrections in
    # MNE-head -> MNI mm space.
    cluster_atlas_offset_x_mm: float = 0.0
    cluster_atlas_offset_y_mm: float = 0.0
    cluster_atlas_offset_z_mm: float = 0.0
    # PR #54: persistent "saved to" feedback for the Cluster sub-tab's
    # save action. ``ui.notify`` toasts vanish in seconds; storing the
    # last-saved path lets the sub-tab render an always-visible label
    # below the save button so users can confirm the file went out
    # even after navigating away and back.
    last_saved_roi_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # PR #55: proxy properties onto the active ROI slot.
    # ------------------------------------------------------------------
    # These exist so the Cluster sub-tab UI (which binds widgets to
    # ``state.cluster_shape``, ``state.cluster_center_x_mm``, etc.) and
    # the existing test suite see the per-ROI fields under their pre-
    # PR-#55 names. The actual storage lives in
    # ``cluster_rois[cluster_active_index]``.
    #
    # Switching the active ROI changes which slot the proxies read /
    # write -- callers re-render after touching ``cluster_active_index``
    # to pick up the new values. The proxies fall back to the first
    # slot when the index is out of range; ``cluster_rois`` is also
    # never empty (CLEAR ROI resets the last slot rather than removing
    # it), so the fallback is defensive against bad external state, not
    # an expected code path.

    @property
    def active_roi(self) -> ROISlot:
        """The currently-active ROI slot. Always returns a slot --
        if the list is empty (shouldn't happen under normal flow) one
        is created. If the index is out of range it clamps to 0."""
        if not self.cluster_rois:
            self.cluster_rois.append(ROISlot())
            self.cluster_active_index = 0
            return self.cluster_rois[0]
        if not (0 <= self.cluster_active_index < len(self.cluster_rois)):
            self.cluster_active_index = 0
        return self.cluster_rois[self.cluster_active_index]

    @property
    def cluster_shape(self) -> str:
        return self.active_roi.shape

    @cluster_shape.setter
    def cluster_shape(self, value: str) -> None:
        self.active_roi.shape = value

    @property
    def cluster_center_x_mm(self) -> float:
        return self.active_roi.center_x_mm

    @cluster_center_x_mm.setter
    def cluster_center_x_mm(self, value: float) -> None:
        self.active_roi.center_x_mm = float(value)

    @property
    def cluster_center_y_mm(self) -> float:
        return self.active_roi.center_y_mm

    @cluster_center_y_mm.setter
    def cluster_center_y_mm(self, value: float) -> None:
        self.active_roi.center_y_mm = float(value)

    @property
    def cluster_center_z_mm(self) -> float:
        return self.active_roi.center_z_mm

    @cluster_center_z_mm.setter
    def cluster_center_z_mm(self, value: float) -> None:
        self.active_roi.center_z_mm = float(value)

    @property
    def cluster_box_half_x_mm(self) -> float:
        return self.active_roi.box_half_x_mm

    @cluster_box_half_x_mm.setter
    def cluster_box_half_x_mm(self, value: float) -> None:
        self.active_roi.box_half_x_mm = float(value)

    @property
    def cluster_box_half_y_mm(self) -> float:
        return self.active_roi.box_half_y_mm

    @cluster_box_half_y_mm.setter
    def cluster_box_half_y_mm(self, value: float) -> None:
        self.active_roi.box_half_y_mm = float(value)

    @property
    def cluster_box_half_z_mm(self) -> float:
        return self.active_roi.box_half_z_mm

    @cluster_box_half_z_mm.setter
    def cluster_box_half_z_mm(self, value: float) -> None:
        self.active_roi.box_half_z_mm = float(value)

    @property
    def cluster_atlas_label(self) -> Optional[str]:
        return self.active_roi.atlas_label

    @cluster_atlas_label.setter
    def cluster_atlas_label(self, value: Optional[str]) -> None:
        self.active_roi.atlas_label = value

    @property
    def library_roi_radius_m(self) -> float:
        """Sphere radius in meters (legacy unit). The per-ROI storage
        is in mm to match the spatial-layer convention -- the meter
        view is kept for back-compat with pre-PR-#55 panel code and
        tests that compute ``radius_m * 1000`` at the boundary."""
        return self.active_roi.radius_mm / 1000.0

    @library_roi_radius_m.setter
    def library_roi_radius_m(self, value: float) -> None:
        self.active_roi.radius_mm = float(value) * 1000.0

    @property
    def library_roi_painted(self) -> Set[str]:
        """Painted-key set for the active ROI. Returned by reference so
        callers that do ``state.library_roi_painted.add(...)`` or
        ``.clear()`` mutate the active slot's set directly -- matches
        the pre-PR-#55 contract where this attribute *was* a mutable
        set on AppState."""
        return self.active_roi.painted

    @library_roi_painted.setter
    def library_roi_painted(self, value: Set[str]) -> None:
        self.active_roi.painted = set(value)

    # ------------------------------------------------------------------
    # PR #55: ROI list manipulation helpers.
    # ------------------------------------------------------------------

    def add_roi(self) -> ROISlot:
        """Append a fresh ROI to ``cluster_rois`` and make it active.

        The new ROI inherits no state from the previous active slot --
        it starts at the dataclass defaults. Auto-names it
        "ROI <n>" where n is the new length of the list.

        Returns the new slot so callers can stamp additional state
        onto it before publishing the change.
        """
        name = f"ROI {len(self.cluster_rois) + 1}"
        slot = ROISlot(name=name)
        self.cluster_rois.append(slot)
        self.cluster_active_index = len(self.cluster_rois) - 1
        return slot

    def set_active_roi(self, index: int) -> None:
        """Switch the active ROI index. No-op if out of range."""
        if 0 <= index < len(self.cluster_rois):
            self.cluster_active_index = index

    def clear_active_roi(self) -> None:
        """Reset the active ROI to default geometry (CLEAR ROI button).

        If there are 2+ ROIs, removes the active slot and advances the
        active index to the previous slot (or 0 if we removed slot 0).
        If there's exactly one ROI, resets its fields to defaults
        rather than removing it -- the proxy properties always need
        at least one slot to point at, and "clear" on a fresh project
        shouldn't disappear the list entirely.
        """
        if len(self.cluster_rois) <= 1:
            self.cluster_rois[0] = ROISlot()
            self.cluster_active_index = 0
            return
        del self.cluster_rois[self.cluster_active_index]
        self.cluster_active_index = max(0, self.cluster_active_index - 1)
        # Renumber default names so the list stays in "ROI 1 ... ROI N"
        # order. Custom-named slots (once renaming lands) would be
        # detected by checking against the auto-name pattern; for now
        # every name is auto so a simple re-stamp is correct.
        for i, slot in enumerate(self.cluster_rois):
            slot.name = f"ROI {i + 1}"

    def subscribe(self, event: str, callback: EventCallback) -> None:
        """Register ``callback`` to be called on ``publish(event, ...)``.

        Multiple subscribers per event are supported and called in
        registration order. Re-subscribing the same callable for the same
        event adds a duplicate registration — callers responsible for
        avoiding duplicate registration if they re-run their setup
        (e.g. ``ui.refreshable`` bodies should subscribe once at module
        scope, not inside the refreshable function).
        """
        self.subscribers.setdefault(event, []).append(callback)

    def unsubscribe(self, event: str, callback: EventCallback) -> bool:
        """Remove ``callback`` from ``event``'s subscriber list.

        Returns True if a registration was removed, False if no matching
        registration was found. Removes only one registration per call
        (matching the first occurrence) so duplicate registrations from
        accidental re-subscription require multiple unsubscribe calls.
        """
        callbacks = self.subscribers.get(event)
        if not callbacks:
            return False
        try:
            callbacks.remove(callback)
            return True
        except ValueError:
            return False

    def publish(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Call every subscriber of ``event`` with the given args.

        Subscriber exceptions are logged and swallowed so one buggy panel
        cannot break event delivery to the others. This matches the GUI's
        broader "errors go to state.last_error, not the user's view" stance.
        """
        for callback in list(self.subscribers.get(event, [])):
            try:
                callback(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Subscriber %r raised on event %r: %s",
                    callback, event, exc,
                )

    def set_busy(self, value: bool) -> None:
        """Toggle the busy flag and notify subscribers.

        ``workers.run_in_background`` calls this with ``True`` before
        dispatching the worker thread and ``False`` after it returns
        (success or failure). The project picker subscribes to
        ``busy_changed`` to disable Open / Close menu items while a
        long task is running — without this, switching projects mid-
        estimate would silently land the result on the new project.
        """
        if self.busy == value:
            return
        self.busy = value
        self.publish("busy_changed", value)

    def set_manifest(self, manifest: Optional[Manifest]) -> None:
        """Swap the active project manifest and notify subscribers.

        Pass ``None`` to clear. The ``project_changed`` event fires AFTER the
        manifest field is updated so subscribers reading ``state.manifest``
        from inside their callback see the new value. Subscribers themselves
        are not cleared — panels stay subscribed across project switches in
        the single-shell GUI; their handlers re-read state and refresh.
        """
        self.manifest = manifest
        self.publish("project_changed", manifest)

    def reset(self) -> None:
        """Return to the welcome-screen state.

        Used when the user closes the current project / switches datasets.
        Drops cached Raws (both source and processed) so memory is released.
        The RawCache instances are kept (not reassigned) so any references
        held elsewhere stay valid. Event subscribers and the estimated
        Montage are also cleared — a fresh dataset is a clean slate.
        """
        self.manifest = None
        self.selected_scan = None
        self.raw_cache.clear()
        self.processed_cache.clear()
        self.preload_path = None
        self.busy = False
        self.estimation_progress = None
        self.last_error = None
        self.subscribers.clear()
        self.montage = None
        self.montage_source_scan = None
        self.activity_raw = None
        self.quality_metrics.clear()
        # Note: library_hbo / library_hbr are deliberately NOT cleared by
        # reset(). They hold immutable bundled data loaded once per process;
        # re-loading on every dataset switch would burn ~100 ms unnecessarily.
        self.library_filter.clear()
        self.library_selected_hrf = None
        # Reset to the default-on state — researchers expect both
        # context overlays when they re-enter the library page.
        self.library_show_brain = True
        self.library_show_scalp = True
        self.library_oxygenation = "both"
        # PR #55: per-ROI cluster state lives in ``cluster_rois``;
        # reset to the default-single-slot list.
        self.cluster_rois = _default_rois()
        self.cluster_active_index = 0
        self.cluster_roi_active = False
        self.cluster_atlas_alignment_affine = None
        self.cluster_atlas_offset_x_mm = 0.0
        self.cluster_atlas_offset_y_mm = 0.0
        self.cluster_atlas_offset_z_mm = 0.0
        self.last_saved_roi_path = None
        self.hrf_selected_channel = None


# Module-level singleton. Page handlers and components import this directly.
state = AppState()
