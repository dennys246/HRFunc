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
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..io.manifest import Manifest, ScanEntry
from ..io.raw_cache import RawCache

logger = logging.getLogger(__name__)

EventCallback = Callable[..., None]


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
        self.hrf_selected_channel = None


# Module-level singleton. Page handlers and components import this directly.
state = AppState()
