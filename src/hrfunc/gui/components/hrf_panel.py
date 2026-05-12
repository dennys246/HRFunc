"""HRFs tab content — estimate HRFs from the preprocessed scan + events.

Sprint 3.3 ships:
- A "Toeplitz" mode wired to ``montage.estimate_hrf``: user picks one or more
  annotation descriptions (e.g. ``"stim_a"``, ``"task-flanker"``) from the
  scan, the panel builds an impulse-series array, dispatches estimation
  via ``workers.run_in_background``, and renders a progress bar driven by
  the ``progress_callback``.
- A "Canonical" mode that bypasses estimation entirely and renders the
  SPM-style double-gamma HRF (the same shape the library uses as a
  reference in ``correlate_canonical``). No events needed.
- Controls: model radio (toeplitz/canonical), event picker (multi-select
  toggles), lambda slider on log scale (1e-5..1e-1, default 1e-3),
  duration field (default 30.0 s).
- Result preview: matplotlib base64 PNG showing all estimated HRF traces
  overlaid by channel (toeplitz mode) or the canonical HRF (canonical mode).

The panel reads from ``state.processed_cache`` for toeplitz mode (per
Sprint 3.2 contract: preprocess in 3.2, estimate in 3.3). Canonical mode
needs only ``state.raw_cache`` (no preprocessing required to generate the
canonical shape).

The panel subscribes to ``scan_selected``, ``scan_loaded``,
``preprocess_done``, and ``hrf_estimated`` so it re-renders when any
upstream tab changes state.

Scientific notes
----------------

- ``estimate_hrf`` runs with ``preprocess=False`` here because the Raw
  comes from ``processed_cache`` (already preprocessed in 3.2). Passing
  ``preprocess=True`` would silently re-run the canonical preprocess on
  preprocessed data, which is wrong.
- The canonical double-gamma uses ``scipy.stats.gamma.pdf`` at peaks 6 s
  and 16 s (with a 1/6 undershoot weight) — identical to the library's
  ``correlate_canonical`` implementation at hrfunc.py:756-761.
- The lambda slider is log-scale: the displayed value is ``10 ** raw``
  where ``raw`` is the slider's actual ``-5..-1`` integer. The library's
  default is ``1e-3``.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from nicegui import background_tasks, ui

from ..state import AppState
from ..workers import make_progress_callback, run_in_background
from ...io.manifest import ScanEntry

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


MODEL_TOEPLITZ = "toeplitz"
MODEL_CANONICAL = "canonical"
DEFAULT_DURATION = 30.0
DEFAULT_LMBDA = 1e-3
LOG_LMBDA_MIN = -5
LOG_LMBDA_MAX = -1


@dataclass
class EstimationOptions:
    """User-controlled options for the HRFs tab.

    Snapshotted at Estimate-click time so the background task sees a stable
    view. Defaults mirror ``montage.estimate_hrf`` library defaults.
    """

    model: str = MODEL_TOEPLITZ
    lmbda: float = DEFAULT_LMBDA
    duration: float = DEFAULT_DURATION
    selected_events: Tuple[str, ...] = field(default_factory=tuple)


def render(state: AppState) -> None:
    """Render the HRFs tab body inside the current NiceGUI context.

    Subscribes a refreshable body to scan/preprocess/HRF-result events so
    the panel reacts to upstream tab changes without rebuilding the whole
    workspace.
    """
    opts = EstimationOptions()

    @ui.refreshable
    def _body() -> None:
        _render_body(state, opts)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)
    state.subscribe("hrf_estimated", _refresh)

    # Progress polling timer — refreshes the body every 0.5 s WHILE an
    # estimation is in flight, so the progress bar advances. The
    # progress_callback fires from a worker thread (run_in_executor) and
    # cannot safely refresh NiceGUI elements from there, so we poll from
    # the main loop instead. Cheap no-op when state.busy is False.
    def _poll_progress() -> None:
        if state.busy and state.estimation_progress is not None:
            _body.refresh()

    ui.timer(0.5, _poll_progress)


def _render_body(state: AppState, opts: EstimationOptions) -> None:
    """Render the HRFs body against the current state.

    Module-level so tests can call it directly inside a synthetic NiceGUI
    context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("HRFs").classes("text-2xl font-semibold")

        if scan is None:
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
            return

        ui.label(scan.display_name or scan.path.name).classes(
            "text-sm font-mono opacity-70"
        )

        # ── Model selector + controls
        with ui.card().classes("w-full"):
            ui.label("Estimation").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_model_radio(opts, _refresh_body_for(state))
            if opts.model == MODEL_TOEPLITZ:
                _render_toeplitz_controls(state, opts, scan)
            else:
                _render_canonical_note()

        # ── Run button + progress / error display
        _render_run_row(state, scan, opts)

        # ── Result preview
        if state.montage is not None:
            ui.separator()
            ui.label("HRF preview").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_hrf_preview(state, scan, opts)


def _refresh_body_for(state: AppState):
    """Return a callable that re-publishes scan_selected to refresh subscribers.

    Used by the model-radio change handler to trigger a re-render of the
    panel body when the user flips between toeplitz and canonical (the
    rendered controls differ between modes).
    """
    def _trigger() -> None:
        state.publish("scan_selected", state.selected_scan)
    return _trigger


def _render_model_radio(
    opts: EstimationOptions, on_change_refresh
) -> None:
    def _set(value: str) -> None:
        opts.model = value
        on_change_refresh()

    ui.radio(
        [MODEL_TOEPLITZ, MODEL_CANONICAL],
        value=opts.model,
        on_change=lambda e: _set(e.value),
    ).props("inline")


def _render_toeplitz_controls(
    state: AppState, opts: EstimationOptions, scan: ScanEntry
) -> None:
    """Event picker + lmbda slider + duration field for toeplitz mode."""
    raw = state.processed_cache.get(scan) if scan in state.processed_cache else None

    # ── Event picker
    ui.label("Events").classes("text-xs uppercase opacity-60 tracking-wide")
    if raw is None:
        ui.label(
            "Preprocess the scan first (Preprocess tab) before estimating."
        ).classes("text-sm opacity-60")
        return

    event_names = sorted_unique_annotation_descriptions(raw)
    if not event_names:
        ui.label(
            "No events found in the preprocessed scan."
        ).classes("text-sm opacity-60")
    else:
        # Seed selection on first render — default to all events so users
        # don't need to tick anything in the common case.
        if not opts.selected_events:
            opts.selected_events = tuple(event_names)

        selected_set = set(opts.selected_events)

        def _toggle(name: str, checked: bool) -> None:
            new_selection = set(opts.selected_events)
            if checked:
                new_selection.add(name)
            else:
                new_selection.discard(name)
            opts.selected_events = tuple(sorted(new_selection))

        with ui.row().classes("gap-2 flex-wrap"):
            for name in event_names:
                ui.checkbox(
                    name,
                    value=name in selected_set,
                    on_change=(lambda e, n=name: _toggle(n, bool(e.value))),
                )

    # ── Lambda slider (log scale)
    ui.label("Regularization (lambda)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    initial_log = int(round(np.log10(opts.lmbda))) if opts.lmbda > 0 else -3
    initial_log = max(LOG_LMBDA_MIN, min(LOG_LMBDA_MAX, initial_log))

    lmbda_display = ui.label(f"lambda = {opts.lmbda:.0e}").classes(
        "text-sm font-mono opacity-80"
    )

    def _on_lmbda_change(event) -> None:
        log_val = int(event.value)
        opts.lmbda = float(10 ** log_val)
        lmbda_display.set_text(f"lambda = {opts.lmbda:.0e}")

    ui.slider(
        min=LOG_LMBDA_MIN,
        max=LOG_LMBDA_MAX,
        step=1,
        value=initial_log,
        on_change=_on_lmbda_change,
    )

    # ── Duration
    ui.label("Duration (seconds)").classes(
        "text-xs uppercase opacity-60 tracking-wide"
    )
    ui.number(
        value=opts.duration,
        min=1.0,
        max=120.0,
        step=1.0,
        format="%.1f",
        on_change=lambda e: setattr(opts, "duration", float(e.value or DEFAULT_DURATION)),
    )

    # ── Edge-expansion advisory (library default 0.15 of duration)
    # estimate_hrf shifts every event onset back by edge_expansion*duration
    # seconds and silently drops events that would fall before t=0. With
    # the library default (0.15) and the user's current duration, that
    # means events in the first ``edge_seconds`` of the scan are lost.
    edge_seconds = 0.15 * opts.duration
    ui.label(
        f"Note: events in the first ~{edge_seconds:.1f} s of the scan are "
        f"dropped by the toeplitz edge-expansion window (library default "
        f"0.15 × duration). Consider this when designing trial-onset timing."
    ).classes("text-xs opacity-60 italic")


def _render_canonical_note() -> None:
    ui.label(
        "Canonical mode renders the SPM-style double-gamma HRF (peak at "
        "~6 s, undershoot at ~16 s) — a fixed reference shape, not "
        "data-driven. Click Generate to display."
    ).classes("text-sm opacity-70")


def _render_run_row(
    state: AppState, scan: ScanEntry, opts: EstimationOptions
) -> None:
    """Render the Run button row + progress / error display."""
    if opts.model == MODEL_TOEPLITZ:
        can_run = (
            scan in state.processed_cache
            and bool(opts.selected_events)
            and not state.busy
        )
        run_label = "Estimate HRFs"
    else:
        can_run = not state.busy
        run_label = "Generate canonical HRF"

    with ui.row().classes("items-center gap-3"):
        ui.button(
            run_label,
            on_click=lambda: _run(state, scan, opts),
        ).props(f"color=primary {'disable' if not can_run else ''}")
        if state.busy:
            prog = state.estimation_progress
            if prog is not None:
                current, total, name = prog
                fraction = (current + 1) / max(total, 1)
                with ui.column().classes("gap-1 flex-grow"):
                    ui.label(
                        f"Channel {current + 1}/{total}: {name}"
                    ).classes("text-xs opacity-70")
                    ui.linear_progress(value=fraction).classes("w-64")
            else:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("Working…").classes("text-sm opacity-70")
        elif opts.model == MODEL_TOEPLITZ and scan not in state.processed_cache:
            ui.label("Waiting for preprocess output…").classes(
                "text-sm opacity-60"
            )
        elif opts.model == MODEL_TOEPLITZ and not opts.selected_events:
            ui.label("Pick at least one event to estimate.").classes(
                "text-sm opacity-60"
            )

    if state.last_error and not state.busy:
        with ui.row().classes("items-center gap-2"):
            ui.icon("error_outline").classes("text-red-400")
            ui.label(state.last_error).classes("text-sm text-red-400")


def _run(
    state: AppState, scan: ScanEntry, opts: EstimationOptions
) -> None:
    """Click handler for Estimate / Generate.

    Snapshots options, dispatches the appropriate sync worker through
    ``workers.run_in_background``. On success, stashes the resulting
    Montage on ``state.montage`` and publishes ``hrf_estimated``.
    """
    if state.busy:
        return

    snapshot = EstimationOptions(
        model=opts.model,
        lmbda=opts.lmbda,
        duration=opts.duration,
        selected_events=opts.selected_events,
    )

    if snapshot.model == MODEL_TOEPLITZ:
        if scan not in state.processed_cache:
            state.last_error = "Preprocess the scan first."
            return
        if not snapshot.selected_events:
            state.last_error = "Pick at least one event."
            return
        raw = state.processed_cache.get(scan)
        progress_cb = make_progress_callback(state)
        sync_call = (
            run_toeplitz_sync, raw, snapshot, progress_cb
        )
    else:
        # Canonical mode doesn't need a processed Raw — only the raw_cache
        # entry to get sfreq/channel count for shaping the output.
        if scan not in state.raw_cache and scan not in state.processed_cache:
            state.last_error = "Load the scan first."
            return
        source_raw = (
            state.processed_cache.get(scan)
            if scan in state.processed_cache
            else state.raw_cache.get(scan)
        )
        sync_call = (
            run_canonical_sync, source_raw, snapshot
        )

    async def _on_done(result) -> None:
        if result is None:
            return
        state.montage = result
        # Track which scan produced this montage so the Activity tab can
        # refuse a toeplitz run when the user switches scans mid-flow.
        state.montage_source_scan = scan
        state.publish("hrf_estimated", scan)

    background_tasks.create(
        run_in_background(state, *sync_call, on_done=_on_done)
    )


def run_toeplitz_sync(
    raw: "mne.io.BaseRaw",
    opts: EstimationOptions,
    progress_callback=None,
):
    """Run montage.estimate_hrf against a preprocessed Raw and return Montage.

    Returns None if the events array can't be built (no annotation samples
    match the selection). The library's ``estimate_hrf`` is called with
    ``preprocess=False`` because the input is already preprocessed.

    Module-level so tests can call without dispatching through workers.
    """
    from ...hrfunc import montage as Montage

    events = build_events_array(raw, opts.selected_events)
    if events is None or not events.any():
        logger.warning(
            "run_toeplitz_sync: no event samples matched the selected "
            "descriptions; refusing to estimate on empty events."
        )
        return None

    m = Montage(nirx_obj=raw)
    m.estimate_hrf(
        raw,
        events=events.tolist(),
        duration=opts.duration,
        lmbda=opts.lmbda,
        preprocess=False,
        progress_callback=progress_callback,
    )
    # estimate_hrf only appends to optode.estimates; it does NOT populate
    # optode.trace (which is what the preview reads). generate_distribution
    # computes trace = mean(estimates) per channel. Without this call, the
    # HRF preview would render an empty plot after a successful estimation.
    m.generate_distribution()
    return m


def run_canonical_sync(
    raw: "mne.io.BaseRaw",
    opts: EstimationOptions,
):
    """Build a canonical SPM-style double-gamma HRF.

    Returns a lightweight object with a ``.canonical_trace`` numpy array
    and ``.duration`` / ``.sfreq`` fields. Not a real Montage — canonical
    mode doesn't go through estimate_hrf, so the per-channel structure
    isn't relevant. Sprint 3.4 (Activity) will not consume this; it has
    its own canonical path via estimate_activity(hrf_model='canonical').
    """
    sfreq = float(raw.info["sfreq"])
    trace = canonical_double_gamma(opts.duration, sfreq)
    return _CanonicalResult(
        canonical_trace=trace, duration=opts.duration, sfreq=sfreq
    )


@dataclass
class _CanonicalResult:
    """Holder for canonical-mode output.

    Deliberately not a Montage — canonical mode skips estimation entirely
    and returns a single reference HRF shape. Stored on ``state.montage``
    via duck-typing; the HRFs tab is the only consumer.
    """

    canonical_trace: np.ndarray
    duration: float
    sfreq: float


def canonical_double_gamma(duration: float, sfreq: float) -> np.ndarray:
    """SPM-style double-gamma canonical HRF.

    Standard SPM canonical: a gamma with peak at ~6 s minus a smaller
    gamma with peak at ~16 s, normalized so the positive peak is 1.0.

    The argument to ``gamma.pdf`` is in seconds, not sample indices.
    This deliberately differs from the library's ``correlate_canonical``
    (hrfunc.py:756-761) which passes raw sample indices — fine when the
    correlation is point-wise but produces an sfreq-dependent peak
    location for visualization. The GUI's canonical preview is meant to
    be the "true" SPM canonical, so it operates in seconds.
    """
    import scipy.stats

    n_samples = max(int(round(duration * sfreq)), 2)
    t_seconds = np.arange(n_samples) / sfreq
    peak1 = scipy.stats.gamma.pdf(t_seconds, 6)
    peak2 = scipy.stats.gamma.pdf(t_seconds, 16) / 6.0
    hrf = peak1 - peak2
    peak = np.max(hrf)
    if peak > 0:
        hrf = hrf / peak
    return hrf


def build_events_array(
    raw: "mne.io.BaseRaw",
    selected_descriptions: Tuple[str, ...],
) -> Optional[np.ndarray]:
    """Convert MNE annotations to a 0/1 impulse series of length n_samples.

    ``estimate_hrf`` consumes a flat list where each sample is 0 or 1; an
    event onset at time ``t`` becomes ``1`` at sample index
    ``round(t * sfreq)``. Annotations whose description is not in
    ``selected_descriptions`` are ignored. Out-of-range onsets are
    dropped silently with a logger.warning so a corrupt annotations table
    doesn't crash the estimation.

    Returns None if there are no annotations to convert at all (so the
    caller can distinguish "nothing selected" from "selection matched but
    fell outside the scan window").
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return None

    sfreq = float(raw.info["sfreq"])
    n_samples = raw.n_times
    selected_set = set(selected_descriptions)

    out = np.zeros(n_samples, dtype=np.int64)
    for ann in annotations:
        if str(ann["description"]) not in selected_set:
            continue
        sample = int(round(float(ann["onset"]) * sfreq))
        if 0 <= sample < n_samples:
            out[sample] = 1
        else:
            logger.warning(
                "build_events_array: dropping annotation at onset %.3fs "
                "(sample %d) — outside scan window 0..%d",
                float(ann["onset"]), sample, n_samples,
            )
    return out


def sorted_unique_annotation_descriptions(raw: "mne.io.BaseRaw") -> List[str]:
    """Distinct annotation description strings, sorted alphabetically.

    Empty list if the Raw has no annotations or all descriptions are empty.
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return []
    seen = sorted({str(ann["description"]) for ann in annotations})
    return [s for s in seen if s]


# ---------------------------------------------------------------------------
# Result preview rendering
# ---------------------------------------------------------------------------


def _render_hrf_preview(
    state: AppState, scan: ScanEntry, opts: EstimationOptions
) -> None:
    """Render the matplotlib preview of the most-recent estimation result."""
    result = state.montage
    if result is None:
        return
    if isinstance(result, _CanonicalResult):
        png = _render_canonical_preview_png(result)
    else:
        png = _render_toeplitz_preview_png(result)
    if png is None:
        ui.label("Preview unavailable.").classes("text-sm opacity-60")
        return
    ui.image(png).classes("max-w-3xl")


def _render_toeplitz_preview_png(montage) -> Optional[str]:
    """Overlay all channel HRF traces on a single matplotlib axes."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable: %s", exc)
        return None

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        channels = getattr(montage, "channels", {})
        plotted = 0
        for ch_name, node in channels.items():
            trace = getattr(node, "trace", None)
            if trace is None or len(trace) == 0:
                continue
            ax.plot(trace, lw=0.6, alpha=0.7, label=ch_name)
            plotted += 1
        if plotted == 0:
            ax.text(
                0.5, 0.5, "No channel HRFs available.",
                ha="center", va="center", transform=ax.transAxes,
            )
        ax.set_title("Estimated HRFs (toeplitz deconvolution)")
        ax.set_xlabel("samples")
        ax.set_ylabel("amplitude (a.u.)")
        if plotted <= 24:
            ax.legend(fontsize=6, loc="upper right")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("toeplitz preview render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)


def _render_canonical_preview_png(result: "_CanonicalResult") -> Optional[str]:
    """Render the canonical HRF as a single line plot."""
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable: %s", exc)
        return None

    fig = None
    try:
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        t = np.arange(len(result.canonical_trace)) / result.sfreq
        ax.plot(t, result.canonical_trace, lw=1.5)
        ax.set_title("Canonical HRF (double-gamma)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude (peak = 1.0)")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("canonical preview render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
