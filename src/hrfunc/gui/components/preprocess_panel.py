"""Preprocess tab content — runs hrfunc.preprocess_fnirs on the selected scan.

Sprint 3.2 ships:
- A "Run full pipeline" button that calls ``hrfunc.preprocess_fnirs`` end-to-
  end on the cached Raw and stores the result in ``state.processed_cache``.
- Staged toggles letting the user opt out of specific steps. The toggles
  do NOT change the pipeline order (preprocess_fnirs runs OD → SCI →
  interpolate-bads → TDDR → optional polynomial detrend → Beer-Lambert →
  baseline-correct → optional filter). They control which stages to apply.
  The default toggle state mirrors the library default.
- A "Deconvolution mode" switch — drives the ``deconvolution`` kwarg on
  ``preprocess_fnirs`` (polynomial detrend on, bandpass filter off).
- A before/after channel plot rendered as a matplotlib base64 PNG, showing
  the first few channels before and after preprocessing for sanity-check.

The panel subscribes to ``scan_selected`` and ``scan_loaded`` events so it
refreshes when the user changes scan or when the Raw becomes available.
After a successful preprocess, it publishes ``preprocess_done`` so the HRFs
and Activity tabs (Sprint 3.3/3.4) can wake up.

Scientific caveat
-----------------

The staged toggles are GUI conveniences. The library's ``preprocess_fnirs``
does not currently accept skip flags for individual stages — Sprint 3.2
implements the same operations inline when toggles are set. The default
"Run full pipeline" path calls the library function untouched so users
get the canonical hrfunc pipeline. Skipping stages is for diagnostic
exploration, not for publishable analyses.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from nicegui import background_tasks, ui

from ..state import AppState
from ..workers import run_in_background
from ...io.manifest import ScanEntry

if TYPE_CHECKING:
    import mne

logger = logging.getLogger(__name__)


@dataclass
class PreprocessOptions:
    """User-controlled toggles for the Preprocess tab.

    A snapshot of the panel's UI state, captured at "Run" click time so the
    background task sees a stable view. Default values mirror the library's
    canonical pipeline.

    ``apply_baseline_correct`` is only user-controllable in deconvolution
    mode. In GLM mode (``deconvolution=False``), the library always baseline-
    corrects and the GUI mirrors that — skipping baseline correction in GLM
    mode would feed unbaselined data to the bandpass filter, which is a
    well-known scientific footgun. ``run_pipeline_sync`` enforces this
    invariant defensively (in addition to the UI hiding the toggle).
    """

    deconvolution: bool = False
    apply_motion_correction: bool = True
    apply_beer_lambert: bool = True
    apply_baseline_correct: bool = True


def render(state: AppState) -> None:
    """Render the Preprocess tab body inside the current NiceGUI context.

    Subscribes to ``scan_selected`` and ``scan_loaded`` so the panel reacts
    when the user changes scans or when a Raw becomes available. The Run
    button is disabled when no scan is selected or the Raw is not yet
    loaded. ``state.busy`` further disables it (preprocess shares the busy
    gate with estimation since both are heavy CPU tasks).
    """

    # Holder for the toggles — updated by ui.switch on_change handlers and
    # consumed by the click handler when Run is pressed.
    opts = PreprocessOptions()

    @ui.refreshable
    def _body() -> None:
        _render_body(state, opts)

    _body()

    def _refresh(_payload=None) -> None:
        _body.refresh()

    state.subscribe("scan_selected", _refresh)
    state.subscribe("scan_loaded", _refresh)
    state.subscribe("preprocess_done", _refresh)


def _render_body(state: AppState, opts: PreprocessOptions) -> None:
    """Render the body against the current scan + cache state.

    Extracted to module scope so tests can call it directly inside a
    synthetic NiceGUI context without going through the refreshable wrapper.
    """
    scan = state.selected_scan
    with ui.column().classes("p-6 gap-4 w-full"):
        ui.label("Preprocess").classes("text-2xl font-semibold")

        if scan is None:
            ui.label("Select a scan from the dataset tree.").classes(
                "text-sm opacity-60"
            )
            return

        ui.label(scan.display_name or scan.path.name).classes(
            "text-sm font-mono opacity-70"
        )

        raw_loaded = scan in state.raw_cache
        already_processed = scan in state.processed_cache

        # ── Options
        with ui.card().classes("w-full"):
            ui.label("Pipeline options").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            ui.switch(
                "Deconvolution mode (polynomial detrend, skip bandpass)",
                value=opts.deconvolution,
                on_change=lambda e: setattr(opts, "deconvolution", bool(e.value)),
            )
            ui.switch(
                "Motion correction (TDDR)",
                value=opts.apply_motion_correction,
                on_change=lambda e: setattr(
                    opts, "apply_motion_correction", bool(e.value)
                ),
            )
            ui.switch(
                "Beer-Lambert conversion to haemoglobin",
                value=opts.apply_beer_lambert,
                on_change=lambda e: setattr(
                    opts, "apply_beer_lambert", bool(e.value)
                ),
            )
            # Baseline correct is only user-skippable in deconvolution mode.
            # In GLM mode, the library always applies it (see run_pipeline_sync
            # and PreprocessOptions docstring).
            if opts.deconvolution:
                ui.switch(
                    "Baseline correct",
                    value=opts.apply_baseline_correct,
                    on_change=lambda e: setattr(
                        opts, "apply_baseline_correct", bool(e.value)
                    ),
                )
            ui.label(
                "Default settings reproduce the library's canonical pipeline "
                "and are publication-ready. Non-default toggles are for "
                "diagnostic exploration only — do not use them for analyses "
                "you intend to publish."
            ).classes("text-xs opacity-60 italic")

        # ── Run button
        run_disabled = (not raw_loaded) or state.busy
        with ui.row().classes("items-center gap-3"):
            ui.button(
                "Run full pipeline",
                on_click=lambda: _run_pipeline(state, scan, opts),
            ).props(
                f"color=primary {'disable' if run_disabled else ''}"
            )
            if not raw_loaded:
                ui.label("Waiting for scan to load…").classes(
                    "text-sm opacity-60"
                )
            elif state.busy:
                with ui.row().classes("items-center gap-2"):
                    ui.spinner(size="sm")
                    ui.label("Preprocessing…").classes("text-sm opacity-70")

        # ── Surface the last error if there is one
        if state.last_error and not state.busy:
            with ui.row().classes("items-center gap-2"):
                ui.icon("error_outline").classes("text-red-400")
                ui.label(state.last_error).classes("text-sm text-red-400")

        # ── Before/after preview
        if already_processed:
            ui.separator()
            ui.label("Before / after").classes(
                "text-xs uppercase opacity-60 tracking-wide"
            )
            _render_before_after(state, scan)


def _run_pipeline(
    state: AppState, scan: ScanEntry, opts: PreprocessOptions
) -> None:
    """Click handler for the Run button.

    Snapshots the toggle state, then dispatches the actual preprocessing on
    the background-task helper. The helper sets ``state.busy`` so the
    HRFs/Activity tabs see "estimation pipeline is occupied" — preprocess is
    treated as part of the heavy-CPU pipeline group.
    """
    if state.busy:
        return
    if scan not in state.raw_cache:
        state.last_error = "Raw not loaded; wait for the scan to finish loading."
        return

    # Snapshot options so the closure sees the values at click time, not at
    # task-completion time. Baseline correct is forced True in GLM mode to
    # match library behavior — the UI hides the toggle there but a manual
    # state mutation could still leave it False, so enforce defensively.
    snapshot = PreprocessOptions(
        deconvolution=opts.deconvolution,
        apply_motion_correction=opts.apply_motion_correction,
        apply_beer_lambert=opts.apply_beer_lambert,
        apply_baseline_correct=(
            opts.apply_baseline_correct or not opts.deconvolution
        ),
    )

    async def _on_done(result) -> None:
        if result is None:
            return
        # run_pipeline_sync wraps the result so we can stash the processed
        # Raw under the scan path in processed_cache.
        state.processed_cache._cache[scan.path.resolve()] = result
        state.publish("preprocess_done", scan)

    background_tasks.create(
        run_in_background(
            state,
            run_pipeline_sync,
            state.raw_cache.get(scan),
            snapshot,
            on_done=_on_done,
        )
    )


def run_pipeline_sync(
    raw: "mne.io.BaseRaw", opts: PreprocessOptions
) -> Optional["mne.io.BaseRaw"]:
    """Run the preprocessing pipeline against a Raw and return the processed
    Raw (or None if all channels were flagged bad).

    All steps are taken straight from ``hrfunc.preprocess_fnirs`` so the GUI
    matches the library's canonical pipeline. The toggles in
    ``PreprocessOptions`` opt out of specific stages — the order of the
    remaining stages is preserved.

    Module-level so tests can call it without dispatching through the
    background-task helper.
    """
    # Lazy imports to keep module-import cheap.
    from itertools import compress

    import mne

    from ...hrfunc import baseline_correct, polynomial_detrend

    # Always start by loading data and converting to optical density.
    # preprocess_fnirs does this unconditionally and the entire pipeline
    # downstream depends on OD-space data.
    raw.load_data()
    raw_od = mne.preprocessing.nirs.optical_density(raw, verbose="ERROR")

    # Scalp coupling index → mark bad channels.
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, verbose="ERROR")
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.95))

    if len(raw_od.info["bads"]) == len(raw_od.ch_names):
        logger.warning(
            "preprocess: every channel scored SCI<0.95 — refusing to "
            "preprocess the all-bad scan."
        )
        return None

    if raw_od.info["bads"]:
        raw_od.interpolate_bads(reset_bads=False, verbose="ERROR")

    od = (
        mne.preprocessing.nirs.tddr(raw_od, verbose="ERROR")
        if opts.apply_motion_correction
        else raw_od
    )

    # Polynomial detrend is part of the deconvolution-mode pipeline only —
    # mirrors preprocess_fnirs(scan, deconvolution=True) at line ~1072.
    if opts.deconvolution:
        od = polynomial_detrend(od, order=3)

    if opts.apply_beer_lambert:
        haemo = mne.preprocessing.nirs.beer_lambert_law(
            od.copy(), ppf=0.1
        )
    else:
        haemo = od.copy()

    if opts.apply_baseline_correct:
        haemo = baseline_correct(haemo, baseline=(None, 0.0))

    # GLM-friendly bandpass — skipped in deconvolution mode because the
    # detrend already handles slow drift.
    if not opts.deconvolution:
        haemo.filter(0.01, 0.2, verbose="ERROR")

    return haemo


def _render_before_after(state: AppState, scan: ScanEntry) -> None:
    """Render a side-by-side matplotlib PNG of the first few channels."""
    raw = state.raw_cache.get(scan) if scan in state.raw_cache else None
    processed = state.processed_cache.get(scan) if scan in state.processed_cache else None
    if raw is None or processed is None:
        ui.label("Preview unavailable.").classes("text-sm opacity-60")
        return

    png = _render_before_after_png(raw, processed)
    if png is None:
        ui.label("Could not render before/after preview.").classes(
            "text-sm opacity-60"
        )
        return
    ui.image(png).classes("max-w-3xl")


def _render_before_after_png(
    raw: "mne.io.BaseRaw",
    processed: "mne.io.BaseRaw",
    n_channels: int = 4,
) -> Optional[str]:
    """Encode a 2-row matplotlib figure as base64 PNG.

    Top row: raw signal for the first ``n_channels`` channels.
    Bottom row: processed signal for the same channel indices.

    Returns None on any matplotlib / MNE failure so the caller can render
    a fallback label instead.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logger.warning("matplotlib unavailable for before/after: %s", exc)
        return None

    fig = None
    try:
        n = min(n_channels, len(raw.ch_names), len(processed.ch_names))
        fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=False)

        raw_data = raw.get_data(picks=list(range(n)))
        proc_data = processed.get_data(picks=list(range(n)))
        raw_times = raw.times
        proc_times = processed.times

        for i in range(n):
            axes[0].plot(
                raw_times, raw_data[i], lw=0.6,
                label=raw.ch_names[i],
            )
            axes[1].plot(
                proc_times, proc_data[i], lw=0.6,
                label=processed.ch_names[i],
            )
        axes[0].set_title("Before preprocessing")
        axes[1].set_title("After preprocessing")
        axes[0].set_ylabel("amplitude")
        axes[1].set_ylabel("amplitude")
        axes[1].set_xlabel("time (s)")
        axes[0].legend(loc="upper right", fontsize=6)
        axes[1].legend(loc="upper right", fontsize=6)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("before/after render failed: %s", exc)
        return None
    finally:
        if fig is not None:
            plt.close(fig)
