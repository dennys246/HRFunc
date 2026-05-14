"""Targeted unit tests for feat/gui-sprint-5-polish (v1.3.0 Sprint 5).

Covers the Export panel side of Sprint 5:

- ``state.hrf_selected_channel`` field default + reset behavior.
- ``save_montage_sync`` — delegates to ``montage.save``.
- ``save_hrf_plots_sync`` — writes one PNG per channel with non-empty
  trace; mkdir's the target folder; skips degenerate channels.
- ``save_quality_csv_sync`` — flattens state.quality_metrics to a CSV;
  header columns match the QualityMetrics fields; one row per
  (scan, stage).
- Export panel render — five rows present, each appropriately
  enabled/disabled based on what's in state.
- Workspace dispatches Export tab to the panel.
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("nicegui")

from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.components import export_panel, hrf_panel, quality_panel  # noqa: E402
from hrfunc.gui.state import AppState, state as global_state  # noqa: E402
from hrfunc.io.manifest import Manifest, ScanEntry  # noqa: E402

gui_app._register_pages()


def _make_fake_raw():
    import mne
    info = mne.create_info(
        ch_names=["S1_D1 hbo", "S1_D1 hbr"], sfreq=10.0, ch_types="misc"
    )
    return mne.io.RawArray(np.zeros((2, 50)), info, verbose="ERROR")


class _FakeNode:
    """Minimal HRF-node-like object for testing the gallery + plot exporters."""

    def __init__(self, trace, trace_std=None, sfreq=10.0):
        self.trace = np.asarray(trace, dtype=np.float64)
        self.trace_std = (
            np.asarray(trace_std, dtype=np.float64)
            if trace_std is not None else None
        )
        self.sfreq = sfreq


class _FakeMontage:
    """Minimal Montage stand-in for export tests — just channels + save()."""

    def __init__(self, channels):
        self.channels = channels
        self.save_called_with = None

    def save(self, filename):
        self.save_called_with = filename


# ---------------------------------------------------------------------------
# state.hrf_selected_channel lifecycle
# ---------------------------------------------------------------------------


class TestStateHrfSelectedChannel:
    def test_field_defaults_none(self):
        s = AppState()
        assert s.hrf_selected_channel is None

    def test_reset_clears_field(self):
        s = AppState()
        s.hrf_selected_channel = "S1_D1 hbo"
        s.reset()
        assert s.hrf_selected_channel is None


# ---------------------------------------------------------------------------
# Sync exporters
# ---------------------------------------------------------------------------


class TestSaveMontageSync:
    def test_delegates_to_montage_save(self, tmp_path):
        m = _FakeMontage(channels={})
        out = tmp_path / "montage.json"
        export_panel.save_montage_sync(m, out)
        assert m.save_called_with == str(out)


class TestSaveHrfPlotsSync:
    def test_writes_one_png_per_channel_with_trace(self, tmp_path):
        montage = _FakeMontage(
            channels={
                "S1_D1 hbo": _FakeNode(trace=np.sin(np.linspace(0, 2, 30))),
                "S1_D1 hbr": _FakeNode(trace=np.cos(np.linspace(0, 2, 30))),
            }
        )
        folder = tmp_path / "plots"
        count = export_panel.save_hrf_plots_sync(
            montage, folder, prefix="sub-01"
        )
        assert count == 2
        files = sorted(folder.glob("*.png"))
        assert len(files) == 2
        assert all(f.stat().st_size > 0 for f in files)
        # Filename includes prefix + safe channel name
        names = {f.name for f in files}
        assert "sub-01_S1_D1_hbo.png" in names

    def test_skips_empty_traces(self, tmp_path):
        montage = _FakeMontage(
            channels={
                "good": _FakeNode(trace=np.sin(np.linspace(0, 1, 10))),
                "empty": _FakeNode(trace=np.array([])),
            }
        )
        folder = tmp_path / "plots"
        count = export_panel.save_hrf_plots_sync(
            montage, folder, prefix="sub"
        )
        assert count == 1

    def test_creates_target_folder(self, tmp_path):
        montage = _FakeMontage(
            channels={"x": _FakeNode(trace=np.sin(np.linspace(0, 1, 10)))}
        )
        folder = tmp_path / "nested" / "subdir" / "plots"
        assert not folder.exists()
        export_panel.save_hrf_plots_sync(montage, folder, prefix="p")
        assert folder.exists()


class TestSaveQualityCsvSync:
    def test_writes_csv_with_header_and_rows(self, tmp_path):
        metrics = {
            Path("/tmp/scan_a.snirf"): {
                "raw": quality_panel.QualityMetrics(
                    snr_mean=2.5, skew_mean=0.1, kurtosis_mean=3.0,
                    sci_mean=0.95, n_channels=4,
                ),
                "preprocessed": quality_panel.QualityMetrics(
                    snr_mean=3.0, skew_mean=0.2, kurtosis_mean=2.8,
                    sci_mean=None, n_channels=4,
                ),
            },
            Path("/tmp/scan_b.snirf"): {
                "raw": quality_panel.QualityMetrics(
                    snr_mean=1.5, skew_mean=0.05, kurtosis_mean=4.0,
                    sci_mean=0.88, n_channels=3,
                ),
            },
        }
        out = tmp_path / "metrics.csv"
        rows_written = export_panel.save_quality_csv_sync(metrics, out)
        # 2 stages for scan_a + 1 stage for scan_b = 3 rows
        assert rows_written == 3

        with open(out) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == [
                "scan_path", "stage", "n_channels", "snr_mean",
                "skew_mean", "kurtosis_mean", "sci_mean",
            ]
            rows = list(reader)
            assert len(rows) == 3
        # Sanity-check one row
        sample = next(
            r for r in rows
            if r[0].endswith("scan_a.snirf") and r[1] == "raw"
        )
        assert sample[2] == "4"
        assert float(sample[3]) == 2.5

    def test_empty_metrics_returns_zero_rows(self, tmp_path):
        out = tmp_path / "empty.csv"
        rows = export_panel.save_quality_csv_sync({}, out)
        assert rows == 0
        # Header still written
        with open(out) as f:
            assert "scan_path" in f.readline()


# ---------------------------------------------------------------------------
# Export panel rendering — User fixture
# ---------------------------------------------------------------------------


async def test_panel_prompts_when_no_scan(user: User, tmp_path):
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                         display_name="a"),),
    )
    await user.open("/")
    await user.should_see("Export")
    await user.should_see("Select a scan from the dataset tree")


async def test_panel_all_five_rows_visible(user: User, tmp_path):
    """When a scan is selected, all five exporter rows render even if
    most buttons are disabled."""
    scan = ScanEntry(
        format="snirf", path=tmp_path / "a.snirf", display_name="a"
    )
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    await user.open("/")
    # Each row's title appears as visible text
    await user.should_see("Processed Raw")
    await user.should_see("Activity Raw")
    await user.should_see("Montage HRFs")
    await user.should_see("HRF plots")
    await user.should_see("Quality metrics")


async def test_panel_processed_row_hint_when_unprocessed(
    user: User, tmp_path
):
    scan = ScanEntry(
        format="snirf", path=tmp_path / "a.snirf", display_name="a"
    )
    global_state.reset()
    global_state.manifest = Manifest(root=tmp_path, scans=(scan,))
    global_state.selected_scan = scan
    await user.open("/")
    await user.should_see("Run the Preprocess tab first")


async def test_panel_subscribes_to_all_state_events(user: User, tmp_path):
    """Export panel re-renders on every state change so button enablement
    tracks the workflow stages."""
    global_state.reset()
    global_state.manifest = Manifest(
        root=tmp_path,
        scans=(ScanEntry(format="snirf", path=tmp_path / "a.snirf",
                         display_name="a"),),
    )
    await user.open("/")
    # Each event has at least one subscriber from the Export panel +
    # other tabs.
    for event in (
        "scan_selected", "scan_loaded", "preprocess_done",
        "hrf_estimated", "activity_estimated", "quality_computed",
    ):
        assert event in global_state.subscribers


# ---------------------------------------------------------------------------
# HRF gallery refactor (Sprint 5.1) — _gather_channel_traces + click handler
# ---------------------------------------------------------------------------


class TestGatherChannelTraces:
    def test_filters_empty_traces(self):
        montage = _FakeMontage(channels={
            "good": _FakeNode(trace=np.sin(np.linspace(0, 1, 10))),
            "empty": _FakeNode(trace=np.array([])),
            "all_zeros": _FakeNode(trace=np.zeros(10)),
        })
        out = hrf_panel._gather_channel_traces(montage)
        assert set(out.keys()) == {"good"}

    def test_handles_montage_without_channels_attr(self):
        out = hrf_panel._gather_channel_traces(object())
        assert out == {}


class TestOnChannelClick:
    def test_updates_state_and_publishes_selection_event(self):
        """The click handler now publishes the focused
        ``hrf_selection_changed`` event (not the global ``hrf_estimated``)
        so only the HRFs-tab body refreshes, avoiding cascading re-renders
        in every other workspace tab."""
        s = AppState()
        s.selected_scan = ScanEntry(
            format="snirf", path=Path("/tmp/x"), display_name="x"
        )
        selection_published = []
        estimated_published = []
        s.subscribe(
            "hrf_selection_changed",
            lambda payload: selection_published.append(payload),
        )
        s.subscribe(
            "hrf_estimated",
            lambda payload: estimated_published.append(payload),
        )
        hrf_panel._on_channel_click(s, "S1_D1 hbo")
        assert s.hrf_selected_channel == "S1_D1 hbo"
        assert selection_published == ["S1_D1 hbo"]
        # hrf_estimated should NOT fire on selection change (otherwise
        # every other tab would refresh per click).
        assert estimated_published == []


class TestSafeFilename:
    """The Windows-friendly filename sanitizer for HRF plot exports."""

    def test_replaces_unsafe_characters(self):
        assert export_panel._safe_filename("S1/D1 hbo") == "S1_D1_hbo"
        assert export_panel._safe_filename("ch:1*?<>|\"\\") == "ch_1"

    def test_collapses_runs_of_underscores(self):
        assert export_panel._safe_filename("a   b") == "a_b"

    def test_strips_leading_trailing_dots(self):
        assert export_panel._safe_filename(".hidden.") == "hidden"

    def test_empty_after_sanitization_falls_back(self):
        assert export_panel._safe_filename("???") == "channel"

    def test_preserves_dots_in_middle(self):
        assert export_panel._safe_filename("v1.2.3") == "v1.2.3"
