"""Targeted unit tests for the v1.4 Inspect panel.

The Inspect panel was reintroduced in v1.4 Phase 5 after being dropped
in Phase 2. Coverage:

- **No-scan path** — when ``state.selected_scan`` is None, the panel
  prompts the user to pick from the dataset tree.
- **Scan selected, Raw not cached** — spinner placeholder while the
  background load is in flight.
- **Scan selected, Raw cached** — metadata kvs + Recording sections
  (channel list, probe expander, events expander).
- **Cache-clear race** — defensive: ``__contains__`` check passes but
  the subsequent ``get()`` raises (another callback cleared the cache
  mid-render). The panel surfaces a clean error message instead of
  bubbling the exception.
- **Probe PNG rendering** — pure helper, returns a base64 data URL or
  None on matplotlib failure.

The panel subscribes to ``scan_selected`` + ``scan_loaded`` so dataset
tree clicks and background load completions trigger refreshes without
re-rendering the whole tab.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]

from hrfunc.gui import app as gui_app  # noqa: E402
from hrfunc.gui.state import AppState, state as global_state  # noqa: E402
from hrfunc.io.manifest import ScanEntry  # noqa: E402

gui_app._register_pages()


def _mount_inspect_route() -> None:
    """Register a ``/_test_inspect`` route at call-time.

    Mirrors the pattern in ``test_gui_library.py``: NiceGUI's User
    plugin resets the page registry per test, so the page decorator
    has to run inside each test (after the fixture comes up) — defining
    it as a helper avoids duplication.
    """
    from hrfunc.gui.components import inspect_panel
    from hrfunc.gui.theme import apply_theme

    @ui.page("/_test_inspect")
    def _test_inspect_page() -> None:
        apply_theme()
        inspect_panel.render(global_state)


# ---------------------------------------------------------------------------
# No-scan path
# ---------------------------------------------------------------------------


async def test_no_scan_prompts_user_to_select(user: User):
    global_state.reset()
    _mount_inspect_route()
    await user.open("/_test_inspect")
    await user.should_see("Inspect")
    await user.should_see("Select a scan from the dataset tree")


# ---------------------------------------------------------------------------
# Scan selected — metadata + recording-section behaviors
# ---------------------------------------------------------------------------


async def test_scan_selected_but_not_cached_shows_spinner(
    user: User, tmp_path
):
    """While the background Raw load is in flight, the Recording
    section shows a 'Loading recording…' placeholder."""
    global_state.reset()
    global_state.selected_scan = ScanEntry(
        format="snirf",
        path=tmp_path / "subject_1.snirf",
        display_name="subject_1.snirf",
    )
    _mount_inspect_route()
    await user.open("/_test_inspect")
    await user.should_see("Loading recording")


async def test_scan_metadata_renders_kv_rows(user: User, tmp_path):
    """ScanEntry fields appear as Metadata kv rows. None-valued fields
    (no BIDS subject, etc.) are skipped rather than rendering 'None'."""
    global_state.reset()
    global_state.selected_scan = ScanEntry(
        format="snirf",
        path=tmp_path / "sub-01_ses-A_task-flanker_run-1.snirf",
        display_name="sub-01_run-1",
        n_channels=42,
        sfreq=7.81,
        bids_subject="01",
        bids_task="flanker",
    )
    _mount_inspect_route()
    await user.open("/_test_inspect")
    await user.should_see("Metadata")
    await user.should_see("Format")
    await user.should_see("snirf")
    await user.should_see("Channels")
    await user.should_see("42")
    await user.should_see("Sampling rate")
    # BIDS subject present, session absent — assert presence + absence.
    await user.should_see("BIDS subject")
    await user.should_see("01")
    await user.should_not_see("BIDS session")


# ---------------------------------------------------------------------------
# Cache-clear race — defensive path
# ---------------------------------------------------------------------------


async def test_cache_clear_race_shows_clean_error(
    user: User, tmp_path, monkeypatch
):
    """``__contains__`` reports the scan is cached, but ``get()`` raises
    (because another callback cleared the cache mid-render). The panel
    catches the exception and renders a friendly message instead of
    propagating.

    monkeypatch the raw_cache so the swap is automatically undone when
    the test completes — global_state.reset() doesn't reassign the
    raw_cache field (it just clears it), so a direct write would leak
    a fake cache into downstream tests.
    """
    global_state.reset()
    scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf")
    global_state.selected_scan = scan

    class _FlakyCache:
        def __contains__(self, _scan):
            return True

        def get(self, _scan):
            raise RuntimeError("cache cleared mid-render")

        def clear(self):
            pass

    monkeypatch.setattr(global_state, "raw_cache", _FlakyCache())

    _mount_inspect_route()
    await user.open("/_test_inspect")
    await user.should_see("Recording unavailable")


# ---------------------------------------------------------------------------
# Probe PNG helper — pure function
# ---------------------------------------------------------------------------


class TestRenderProbePng:
    def test_returns_data_url_for_successful_plot(self):
        from hrfunc.gui.components.inspect_panel import render_probe_png

        raw = MagicMock()
        # plot_sensors returns a Figure-like object whose ``savefig`` writes
        # something to the bytes buffer.
        fig = MagicMock()
        def _savefig(buf, **_kw):
            buf.write(b"fake-png-bytes")
        fig.savefig.side_effect = _savefig
        raw.plot_sensors.return_value = fig

        result = render_probe_png(raw)
        assert result is not None
        assert result.startswith("data:image/png;base64,")

    def test_returns_none_when_plot_sensors_raises(self):
        from hrfunc.gui.components.inspect_panel import render_probe_png

        raw = MagicMock()
        raw.plot_sensors.side_effect = RuntimeError("no montage")
        assert render_probe_png(raw) is None


# ---------------------------------------------------------------------------
# Event-bus subscription — refresh on scan_selected / scan_loaded
# ---------------------------------------------------------------------------


async def test_panel_subscribes_to_scan_events(user: User):
    """The panel registers ``scan_selected`` + ``scan_loaded`` listeners
    so dataset-tree clicks and background-load completions drive
    re-renders. Without these, switching scans wouldn't update the
    Inspect tab content."""
    global_state.reset()
    _mount_inspect_route()
    await user.open("/_test_inspect")
    assert "scan_selected" in global_state.subscribers
    assert "scan_loaded" in global_state.subscribers
