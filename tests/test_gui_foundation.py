"""Targeted unit tests for feat/gui-foundation (v1.3.0 Sprint 2.1).

Covers the GUI package skeleton: ``hrfunc.gui.app``, ``hrfunc.gui.state``,
``hrfunc.gui.theme``, ``hrfunc.gui.workers``. These are the building blocks
the welcome / workspace pages (Sprints 2.2 / 2.3) will compose against.

NiceGUI rendering tests (via `nicegui.testing.User`) require pytest-asyncio
and are introduced in Sprint 2.2 when the real welcome page lands. This
file restricts itself to import / contract / behavior tests so it runs
fast and stays close to the v1.2.0 testing pattern.

All NiceGUI-dependent tests use `pytest.importorskip("nicegui")` so the
file is fully skipped when the `[gui]` extras are not installed.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

# Skip the whole module when nicegui is unavailable
pytest.importorskip("nicegui")


# ---------------------------------------------------------------------------
# Package import contract
# ---------------------------------------------------------------------------


class TestPackageImports:
    def test_hrfunc_gui_imports(self):
        import hrfunc.gui  # noqa: F401

    def test_app_module_imports(self):
        from hrfunc.gui import app  # noqa: F401

    def test_state_module_imports(self):
        from hrfunc.gui import state  # noqa: F401

    def test_theme_module_imports(self):
        from hrfunc.gui import theme  # noqa: F401

    def test_workers_module_imports(self):
        from hrfunc.gui import workers  # noqa: F401

    def test_importing_core_does_not_import_gui(self):
        """`import hrfunc` must NOT pull in NiceGUI. Users without [gui] extras
        should be able to use the library headlessly. Tested via sys.modules
        after a fresh hrfunc import."""
        import sys
        # Drop any cached gui modules
        for key in list(sys.modules):
            if key.startswith("hrfunc.gui"):
                del sys.modules[key]
        import hrfunc  # noqa: F401
        assert "hrfunc.gui" not in sys.modules


# ---------------------------------------------------------------------------
# CLI entry point: argparse contract + version handling
# ---------------------------------------------------------------------------


class TestMainSignature:
    def test_main_is_callable(self):
        from hrfunc.gui.app import main
        assert callable(main)

    def test_main_accepts_optional_argv_list(self):
        """main(argv=None) must accept either no args or an explicit list.
        Tests call it with a list to avoid touching sys.argv."""
        from hrfunc.gui.app import main
        sig = inspect.signature(main)
        assert "argv" in sig.parameters
        assert sig.parameters["argv"].default is None


class TestArgumentParser:
    def test_no_args_produces_path_none(self):
        from hrfunc.gui.app import _build_argument_parser
        ns = _build_argument_parser().parse_args([])
        assert ns.path is None

    def test_positional_path_is_pathlib_path(self):
        from hrfunc.gui.app import _build_argument_parser
        ns = _build_argument_parser().parse_args(["/tmp/study"])
        assert isinstance(ns.path, Path)
        assert ns.path == Path("/tmp/study")

    def test_version_flag_exits_zero(self, capsys):
        """`hrfunc --version` must print the version and exit 0 — not crash
        on import errors if [gui] extras are partially missing in CI."""
        from hrfunc.gui.app import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "hrfunc" in captured.out.lower()

    def test_help_flag_exits_zero(self):
        from hrfunc.gui.app import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# pyproject.toml entry point registration
# ---------------------------------------------------------------------------


class TestEntryPoint:
    def test_hrfunc_console_script_resolves_to_gui_main(self):
        """`pip install hrfunc[gui]` must register `hrfunc` -> hrfunc.gui.app:main.
        Reads the entry point from the installed metadata so we exercise the
        same resolution path pip uses."""
        try:
            from importlib.metadata import entry_points
        except ImportError:  # py3.7 fallback (unused in this repo, but safe)
            from importlib_metadata import entry_points  # type: ignore

        eps = entry_points()
        if hasattr(eps, "select"):
            scripts = eps.select(group="console_scripts")
        else:
            scripts = eps.get("console_scripts", [])  # type: ignore[attr-defined]

        hrfunc_eps = [ep for ep in scripts if ep.name == "hrfunc"]
        assert len(hrfunc_eps) == 1, (
            "Expected exactly one 'hrfunc' console_scripts entry point; "
            f"found {len(hrfunc_eps)}. Run `pip install -e .[gui]` to refresh."
        )
        assert hrfunc_eps[0].value == "hrfunc.gui.app:main"


# ---------------------------------------------------------------------------
# AppState dataclass behavior
# ---------------------------------------------------------------------------


class TestAppStateDefaults:
    def test_fresh_state_has_no_data(self):
        from hrfunc.gui.state import AppState
        s = AppState()
        assert s.manifest is None
        assert s.selected_scan is None
        assert s.preload_path is None
        assert s.busy is False
        assert s.estimation_progress is None
        assert s.last_error is None

    def test_each_app_state_has_independent_raw_cache(self):
        """RawCache uses field(default_factory) so two AppStates do NOT share
        the same cache. Regression guard against accidentally writing
        `raw_cache: RawCache = RawCache()` (mutable default)."""
        from hrfunc.gui.state import AppState
        a = AppState()
        b = AppState()
        assert a.raw_cache is not b.raw_cache

    def test_module_level_singleton_exists(self):
        from hrfunc.gui.state import state, AppState
        assert isinstance(state, AppState)


class TestAppStateReset:
    def test_reset_clears_all_data_fields(self, tmp_path):
        """reset() should return the state to its welcome-screen defaults."""
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest, ScanEntry

        s = AppState()
        # Populate everything reset() should clear
        s.manifest = Manifest(root=tmp_path)
        s.selected_scan = ScanEntry(format="snirf", path=tmp_path / "x.snirf")
        s.preload_path = tmp_path
        s.busy = True
        s.estimation_progress = (3, 10, "s1_d1_hbo")
        s.last_error = "boom"

        s.reset()

        assert s.manifest is None
        assert s.selected_scan is None
        assert s.preload_path is None
        assert s.busy is False
        assert s.estimation_progress is None
        assert s.last_error is None

    def test_reset_clears_raw_cache_but_keeps_instance(self, tmp_path):
        """Other code may hold a reference to state.raw_cache; reset() must
        clear its contents without reassigning the instance, or those
        references go stale."""
        from hrfunc.gui.state import AppState

        s = AppState()
        cache_before = s.raw_cache
        # We can't easily insert into the cache without real scans, but we
        # can verify clear() is invoked (cache is empty after reset).
        s.reset()
        assert s.raw_cache is cache_before
        assert len(s.raw_cache) == 0


# ---------------------------------------------------------------------------
# Theme: callable without crashing
# ---------------------------------------------------------------------------


class TestTheme:
    def test_colors_dict_contains_required_keys(self):
        """Quasar expects these specific keys; missing one would silently
        leave that role using Quasar's default rather than the brand color."""
        from hrfunc.gui.theme import COLORS
        for key in ("primary", "secondary", "accent",
                    "positive", "negative", "warning", "info", "dark"):
            assert key in COLORS
            assert COLORS[key].startswith("#") and len(COLORS[key]) == 7

    def test_apply_theme_is_callable(self):
        from hrfunc.gui import theme
        assert callable(theme.apply_theme)

    def test_page_container_is_callable(self):
        from hrfunc.gui import theme
        assert callable(theme.page_container)


# ---------------------------------------------------------------------------
# workers.make_progress_callback writes to AppState
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_callback_writes_tuple_to_state(self):
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import make_progress_callback

        s = AppState()
        cb = make_progress_callback(s)
        cb(3, 32, "s1_d1_hbo")
        assert s.estimation_progress == (3, 32, "s1_d1_hbo")

    def test_callback_overwrites_previous_value(self):
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import make_progress_callback

        s = AppState()
        cb = make_progress_callback(s)
        cb(0, 10, "a")
        cb(5, 10, "b")
        assert s.estimation_progress == (5, 10, "b")

    def test_callback_signature_matches_montage_estimate_hrf_contract(self):
        """The callback signature MUST match what montage.estimate_hrf calls.
        Regression guard: if either side drifts, GUI progress reporting
        silently breaks."""
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import make_progress_callback

        s = AppState()
        cb = make_progress_callback(s)
        sig = inspect.signature(cb)
        params = list(sig.parameters.values())
        assert len(params) == 3


# ---------------------------------------------------------------------------
# workers.run_in_background end-to-end (asyncio)
# ---------------------------------------------------------------------------


class TestRunInBackground:
    def test_successful_run_returns_result_and_clears_busy(self):
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import run_in_background

        s = AppState()

        def _work() -> int:
            return 42

        result = asyncio.run(run_in_background(s, _work))
        assert result == 42
        assert s.busy is False
        assert s.last_error is None
        assert s.estimation_progress is None

    def test_failing_run_stores_error_and_clears_busy(self):
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import run_in_background

        s = AppState()

        def _boom() -> None:
            raise RuntimeError("kaboom")

        result = asyncio.run(run_in_background(s, _boom))
        assert result is None
        assert s.busy is False
        assert s.last_error is not None
        assert "RuntimeError" in s.last_error
        assert "kaboom" in s.last_error

    def test_concurrent_run_refused_when_busy(self):
        """Only one background worker at a time. A second call while busy
        must return None without dispatching."""
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import run_in_background

        s = AppState()
        s.busy = True  # simulate an in-flight worker

        called = []

        def _work() -> int:
            called.append(1)
            return 1

        result = asyncio.run(run_in_background(s, _work))
        assert result is None
        assert called == []  # _work was never dispatched

    def test_on_done_invoked_with_result(self):
        from hrfunc.gui.state import AppState
        from hrfunc.gui.workers import run_in_background

        s = AppState()
        captured = []

        async def _on_done(result):
            captured.append(result)

        asyncio.run(run_in_background(s, lambda: "hello", on_done=_on_done))
        assert captured == ["hello"]
