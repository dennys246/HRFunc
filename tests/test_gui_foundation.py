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
        after a fresh hrfunc import.

        Saves and restores gui module entries so this test doesn't poison
        subsequent tests in the gate (which import-bind to the same module
        objects we'd otherwise replace)."""
        import sys
        saved_gui_modules = {
            k: v for k, v in sys.modules.items() if k.startswith("hrfunc.gui")
        }
        try:
            for key in list(sys.modules):
                if key.startswith("hrfunc.gui"):
                    del sys.modules[key]
            import hrfunc  # noqa: F401
            assert "hrfunc.gui" not in sys.modules
        finally:
            # Restore so test_gui_welcome and test_gui_workspace bind to
            # the same state/page-registration objects they imported.
            sys.modules.update(saved_gui_modules)


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


class TestClusterMultiROI:
    """PR #55 introduced per-ROI state via ``cluster_rois: list[ROISlot]``.
    The pre-existing ``cluster_*`` / ``library_roi_*`` field names now
    proxy onto the active slot so existing panel bindings keep working.

    These tests pin down the proxy semantics, list mutation helpers,
    and reset behaviour.
    """

    def test_default_state_has_one_roi_slot(self):
        from hrfunc.gui.state import AppState, ROISlot

        s = AppState()
        assert len(s.cluster_rois) == 1
        assert s.cluster_active_index == 0
        assert isinstance(s.cluster_rois[0], ROISlot)
        # Default slot matches the pre-PR-#55 defaults so callers that
        # never touch the multi-ROI helpers see the same starting state.
        assert s.cluster_shape == "sphere"
        assert s.cluster_center_x_mm == 0.0
        assert s.cluster_box_half_x_mm == 20.0
        assert s.library_roi_radius_m == 0.02
        assert s.library_roi_painted == set()

    def test_each_app_state_has_independent_roi_list(self):
        """The list uses ``field(default_factory=...)`` so two AppStates
        don't share the same list. Mirrors the raw_cache guarantee."""
        from hrfunc.gui.state import AppState

        a = AppState()
        b = AppState()
        assert a.cluster_rois is not b.cluster_rois
        a.add_roi()
        assert len(b.cluster_rois) == 1

    def test_proxy_setter_writes_to_active_slot(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.cluster_center_x_mm = 12.5
        s.cluster_shape = "atlas_region"
        s.cluster_atlas_label = "Frontal Pole"
        s.library_roi_radius_m = 0.03
        s.library_roi_painted.add("hbo:s1_d1_hbo-temp")

        slot = s.cluster_rois[0]
        assert slot.center_x_mm == 12.5
        assert slot.shape == "atlas_region"
        assert slot.atlas_label == "Frontal Pole"
        # Storage is in mm; the meters proxy converts at the boundary.
        assert slot.radius_mm == 30.0
        assert slot.painted == {"hbo:s1_d1_hbo-temp"}

    def test_painted_set_returned_by_reference(self):
        """Pre-PR-#55 panels did ``state.library_roi_painted.clear()``
        and ``.add(...)`` -- the proxy must return the same set so those
        mutations land on the active slot."""
        from hrfunc.gui.state import AppState

        s = AppState()
        s.library_roi_painted.add("k1")
        assert s.library_roi_painted is s.cluster_rois[0].painted
        s.library_roi_painted.clear()
        assert s.cluster_rois[0].painted == set()

    def test_add_roi_appends_and_activates(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.cluster_center_x_mm = 5.0
        slot2 = s.add_roi()

        assert len(s.cluster_rois) == 2
        assert s.cluster_active_index == 1
        assert slot2.name == "ROI 2"
        # New slot starts at defaults -- not inheriting from the prior.
        assert slot2.center_x_mm == 0.0
        # The prior slot is unaffected by writes through the proxy.
        s.cluster_center_x_mm = -3.0
        assert s.cluster_rois[0].center_x_mm == 5.0
        assert s.cluster_rois[1].center_x_mm == -3.0

    def test_set_active_roi_switches_proxy_view(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.cluster_center_x_mm = 1.0
        s.add_roi()
        s.cluster_center_x_mm = 2.0

        s.set_active_roi(0)
        assert s.cluster_center_x_mm == 1.0
        s.set_active_roi(1)
        assert s.cluster_center_x_mm == 2.0

    def test_set_active_roi_ignores_out_of_range(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.set_active_roi(99)  # should be a no-op, not raise
        assert s.cluster_active_index == 0
        s.set_active_roi(-1)
        assert s.cluster_active_index == 0

    def test_clear_active_roi_resets_last_slot(self):
        """With only one ROI in the list, CLEAR ROI resets it in place
        rather than removing it -- the proxy properties need at least
        one slot to point at."""
        from hrfunc.gui.state import AppState

        s = AppState()
        s.cluster_shape = "atlas_region"
        s.cluster_center_x_mm = 50.0
        s.library_roi_painted.add("k1")

        s.clear_active_roi()

        assert len(s.cluster_rois) == 1
        assert s.cluster_shape == "sphere"
        assert s.cluster_center_x_mm == 0.0
        assert s.library_roi_painted == set()

    def test_clear_active_roi_drops_active_when_multiple(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.add_roi()
        s.add_roi()
        # 3 slots: ROI 1 / ROI 2 / ROI 3 (active is the last)
        assert [r.name for r in s.cluster_rois] == ["ROI 1", "ROI 2", "ROI 3"]
        assert s.cluster_active_index == 2

        s.clear_active_roi()
        # Drop the third -> 2 remaining, active falls back to index 1.
        assert len(s.cluster_rois) == 2
        assert s.cluster_active_index == 1
        # Names re-stamped so the list stays "ROI 1 ... ROI N" in order.
        assert [r.name for r in s.cluster_rois] == ["ROI 1", "ROI 2"]

    def test_active_roi_recovers_from_out_of_range_index(self):
        """Defensive: if external state lands the index out of range,
        the property clamps to 0 rather than raising."""
        from hrfunc.gui.state import AppState

        s = AppState()
        s.cluster_active_index = 99
        slot = s.active_roi
        assert slot is s.cluster_rois[0]
        assert s.cluster_active_index == 0

    def test_reset_returns_to_one_default_slot(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        s.add_roi()
        s.add_roi()
        s.cluster_shape = "atlas_region"
        s.cluster_atlas_label = "Frontal Pole"
        s.cluster_roi_active = True

        s.reset()

        assert len(s.cluster_rois) == 1
        assert s.cluster_active_index == 0
        assert s.cluster_shape == "sphere"
        assert s.cluster_atlas_label is None
        assert s.cluster_roi_active is False


class TestSetBusy:
    """``set_busy`` toggles the flag and publishes ``busy_changed``.

    Used by the v1.4 project picker dropdown to disable Open / Close
    while a long task is running. The event mechanism (vs polling) lets
    UI components subscribe to the change without re-checking the flag
    on every render.
    """

    def test_set_busy_updates_field(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        assert s.busy is False
        s.set_busy(True)
        assert s.busy is True

    def test_set_busy_publishes_event(self):
        from hrfunc.gui.state import AppState

        s = AppState()
        received = []
        s.subscribe("busy_changed", lambda v: received.append(v))

        s.set_busy(True)
        s.set_busy(False)
        assert received == [True, False]

    def test_set_busy_no_op_when_value_unchanged(self):
        """Setting the same value twice should NOT publish a duplicate
        event — the picker would refresh its menu twice for nothing."""
        from hrfunc.gui.state import AppState

        s = AppState()
        received = []
        s.subscribe("busy_changed", lambda v: received.append(v))

        s.set_busy(True)
        s.set_busy(True)
        assert received == [True]


class TestSetManifest:
    """``set_manifest`` swaps the active project and fires ``project_changed``.

    Used by the v1.4 single-shell GUI's project picker dropdown. Subscribers
    blank or rebuild their refreshables when the manifest changes underneath
    them.
    """

    def test_set_manifest_updates_field(self, tmp_path):
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        s = AppState()
        m = Manifest(root=tmp_path)
        s.set_manifest(m)
        assert s.manifest is m

    def test_set_manifest_publishes_project_changed(self, tmp_path):
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        s = AppState()
        received = []
        s.subscribe("project_changed", lambda m: received.append(m))

        m = Manifest(root=tmp_path)
        s.set_manifest(m)
        assert received == [m]

    def test_set_manifest_none_clears_and_publishes(self, tmp_path):
        """Passing None clears the manifest and notifies subscribers with
        None — the contract for project-close from the picker dropdown."""
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        s = AppState()
        s.manifest = Manifest(root=tmp_path)
        received = []
        s.subscribe("project_changed", lambda m: received.append(m))

        s.set_manifest(None)
        assert s.manifest is None
        assert received == [None]

    def test_project_changed_fires_after_field_assignment(self, tmp_path):
        """Subscribers reading ``state.manifest`` from inside their callback
        must see the NEW value, not the pre-swap one. This is the contract
        that lets a panel's refreshable rebuild against the new manifest
        directly in the event handler."""
        from hrfunc.gui.state import AppState
        from hrfunc.io.manifest import Manifest

        s = AppState()
        observed = []

        def reader(payload):
            observed.append(s.manifest)  # read live state, ignore payload

        s.subscribe("project_changed", reader)

        m = Manifest(root=tmp_path)
        s.set_manifest(m)
        assert observed == [m]


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


# ---------------------------------------------------------------------------
# Regression: ui.run kwargs must compose with the installed NiceGUI
# ---------------------------------------------------------------------------


class TestUiRunKwargs:
    """Sprint 2.1 shipped ``ui.run(show_welcome_message_on_startup=False)``
    — that kwarg doesn't exist on the real NiceGUI ``Config.__init__``.
    Tests using the User fixture intercept ``ui.run`` and never validate
    the kwargs against the real signature, so the bug shipped to v1.3.0
    and only surfaced when researchers actually launched ``hrfunc`` from
    the installed shortcut. This test introspects the live signature so
    a future kwarg drift breaks CI instead of breaking users."""

    def test_every_ui_run_kwarg_we_pass_is_valid(self):
        import inspect

        from nicegui import ui

        from hrfunc.gui import app

        # Pull the kwargs string out of _launch_gui's source. Cheap parser:
        # find the `ui.run(` call and pull every `keyword=...` token before
        # the closing paren. Resilient to formatting changes.
        source = inspect.getsource(app._launch_gui)
        marker = "ui.run("
        start = source.index(marker) + len(marker)
        # Match the closing paren by walking the source character-by-character
        # with a paren counter. Avoids regex pitfalls with nested tuples
        # like ``window_size=(1400, 900)``.
        depth = 1
        end = start
        while end < len(source) and depth > 0:
            ch = source[end]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            end += 1
        call_args = source[start:end - 1]

        # Extract keyword names (before the `=` on each comma-separated arg).
        kwarg_names = []
        depth = 0
        token = ""
        for ch in call_args + ",":
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            if ch == "," and depth == 0:
                bit = token.strip()
                if "=" in bit:
                    name = bit.split("=", 1)[0].strip()
                    if name and name.isidentifier():
                        kwarg_names.append(name)
                token = ""
            else:
                token += ch

        sig = inspect.signature(ui.run)
        valid = set(sig.parameters)
        unknown = [k for k in kwarg_names if k not in valid]
        assert not unknown, (
            f"_launch_gui passes ui.run kwargs that aren't in the installed "
            f"NiceGUI's Config.__init__ signature: {unknown}. "
            f"Check the NiceGUI changelog for renamed/removed parameters."
        )
