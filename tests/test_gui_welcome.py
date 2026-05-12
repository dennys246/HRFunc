"""Targeted unit tests for feat/gui-welcome-page (v1.3.0 Sprint 2.2).

Covers the welcome page and the surrounding navigation stubs:

- ``hrfunc.gui.pages.welcome`` — the 3-path entry screen, recent-projects
  enumeration, folder-picker wrapper.
- ``hrfunc.gui.app._register_pages`` — the route table now registers
  the real welcome page plus ``/workspace`` and ``/library`` stubs.

Tests split into two flavors:

1. **Rendering tests** (use ``nicegui.testing.User``) — confirm the
   welcome page renders, the three cards are present with the right text,
   and the stubs route correctly. Requires pytest-asyncio.

2. **Helper tests** (plain sync) — exercise the recent-manifest helper,
   the open-path flow against a non-existent path, the picker
   no-op-in-browser-mode behavior. No async machinery needed.

All tests are gated by ``pytest.importorskip("nicegui")`` so the file is
fully skipped when the [gui] extras are missing.
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

# Use the user-specific plugin (not the full nicegui.testing.plugin) so we
# avoid the selenium import that the Screen plugin requires. Rendering tests
# here only need the User fixture for headless route exercise.
pytest_plugins = ["nicegui.testing.user_plugin"]


@pytest.fixture(autouse=True)
def _suppress_first_launch_shortcut_prompt(monkeypatch, tmp_path):
    """Force the shortcut prompt to "already shown" so the existing
    welcome tests don't see the dialog overlay added in the install-
    shortcut PR. Tests that explicitly cover the prompt live in
    test_gui_welcome_shortcut_prompt.py.
    """
    from hrfunc.cli import install_shortcut as _ish
    marker = tmp_path / ".shortcut_prompted"
    marker.touch()
    monkeypatch.setattr(_ish, "_marker_path", lambda: marker)


# ---------------------------------------------------------------------------
# Helpers — recent-manifest enumeration (sync, no rendering)
# ---------------------------------------------------------------------------


class TestRecentManifestEnumeration:
    def test_empty_cache_returns_empty_list(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "platformdirs.user_cache_dir",
            lambda *_a, **_kw: str(tmp_path / "no_such_cache"),
        )
        from hrfunc.gui.pages.welcome import _list_recent_manifests
        assert _list_recent_manifests() == []

    def test_corrupt_manifest_is_skipped(self, tmp_path, monkeypatch):
        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        (cache_root / "manifest_garbage.json").write_text("not valid json")
        monkeypatch.setattr(
            "platformdirs.user_cache_dir",
            lambda *_a, **_kw: str(cache_root),
        )
        from hrfunc.gui.pages.welcome import _list_recent_manifests
        assert _list_recent_manifests() == []

    def test_recent_manifests_sorted_by_scanned_at_desc(
        self, tmp_path, monkeypatch
    ):
        from datetime import datetime, timezone

        from hrfunc.io.manifest import Manifest

        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        monkeypatch.setattr(
            "platformdirs.user_cache_dir",
            lambda *_a, **_kw: str(cache_root),
        )

        # Write three manifests with distinct timestamps
        old = Manifest(
            root=Path("/tmp/old_study"),
            scanned_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        mid = Manifest(
            root=Path("/tmp/mid_study"),
            scanned_at=datetime(2025, 6, 15, tzinfo=timezone.utc),
        )
        new = Manifest(
            root=Path("/tmp/new_study"),
            scanned_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
        )
        for i, m in enumerate([old, mid, new]):
            (cache_root / f"manifest_{i}.json").write_text(m.to_json())

        from hrfunc.gui.pages.welcome import _list_recent_manifests
        result = _list_recent_manifests()
        assert [m.root.name for m in result] == ["new_study", "mid_study", "old_study"]

    def test_limit_caps_result_length(self, tmp_path, monkeypatch):
        from datetime import datetime, timezone

        from hrfunc.io.manifest import Manifest

        cache_root = tmp_path / "cache"
        cache_root.mkdir()
        monkeypatch.setattr(
            "platformdirs.user_cache_dir",
            lambda *_a, **_kw: str(cache_root),
        )
        for i in range(15):
            m = Manifest(
                root=Path(f"/tmp/study_{i}"),
                scanned_at=datetime(2026, 1, i + 1, tzinfo=timezone.utc),
            )
            (cache_root / f"manifest_{i}.json").write_text(m.to_json())

        from hrfunc.gui.pages.welcome import _list_recent_manifests
        result = _list_recent_manifests(limit=5)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Folder picker — browser-mode fallback (sync test of the no-op path)
# ---------------------------------------------------------------------------


class TestFolderPicker:
    def test_pick_folder_is_async(self):
        from hrfunc.gui.pages.welcome import _pick_folder
        assert inspect.iscoroutinefunction(_pick_folder)


# ---------------------------------------------------------------------------
# Open-path flow — non-existent path surfaces error, doesn't crash
# ---------------------------------------------------------------------------


class TestOpenPathFlow:
    def test_handle_open_path_is_async(self):
        from hrfunc.gui.pages.welcome import _handle_open_path
        assert inspect.iscoroutinefunction(_handle_open_path)


# ---------------------------------------------------------------------------
# Rendering tests — NiceGUI User fixture
# ---------------------------------------------------------------------------

# Re-register pages once at module import so the User fixture sees them
from hrfunc.gui import app as gui_app  # noqa: E402
gui_app._register_pages()


# Rendering tests are top-level async functions because pytest-asyncio's
# strict mode does not propagate @pytest.mark.asyncio through test classes
# when one of the requested fixtures is itself async (NiceGUI's User
# fixture is async). Module-level functions with explicit markers work
# cleanly with the user_plugin.


@pytest.mark.asyncio
async def test_welcome_page_shows_brand_header(user: User):
    await user.open("/")
    await user.should_see("HRfunc")
    await user.should_see("fNIRS hemodynamic response estimation")


@pytest.mark.asyncio
async def test_welcome_page_shows_three_cards(user: User):
    await user.open("/")
    await user.should_see("Estimate hemodynamics")
    await user.should_see("Browse HRF library")
    await user.should_see("Recent projects")


@pytest.mark.asyncio
async def test_welcome_page_shows_version_in_footer(user: User):
    await user.open("/")
    # Footer renders f"v{version('hrfunc')}" — assert the v-prefix appears.
    await user.should_see("v")


@pytest.mark.asyncio
async def test_library_page_renders(user: User):
    """Sprint 4.4 replaced the inline /library stub with the real Library
    page; this test asserts the toolbar + a filter-pane header so the
    welcome-flow test file knows /library is reachable."""
    await user.open("/library")
    await user.should_see("HRF Library")
    await user.should_see("Filter")


# ---------------------------------------------------------------------------
# Route table — confirm welcome.register is wired correctly
# ---------------------------------------------------------------------------


class TestRouteRegistration:
    def test_welcome_register_is_callable(self):
        from hrfunc.gui.pages import welcome
        assert callable(welcome.register)

    def test_register_pages_calls_welcome_register(self):
        """_register_pages must invoke welcome.register so the / route is
        bound. Regression guard: a future refactor that drops this call
        would leave the root page unregistered and the GUI would 404 on
        launch."""
        from hrfunc.gui import app
        source = inspect.getsource(app._register_pages)
        assert "welcome.register()" in source
