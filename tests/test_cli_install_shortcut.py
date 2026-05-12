"""Targeted unit tests for feat/cli-install-shortcut (v1.3.0 polish PR).

Covers:

- ``icon_path`` resolves to the bundled PNG in src/hrfunc/assets/.
- ``InstallResult`` / ``UninstallResult`` dataclass shapes.
- First-launch marker semantics (``was_prompted`` / ``set_prompted``
  honoring a tmp_path-redirected cache directory).
- ``install_shortcut`` happy-path: when pyshortcuts succeeds + a file
  appears in a known location, returns ok=True with a sensible message.
- ``install_shortcut`` failure paths: pyshortcuts raises, no shortcut
  written, no console-script in PATH.
- ``uninstall_shortcut`` removes matching files and reports the count.
- ``gui/app.py`` subcommand prefilter: ``hrfunc install-shortcut`` and
  ``hrfunc uninstall-shortcut`` dispatch to the right handlers without
  touching the GUI launch path.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hrfunc.cli import install_shortcut as ish
from hrfunc.gui import app as gui_app


# ---------------------------------------------------------------------------
# Icon resolution
# ---------------------------------------------------------------------------


class TestIconPath:
    def test_resolves_to_bundled_png(self):
        path = ish.icon_path()
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


class TestMacIconPath:
    """``mac_icon_path`` converts our bundled PNG to a real macOS ICNS
    file. pyshortcuts copies whatever we hand it verbatim into the .app
    bundle's Resources directory; without conversion, Finder shows the
    generic app icon because the file's magic bytes don't match the
    IconFamily format despite the .icns extension."""

    def test_produces_real_icns_file(self, tmp_path):
        """The converted file must have the macOS IconFamily magic bytes
        (``icns`` at offset 0), not PNG magic bytes."""
        result = ish.mac_icon_path()
        assert result is not None
        assert result.exists()
        assert result.suffix == ".icns"
        # Validate magic bytes — the IconFamily format starts with `icns`.
        with open(result, "rb") as f:
            magic = f.read(4)
        assert magic == b"icns", (
            f"expected macOS IconFamily magic 'icns', got {magic!r}; "
            f"this means the file is not a real .icns (likely a PNG with "
            f"a renamed extension) and macOS Finder will show the generic "
            f"app icon."
        )

    def test_pillow_missing_returns_none(self, monkeypatch):
        """Without Pillow, the helper degrades gracefully (caller falls
        back to the PNG)."""
        import sys
        monkeypatch.setitem(sys.modules, "PIL", None)
        monkeypatch.setitem(sys.modules, "PIL.Image", None)
        result = ish.mac_icon_path()
        assert result is None

    def test_no_source_png_returns_none(self, monkeypatch):
        """If the source PNG is somehow missing, the helper can't
        convert anything and returns None."""
        monkeypatch.setattr(ish, "icon_path", lambda: None)
        result = ish.mac_icon_path()
        assert result is None


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


class TestResultDataclasses:
    def test_install_result_shape(self):
        r = ish.InstallResult(ok=True, locations=[Path("/x")], message="ok")
        assert r.ok is True
        assert r.locations == [Path("/x")]
        assert r.message == "ok"

    def test_uninstall_result_shape(self):
        r = ish.UninstallResult(ok=False, removed=[], message="nope")
        assert r.ok is False
        assert r.removed == []


# ---------------------------------------------------------------------------
# First-launch marker
# ---------------------------------------------------------------------------


class TestPromptMarker:
    def test_was_prompted_false_when_marker_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            ish, "_marker_path", lambda: tmp_path / ".shortcut_prompted"
        )
        assert ish.was_prompted() is False

    def test_set_prompted_writes_marker(self, tmp_path, monkeypatch):
        marker = tmp_path / "nested" / ".shortcut_prompted"
        monkeypatch.setattr(ish, "_marker_path", lambda: marker)
        ish.set_prompted()
        assert marker.exists()

    def test_set_prompted_is_idempotent(self, tmp_path, monkeypatch):
        marker = tmp_path / ".shortcut_prompted"
        monkeypatch.setattr(ish, "_marker_path", lambda: marker)
        ish.set_prompted()
        first_mtime = marker.stat().st_mtime
        ish.set_prompted()  # no error on second call
        # Path.touch is idempotent; just verify the file is still there.
        assert marker.exists()


# ---------------------------------------------------------------------------
# install_shortcut — failure paths (easier to test deterministically)
# ---------------------------------------------------------------------------


class TestInstallShortcutFailures:
    def test_pyshortcuts_missing_reports_friendly_message(
        self, monkeypatch
    ):
        # Stub the import so the function takes the ImportError branch.
        monkeypatch.setitem(__import__("sys").modules, "pyshortcuts", None)
        result = ish.install_shortcut()
        assert result.ok is False
        assert "pyshortcuts" in result.message

    def test_target_script_missing_reports_reinstall(self, monkeypatch):
        monkeypatch.setattr(ish, "_resolve_target_script", lambda: None)
        result = ish.install_shortcut()
        assert result.ok is False
        assert "console script" in result.message
        assert "force-reinstall" in result.message

    def test_pyshortcuts_raises_returns_friendly_error(self, monkeypatch):
        monkeypatch.setattr(
            ish, "_resolve_target_script", lambda: "/usr/local/bin/hrfunc"
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated pyshortcuts crash")

        import pyshortcuts
        monkeypatch.setattr(pyshortcuts, "make_shortcut", _boom)
        result = ish.install_shortcut()
        assert result.ok is False
        assert "Shortcut install failed" in result.message
        assert "simulated pyshortcuts crash" in result.message

    def test_make_shortcut_succeeds_but_no_file_found(self, monkeypatch):
        """If pyshortcuts thinks it worked but we can't find any output file,
        report a clear "no files found" message rather than silently
        succeeding."""
        monkeypatch.setattr(
            ish, "_resolve_target_script", lambda: "/usr/local/bin/hrfunc"
        )
        import pyshortcuts
        monkeypatch.setattr(pyshortcuts, "make_shortcut", lambda **kw: None)
        monkeypatch.setattr(ish, "_discover_shortcuts", lambda: [])

        result = ish.install_shortcut()
        assert result.ok is False
        assert "no shortcut file was found" in result.message


# ---------------------------------------------------------------------------
# install_shortcut — happy path
# ---------------------------------------------------------------------------


class TestInstallShortcutSuccess:
    def test_returns_ok_when_pyshortcuts_writes_a_file(self, monkeypatch, tmp_path):
        """Stub make_shortcut to "write" a file we can verify."""
        monkeypatch.setattr(
            ish, "_resolve_target_script", lambda: "/usr/local/bin/hrfunc"
        )
        # Force the non-macOS path for cross-platform behavior testing
        monkeypatch.setattr(ish.platform, "system", lambda: "Linux")

        fake_path = tmp_path / "HRfunc.desktop"
        fake_path.write_text("[Desktop Entry]\n")

        called_with = {}

        def _fake_make_shortcut(**kwargs):
            called_with.update(kwargs)

        import pyshortcuts
        monkeypatch.setattr(pyshortcuts, "make_shortcut", _fake_make_shortcut)
        monkeypatch.setattr(ish, "_discover_shortcuts", lambda: [fake_path])

        result = ish.install_shortcut()
        assert result.ok is True
        assert result.locations == [fake_path]
        # Verify the kwargs pyshortcuts was called with
        assert called_with["name"] == ish.SHORTCUT_NAME
        assert called_with["terminal"] is False
        # Linux default: startmenu=True, desktop=False
        assert called_with["startmenu"] is True
        assert called_with["desktop"] is False
        # Icon arg should be the bundled icon path (string form)
        assert called_with["icon"].endswith("executable_icon.png")


class TestInstallShortcutMacOS:
    """pyshortcuts.darwin returns None when desktop=False (no separate
    start-menu path exists on macOS). Our wrapper handles this by passing
    desktop=True on macOS and then post-moving the .app to ~/Applications
    for Spotlight discovery + a tidy desktop."""

    def test_macos_passes_desktop_true_to_pyshortcuts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ish.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(
            ish, "_resolve_target_script", lambda: "/usr/local/bin/hrfunc"
        )
        monkeypatch.setattr(ish, "_move_mac_app_to_applications", lambda: None)

        called_with = {}
        import pyshortcuts
        monkeypatch.setattr(
            pyshortcuts, "make_shortcut",
            lambda **kw: called_with.update(kw),
        )
        monkeypatch.setattr(
            ish, "_discover_shortcuts",
            lambda: [tmp_path / "fake.app"],
        )
        (tmp_path / "fake.app").mkdir()

        ish.install_shortcut()
        assert called_with["desktop"] is True
        assert called_with["startmenu"] is False

    def test_move_mac_app_to_applications(self, monkeypatch, tmp_path):
        """The post-move helper relocates Desktop/HRfunc.app → Applications/HRfunc.app."""
        # Fake home directory layout
        home = tmp_path
        desktop = home / "Desktop"
        desktop.mkdir()
        app_bundle = desktop / f"{ish.SHORTCUT_NAME}.app"
        app_bundle.mkdir()
        (app_bundle / "Info.plist").write_text("<plist/>")

        monkeypatch.setattr(ish.Path, "home", classmethod(lambda cls: home))

        result = ish._move_mac_app_to_applications()
        assert result is not None
        assert result == home / "Applications" / f"{ish.SHORTCUT_NAME}.app"
        assert result.exists()
        assert (result / "Info.plist").exists()
        # Source removed
        assert not app_bundle.exists()

    def test_move_mac_app_no_op_when_source_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ish.Path, "home", classmethod(lambda cls: tmp_path))
        # No Desktop/HRfunc.app to start with
        result = ish._move_mac_app_to_applications()
        assert result is None

    def test_move_mac_app_overwrites_existing_destination(
        self, monkeypatch, tmp_path
    ):
        home = tmp_path
        (home / "Desktop").mkdir()
        new_bundle = home / "Desktop" / f"{ish.SHORTCUT_NAME}.app"
        new_bundle.mkdir()
        (new_bundle / "Info.plist").write_text("new")

        # Pre-existing destination
        (home / "Applications").mkdir()
        old_dest = home / "Applications" / f"{ish.SHORTCUT_NAME}.app"
        old_dest.mkdir()
        (old_dest / "Info.plist").write_text("old")

        monkeypatch.setattr(ish.Path, "home", classmethod(lambda cls: home))
        result = ish._move_mac_app_to_applications()
        assert result == old_dest
        # New content replaced the old
        assert (result / "Info.plist").read_text() == "new"


class TestDiscoverShortcutsMacOS:
    """_discover_shortcuts walks ~/Applications on macOS so uninstall
    can find .app bundles we post-moved there."""

    def test_includes_applications_folder_on_mac(self, monkeypatch, tmp_path):
        home = tmp_path
        apps = home / "Applications"
        apps.mkdir()
        (apps / f"{ish.SHORTCUT_NAME}.app").mkdir()

        monkeypatch.setattr(ish.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(ish.Path, "home", classmethod(lambda cls: home))

        # Stub pyshortcuts.get_folders to return empty for desktop/startmenu
        import pyshortcuts
        from collections import namedtuple
        UserFolders = namedtuple("UserFolders", ("home", "desktop", "startmenu"))
        monkeypatch.setattr(
            pyshortcuts, "get_folders",
            lambda: UserFolders(str(home), "", ""),
        )

        found = ish._discover_shortcuts()
        # Should find the .app in ~/Applications
        assert len(found) == 1
        assert found[0].name == f"{ish.SHORTCUT_NAME}.app"


# ---------------------------------------------------------------------------
# uninstall_shortcut
# ---------------------------------------------------------------------------


class TestUninstallShortcut:
    def test_no_files_found_returns_ok_with_zero_removed(self, monkeypatch):
        monkeypatch.setattr(ish, "_discover_shortcuts", lambda: [])
        result = ish.uninstall_shortcut()
        assert result.ok is True
        assert result.removed == []
        assert "No HRfunc shortcut" in result.message

    def test_removes_files_and_reports_count(self, monkeypatch, tmp_path):
        a = tmp_path / "HRfunc.lnk"
        b = tmp_path / "HRfunc.desktop"
        a.write_text("fake shortcut a")
        b.write_text("fake shortcut b")
        monkeypatch.setattr(ish, "_discover_shortcuts", lambda: [a, b])

        result = ish.uninstall_shortcut()
        assert result.ok is True
        assert set(result.removed) == {a, b}
        assert not a.exists()
        assert not b.exists()
        assert "2 HRfunc shortcut(s)" in result.message

    def test_removes_directory_bundle(self, monkeypatch, tmp_path):
        """macOS .app bundles are directories; verify we use rmtree."""
        bundle = tmp_path / "HRfunc.app"
        bundle.mkdir()
        (bundle / "Contents").mkdir()
        (bundle / "Contents" / "Info.plist").write_text("<plist/>")
        monkeypatch.setattr(ish, "_discover_shortcuts", lambda: [bundle])

        result = ish.uninstall_shortcut()
        assert result.ok is True
        assert not bundle.exists()


# ---------------------------------------------------------------------------
# CLI subcommand prefilter (gui/app.py)
# ---------------------------------------------------------------------------


class TestCliSubcommandDispatch:
    def test_install_shortcut_subcommand_dispatches(self, monkeypatch, capsys):
        called = {}

        def _fake_install():
            called["install"] = True
            return ish.InstallResult(
                ok=True, locations=[Path("/x")], message="installed!"
            )

        # Patch the function the CLI imports lazily
        monkeypatch.setattr(ish, "install_shortcut", _fake_install)
        monkeypatch.setattr(ish, "set_prompted", lambda: None)

        rc = gui_app.main(["install-shortcut"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "installed!" in out
        assert called == {"install": True}

    def test_install_shortcut_subcommand_passes_through_marker(
        self, monkeypatch
    ):
        """Manual install should set the prompted marker so the welcome
        page doesn't ask again."""
        monkeypatch.setattr(
            ish, "install_shortcut",
            lambda: ish.InstallResult(
                ok=True, locations=[Path("/x")], message="ok"
            ),
        )
        marker_calls = []
        monkeypatch.setattr(
            ish, "set_prompted", lambda: marker_calls.append(1)
        )
        gui_app.main(["install-shortcut"])
        assert marker_calls == [1]

    def test_install_shortcut_failure_returns_nonzero(self, monkeypatch):
        monkeypatch.setattr(
            ish, "install_shortcut",
            lambda: ish.InstallResult(
                ok=False, locations=[], message="boom"
            ),
        )
        monkeypatch.setattr(ish, "set_prompted", lambda: None)
        rc = gui_app.main(["install-shortcut"])
        assert rc == 1

    def test_uninstall_shortcut_subcommand_dispatches(self, monkeypatch, capsys):
        monkeypatch.setattr(
            ish, "uninstall_shortcut",
            lambda: ish.UninstallResult(
                ok=True, removed=[Path("/x")], message="removed 1"
            ),
        )
        rc = gui_app.main(["uninstall-shortcut"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "removed 1" in out

    def test_unexpected_args_to_install_returns_2(self, capsys):
        rc = gui_app.main(["install-shortcut", "--surprise"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "unexpected" in err

    def test_help_alias_forwards_to_argparse(self):
        # argparse exits via SystemExit on --help. Verify we route there.
        with pytest.raises(SystemExit) as exc_info:
            gui_app.main(["help"])
        # argparse exits 0 for --help
        assert exc_info.value.code == 0

    def test_bare_call_does_not_match_subcommand_names(self, monkeypatch):
        """A path argument that happens to start with a non-subcommand
        word must fall through to GUI launch, not trip the prefilter."""
        called = {}

        def _fake_launch(argv):
            called["argv"] = argv
            return 0

        monkeypatch.setattr(gui_app, "_launch_gui", _fake_launch)
        gui_app.main(["/some/dataset/path"])
        # The path is forwarded unchanged to _launch_gui
        assert called["argv"] == ["/some/dataset/path"]

    def test_bare_call_with_no_args_launches_gui(self, monkeypatch):
        called = {}

        def _fake_launch(argv):
            called["argv"] = argv
            return 0

        monkeypatch.setattr(gui_app, "_launch_gui", _fake_launch)
        gui_app.main([])
        assert called["argv"] == []
