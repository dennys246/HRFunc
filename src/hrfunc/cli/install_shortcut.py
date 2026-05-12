"""Cross-platform "install HRfunc to system menu" shortcut helpers.

The HRfunc GUI is designed for fNIRS researchers who don't otherwise live
in a terminal. Requiring them to type ``hrfunc`` every time they want to
analyze a scan defeats the point of shipping a GUI — so we install a
system-level launcher (Spotlight on macOS, Start menu on Windows,
Activities on Linux) that points at the installed ``hrfunc`` console
script.

The heavy lifting is delegated to ``pyshortcuts`` (a small cross-platform
shortcut-builder library that handles the per-OS file formats and icon
conversions). HRfunc only contributes:

- ``install_shortcut`` — the high-level entry point. Resolves the icon
  asset bundled inside ``hrfunc.assets``, dispatches to
  ``pyshortcuts.make_shortcut``, and returns a ``InstallResult`` so
  callers can surface success / failure consistently.
- ``uninstall_shortcut`` — best-effort removal by walking the standard
  shortcut directories and deleting matching files.
- First-launch prompt bookkeeping (``was_prompted`` / ``set_prompted``)
  via an XDG-cache marker file so the welcome page only asks once.

The bundled icon lives at ``src/hrfunc/assets/executable_icon.png``
(500×500 RGBA). ``pyshortcuts`` converts to ``.icns`` (macOS) and
``.ico`` (Windows) automatically; Linux uses the PNG directly.
"""

from __future__ import annotations

import logging
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# The user-facing name that appears in Spotlight / Start menu / Activities.
SHORTCUT_NAME = "HRfunc"
SHORTCUT_DESCRIPTION = (
    "fNIRS hemodynamic response function estimation "
    "and neural activity recovery"
)


@dataclass
class InstallResult:
    """Outcome of an install_shortcut call.

    ``ok`` is True only when the underlying pyshortcuts call succeeded
    AND the resulting file exists on disk. ``locations`` lists the
    shortcut files we believe were written (for user feedback / later
    uninstall).  ``message`` is a human-readable summary suitable for a
    toast notification.
    """

    ok: bool
    locations: List[Path]
    message: str


@dataclass
class UninstallResult:
    ok: bool
    removed: List[Path]
    message: str


# ---------------------------------------------------------------------------
# Icon resolution
# ---------------------------------------------------------------------------


def icon_path() -> Optional[Path]:
    """Return the absolute path to the bundled executable icon.

    The icon ships inside the installed ``hrfunc`` package at
    ``hrfunc/assets/executable_icon.png``. Returns None if the file is
    somehow missing (e.g. a broken install) so callers can fall back to
    pyshortcuts' generic default icon rather than crashing.
    """
    try:
        from importlib import resources

        # Python 3.9+: files() returns a Traversable. We need a real
        # filesystem path because pyshortcuts opens the file directly
        # (no Traversable support).
        ref = resources.files("hrfunc.assets") / "executable_icon.png"
        path = Path(str(ref))
        if path.exists():
            return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("icon resolution via importlib.resources failed: %s", exc)

    # Fallback: locate via the package's __file__.
    try:
        import hrfunc

        pkg_dir = Path(hrfunc.__file__).parent
        candidate = pkg_dir / "assets" / "executable_icon.png"
        if candidate.exists():
            return candidate
    except Exception as exc:  # noqa: BLE001
        logger.debug("icon resolution via __file__ failed: %s", exc)

    logger.warning(
        "executable_icon.png not found in installed package; the "
        "shortcut will use pyshortcuts' default icon."
    )
    return None


# ---------------------------------------------------------------------------
# Install / uninstall
# ---------------------------------------------------------------------------


def install_shortcut(
    *,
    desktop: Optional[bool] = None,
    startmenu: Optional[bool] = None,
) -> InstallResult:
    """Install a system-level launcher pointing at the ``hrfunc`` GUI.

    Per-OS placement defaults (when callers pass None):

    - **macOS**: ``desktop=True, startmenu=False``. pyshortcuts'
      darwin.py exits early when ``desktop=False`` (it has no separate
      start-menu code path), so we must let it write to Desktop. We
      then post-move the resulting ``.app`` bundle into
      ``~/Applications`` so it's Spotlight-indexed at the canonical
      location and doesn't clutter the user's desktop.
    - **Windows**: ``desktop=False, startmenu=True``. Writes to the
      Start menu Programs folder — searchable, no desktop clutter.
    - **Linux**: ``desktop=False, startmenu=True``. Writes to
      ``~/.local/share/applications/`` — picked up by Activities /
      app menus on GNOME / KDE / etc.

    Returns an ``InstallResult``. Callers should surface ``message`` to
    the user (toast on the welcome page, stderr line in the CLI).
    """
    try:
        import pyshortcuts
    except ImportError:
        return InstallResult(
            ok=False,
            locations=[],
            message=(
                "pyshortcuts is not installed. Install the GUI extras: "
                "`pip install hrfunc[gui]`."
            ),
        )

    target = _resolve_target_script()
    if target is None:
        return InstallResult(
            ok=False,
            locations=[],
            message=(
                "Could not find the installed `hrfunc` console script. "
                "Reinstall HRfunc with `pip install --force-reinstall "
                "hrfunc[gui]` and try again."
            ),
        )

    icon = icon_path()
    icon_arg = str(icon) if icon is not None else None

    # Per-OS placement defaults — see docstring for the rationale.
    is_mac = platform.system() == "Darwin"
    if desktop is None:
        desktop = True if is_mac else False
    if startmenu is None:
        startmenu = False if is_mac else True

    try:
        pyshortcuts.make_shortcut(
            script=target,
            name=SHORTCUT_NAME,
            description=SHORTCUT_DESCRIPTION,
            icon=icon_arg,
            terminal=False,
            desktop=desktop,
            startmenu=startmenu,
            noexe=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("pyshortcuts.make_shortcut failed: %s", exc)
        return InstallResult(
            ok=False,
            locations=[],
            message=f"Shortcut install failed: {type(exc).__name__}: {exc}",
        )

    # On macOS, move the .app bundle from Desktop into ~/Applications
    # so Spotlight finds it at the canonical location and the desktop
    # stays clean.
    if is_mac:
        moved = _move_mac_app_to_applications()
        if moved is not None:
            logger.info("moved HRfunc.app to %s", moved)

    written = _discover_shortcuts()
    if not written:
        return InstallResult(
            ok=False,
            locations=[],
            message=(
                "pyshortcuts ran but no shortcut file was found. "
                "This can happen on managed Linux distros without a "
                "writable Applications / Activities folder."
            ),
        )

    where = _describe_locations(written)
    return InstallResult(
        ok=True,
        locations=written,
        message=f"HRfunc shortcut installed: {where}",
    )


def _move_mac_app_to_applications() -> Optional[Path]:
    """Move ``~/Desktop/HRfunc.app`` to ``~/Applications/HRfunc.app``.

    pyshortcuts' macOS path always writes the .app bundle to Desktop.
    Spotlight will index it there (the home directory is included in
    Spotlight's default index), but the canonical location for user-
    installed apps is ``~/Applications``. Moving keeps the user's
    desktop tidy without losing discoverability.

    Returns the new path on a successful move, None if no .app was
    found on Desktop or the move failed (we log + continue rather than
    failing the install).
    """
    home = Path.home()
    source = home / "Desktop" / f"{SHORTCUT_NAME}.app"
    if not source.exists():
        return None
    dest_dir = home / "Applications"
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not create %s: %s", dest_dir, exc)
        return None
    dest = dest_dir / source.name
    try:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(source), str(dest))
        return dest
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not move %s → %s: %s", source, dest, exc)
        return None


def uninstall_shortcut() -> UninstallResult:
    """Delete any HRfunc shortcut files we can find.

    Walks the standard pyshortcuts output directories (Desktop +
    Applications / Start menu / Activities for the current OS) and
    deletes files whose stem matches ``SHORTCUT_NAME``. Returns an
    ``UninstallResult`` describing what was removed.
    """
    candidates = _discover_shortcuts()
    if not candidates:
        return UninstallResult(
            ok=True,
            removed=[],
            message="No HRfunc shortcut found to remove.",
        )

    removed: List[Path] = []
    failed: List[Path] = []
    for path in candidates:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to remove %s: %s", path, exc)
            failed.append(path)

    if failed and not removed:
        return UninstallResult(
            ok=False,
            removed=[],
            message=(
                f"Could not remove HRfunc shortcuts: "
                f"{', '.join(str(p) for p in failed)}"
            ),
        )
    return UninstallResult(
        ok=True,
        removed=removed,
        message=f"Removed {len(removed)} HRfunc shortcut(s).",
    )


# ---------------------------------------------------------------------------
# First-launch prompt bookkeeping
# ---------------------------------------------------------------------------


def _marker_path() -> Path:
    """Path to the XDG-cache marker that records 'we already asked'.

    Returns ``<cache>/hrfunc/.shortcut_prompted``. The directory is
    created on demand by ``set_prompted``; ``was_prompted`` only reads.
    """
    try:
        import platformdirs

        cache_dir = Path(platformdirs.user_cache_dir("hrfunc"))
    except ImportError:
        # platformdirs is a runtime dep; this branch is defensive.
        cache_dir = Path.home() / ".cache" / "hrfunc"
    return cache_dir / ".shortcut_prompted"


def was_prompted() -> bool:
    """True if the welcome-page first-launch prompt has been shown before."""
    return _marker_path().exists()


def set_prompted() -> None:
    """Mark that the first-launch prompt has been shown (don't ask again)."""
    marker = _marker_path()
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to write shortcut marker %s: %s", marker, exc)


# ---------------------------------------------------------------------------
# Internals — target script + shortcut-file discovery
# ---------------------------------------------------------------------------


def _resolve_target_script() -> Optional[str]:
    """Locate the installed ``hrfunc`` console script in PATH.

    pyshortcuts wraps the given path in an OS-specific launcher
    (``.command`` on Mac, ``.lnk`` on Windows, ``.desktop`` on Linux)
    that runs it directly when clicked. We pass ``noexe=True`` to
    pyshortcuts so it doesn't prepend a Python interpreter — the
    ``hrfunc`` console script is already a Python entry-point wrapper.
    """
    path = shutil.which("hrfunc")
    if path is None:
        return None
    return path


def _discover_shortcuts() -> List[Path]:
    """Return existing HRfunc shortcut files in the standard locations.

    Walks pyshortcuts' get_folders() output plus a hardcoded
    ``~/Applications`` directory on macOS (where we post-move .app
    bundles — pyshortcuts.get_folders() doesn't include that path
    because pyshortcuts itself writes only to Desktop on macOS).

    Used both as a post-install verification (did pyshortcuts actually
    write the file?) and as the basis for uninstall.
    """
    try:
        import pyshortcuts
    except ImportError:
        return []

    folders = pyshortcuts.get_folders()
    candidates: List[Path] = []

    # Different OSes use different extensions; check both with-extension
    # and bare-stem matches.
    suffixes = [
        ".lnk",   # Windows
        ".desktop",  # Linux
        ".app",  # macOS application bundle
        ".command",  # macOS command file
    ]

    search_dirs: List[Path] = []
    for folder_attr in ("desktop", "startmenu"):
        folder = getattr(folders, folder_attr, None)
        if folder:  # treat empty string as absent (pyshortcuts darwin returns "")
            search_dirs.append(Path(folder))

    # macOS-specific: post-install we move the .app to ~/Applications.
    # pyshortcuts.get_folders() doesn't list that folder, so add it.
    if platform.system() == "Darwin":
        search_dirs.append(Path.home() / "Applications")

    for folder_path in search_dirs:
        if not folder_path.exists():
            continue
        for suffix in suffixes:
            candidate = folder_path / f"{SHORTCUT_NAME}{suffix}"
            if candidate.exists():
                candidates.append(candidate)

    return candidates


def _describe_locations(paths: List[Path]) -> str:
    """Human-readable summary of where shortcuts were installed."""
    if not paths:
        return "no files"
    system = platform.system()
    if system == "Darwin":
        return f"available in Spotlight as '{SHORTCUT_NAME}'"
    if system == "Windows":
        return f"available in the Start menu as '{SHORTCUT_NAME}'"
    if system == "Linux":
        return f"available in Activities / Applications as '{SHORTCUT_NAME}'"
    return ", ".join(str(p) for p in paths)
