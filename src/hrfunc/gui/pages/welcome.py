"""Welcome page — the 3-path entry screen.

Three user personas, three entry paths:

1. **Open my data** — folder/file picker → scan → workspace
   For users with their own fNIRS recordings who want to estimate or
   localize HRFs.

2. **Browse HRF library** — straight to library page
   For users without local data who want to explore the bundled
   literature-derived HRF databases (hbo_hrfs.json + hbr_hrfs.json,
   ~4MB of contextually-indexed HRFs).

3. **Recent projects** — list of previously-scanned folders
   For users picking up a previous session. Reads scan manifests from
   the XDG cache (written by ``scan_folder``).

Each card is a clickable button styled with the brand palette. The page is
rendered fresh on every visit (no caching of UI state) so navigation
between pages always shows the live state of ``state.preload_path`` and
the on-disk recent-projects cache.

Behavior:
- If ``state.preload_path`` is set (CLI arg ``hrfunc <path>``), the page
  auto-triggers the open-data flow with that path on first render.
- Otherwise the user clicks one of three cards. The choice routes via
  ``ui.navigate.to`` to ``/workspace`` (Sprint 2.3) or ``/library`` (Sprint 4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from nicegui import app, ui

from ..state import AppState, state as global_state
from ..theme import apply_theme, page_container
from ..workers import run_in_background
from ...io.manifest import Manifest
from ...io.scan import scan_folder

logger = logging.getLogger(__name__)


def register() -> None:
    """Register the welcome page handler at ``/``.

    Called once by ``app._register_pages()`` at startup. The decorator
    ensures NiceGUI knows about the route before ``ui.run`` starts the
    server.
    """

    @ui.page("/")
    async def welcome_page() -> None:
        await _render(global_state)


async def _render(state: AppState) -> None:
    """Render the welcome page against the given AppState.

    Split from the ``@ui.page`` handler so tests can call it with a fresh
    AppState instance without going through NiceGUI's route registration.
    """
    apply_theme()

    with page_container():
        _render_header()
        _render_cards(state)
        _render_footer()

    # First-launch shortcut prompt — show a one-time dialog asking the
    # user whether they want HRFunc added to their system menu
    # (Spotlight / Start menu / Activities). The XDG-cache marker means
    # this dialog appears at most once per machine.
    _maybe_show_shortcut_prompt()

    if state.preload_path is not None:
        path = state.preload_path
        state.preload_path = None  # consume so subsequent renders don't re-trigger
        await _handle_open_path(state, path)


def _maybe_show_shortcut_prompt() -> None:
    """Show the first-launch shortcut prompt if the user hasn't been asked yet.

    Skips silently if pyshortcuts isn't installed (e.g. user has the GUI
    extras half-installed) or if the marker file already exists. The
    "Yes" path runs the install on a background thread because some
    pywebview backends serialize file-system writes with UI events on
    Linux — keeping the dialog responsive.
    """
    try:
        from ...cli.install_shortcut import (
            install_shortcut,
            set_prompted,
            was_prompted,
        )
    except ImportError:
        logger.debug("install_shortcut module unavailable; skipping prompt")
        return

    if was_prompted():
        return

    with ui.dialog() as dialog, ui.card().classes("max-w-md"):
        ui.label("Add HRFunc to your system menu?").classes(
            "text-lg font-semibold"
        )
        ui.label(
            "We can install a launcher so HRFunc appears in your system's "
            "app search (Spotlight on macOS, Start menu on Windows, "
            "Activities on Linux). You'll be able to open it like any "
            "other desktop app — no terminal needed."
        ).classes("text-sm opacity-80")
        ui.label(
            "You can change this later with `hrfunc install-shortcut` or "
            "`hrfunc uninstall-shortcut`."
        ).classes("text-xs opacity-60 italic")

        def _yes() -> None:
            result = install_shortcut()
            set_prompted()
            dialog.close()
            ui.notify(
                result.message,
                type="positive" if result.ok else "negative",
            )

        def _later() -> None:
            # Don't write the marker — re-prompt on next launch.
            dialog.close()

        def _never() -> None:
            set_prompted()
            dialog.close()
            ui.notify(
                "Got it. Run `hrfunc install-shortcut` later if you "
                "change your mind.",
                type="info",
            )

        with ui.row().classes("w-full justify-end gap-2 mt-2"):
            ui.button("Don't ask again", on_click=_never).props("flat")
            ui.button("Not now", on_click=_later).props("flat")
            ui.button("Yes, add it", on_click=_yes).props("color=primary")

    dialog.open()


def _render_header() -> None:
    with ui.column().classes("w-full items-center mt-12 mb-8 gap-2"):
        ui.label("HRFunc").classes("text-6xl font-bold tracking-tight")
        ui.label("fNIRS hemodynamic response estimation").classes(
            "text-xl opacity-70"
        )


def _render_cards(state: AppState) -> None:
    with ui.grid(columns=3).classes("w-full gap-6"):
        _card(
            label="Open my data",
            description="Pick a folder of fNIRS scans. Estimate or localize HRFs against the literature database.",
            icon="folder_open",
            on_click=lambda: _handle_open_my_data(state),
        )
        _card(
            label="Browse HRF library",
            description="Explore the bundled literature-derived HRFs. No data of your own required.",
            icon="library_books",
            on_click=_handle_browse_library,
        )
        _card(
            label="Recent projects",
            description="Reload a previously-scanned folder from your cache.",
            icon="history",
            on_click=lambda: _show_recent_projects(state),
        )


def _card(label: str, description: str, icon: str, on_click) -> None:
    """One welcome-screen card. Buttons styled as Quasar cards for visual weight."""
    with ui.card().classes(
        "p-6 cursor-pointer hover:bg-slate-800 transition-colors h-full"
    ).on("click", on_click):
        with ui.column().classes("gap-3 h-full"):
            ui.icon(icon, size="3rem").classes("text-primary")
            ui.label(label).classes("text-2xl font-semibold")
            ui.label(description).classes("text-sm opacity-70 leading-relaxed")


def _render_footer() -> None:
    from importlib.metadata import version

    try:
        v = version("hrfunc")
    except Exception:
        v = "unknown"
    with ui.row().classes("w-full justify-center mt-12 opacity-50"):
        ui.label(f"v{v}").classes("text-xs")


# ---------------------------------------------------------------------------
# Card click handlers
# ---------------------------------------------------------------------------


async def _handle_open_my_data(state: AppState) -> None:
    """Open a folder picker, then scan + navigate.

    In native (pywebview) mode, uses the OS folder dialog. In browser mode
    or when the native window is not yet ready, no-ops with a notification
    — this matches the GUI's "native-only" intended deployment.
    """
    path = await _pick_folder()
    if path is None:
        return
    await _handle_open_path(state, path)


async def _handle_open_path(state: AppState, path: Path) -> None:
    """Scan the given folder and navigate to /workspace when done.

    Used both by the folder picker and by the CLI preload flow. Errors
    surface via ``state.last_error`` and a UI notification; navigation
    only happens on success.
    """
    async def _on_done(result: Optional[Manifest]) -> None:
        if result is None:
            ui.notify(f"Scan failed: {state.last_error}", type="negative")
            return
        state.manifest = result
        ui.navigate.to("/workspace")

    if not path.exists():
        ui.notify(f"Path does not exist: {path}", type="negative")
        return
    if path.is_file():
        # Single-file open — wrap into a minimal manifest via scan of parent
        # for now; Sprint 3 will refine single-file flow.
        path = path.parent

    await run_in_background(state, scan_folder, path, on_done=_on_done)


def _handle_browse_library() -> None:
    """Navigate to the library page (Sprint 4 implements; 2.2 has a stub)."""
    ui.navigate.to("/library")


def _show_recent_projects(state: AppState) -> None:
    """Show a dialog listing recently-scanned manifests from the XDG cache.

    Each item shows the root path and the scan timestamp. Clicking an item
    loads that manifest and navigates to /workspace.
    """
    recent = _list_recent_manifests()
    with ui.dialog() as dialog, ui.card().classes("min-w-96"):
        ui.label("Recent projects").classes("text-xl font-semibold mb-2")
        if not recent:
            ui.label(
                "No recent projects yet. Open a folder to get started."
            ).classes("opacity-70")
        else:
            for manifest in recent:
                with ui.row().classes(
                    "w-full items-center justify-between gap-2 p-2 "
                    "hover:bg-slate-800 cursor-pointer rounded"
                ).on("click", lambda m=manifest: _load_recent(state, m, dialog)):
                    with ui.column().classes("gap-0"):
                        ui.label(str(manifest.root)).classes("font-mono text-sm")
                        ui.label(
                            f"{len(manifest.scans)} scans · "
                            f"{manifest.scanned_at.strftime('%Y-%m-%d %H:%M')}"
                        ).classes("text-xs opacity-60")
        with ui.row().classes("w-full justify-end mt-4"):
            ui.button("Close", on_click=dialog.close).props("flat")
    dialog.open()


def _load_recent(state: AppState, manifest: Manifest, dialog) -> None:
    state.manifest = manifest
    dialog.close()
    ui.navigate.to("/workspace")


# ---------------------------------------------------------------------------
# Helpers — folder picker + recent-manifest enumeration
# ---------------------------------------------------------------------------


async def _pick_folder() -> Optional[Path]:
    """Open the OS folder picker via pywebview.

    Returns the chosen Path, or None if the user cancelled or native mode
    is unavailable. The pywebview API blocks the calling thread, so we run
    it in an executor to keep the asyncio loop free.
    """
    try:
        import webview
    except ImportError:
        ui.notify("pywebview not installed", type="negative")
        return None

    window = getattr(app, "native", None) and app.native.main_window
    if window is None:
        ui.notify(
            "Folder picker requires native window mode. "
            "Launch with `hrfunc` (not via browser).",
            type="warning",
        )
        return None

    import asyncio

    loop = asyncio.get_event_loop()
    paths = await loop.run_in_executor(
        None,
        lambda: window.create_file_dialog(webview.FOLDER_DIALOG),
    )
    if not paths:
        return None
    return Path(paths[0])


def _list_recent_manifests(limit: int = 10) -> List[Manifest]:
    """Enumerate cached manifests from the XDG cache directory.

    Reads every ``manifest_*.json`` file in ``platformdirs.user_cache_dir``,
    deserializes via ``Manifest.from_json``, sorts by ``scanned_at`` descending,
    and returns the top ``limit``. Corrupt or schema-mismatched files are
    silently skipped — same fail-safe contract as the scanner cache itself.
    """
    try:
        import platformdirs
    except ImportError:
        return []

    cache_dir = Path(platformdirs.user_cache_dir("hrfunc"))
    if not cache_dir.exists():
        return []

    manifests: List[Manifest] = []
    for cache_file in cache_dir.glob("manifest_*.json"):
        try:
            m = Manifest.from_json(cache_file.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — fail-safe per scan.py contract
            logger.debug("Skipping recent-manifest %s: %s", cache_file, exc)
            continue
        manifests.append(m)

    manifests.sort(key=lambda m: m.scanned_at, reverse=True)
    return manifests[:limit]
