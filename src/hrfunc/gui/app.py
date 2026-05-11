"""HRFunc GUI entry point.

`hrfunc` (the CLI command installed by ``pip install hrfunc[gui]``) calls
``main()``, which:

1. Parses argv for an optional dataset path
2. Stashes the path on the AppState singleton for the welcome page to consume
3. Registers page handlers
4. Opens a native desktop window via NiceGUI + pywebview

Usage:
    hrfunc                              # launch the GUI
    hrfunc /path/to/study               # launch with that folder preloaded
    hrfunc subject_01.snirf             # launch with a single file preloaded
    hrfunc --version                    # print version and exit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns the exit code (0 on success).

    Args:
        argv: Argument list (excluding the program name). When None
            (production), argparse pulls from sys.argv. Tests pass an
            explicit list to exercise without touching sys.argv.

    Returns:
        Exit code. 0 if the GUI ran cleanly. Non-zero on argument-parsing
        failure (argparse handles those itself via SystemExit).
    """
    parser = _build_argument_parser()
    args = parser.parse_args(argv)

    # Import NiceGUI lazily so `hrfunc --version` and `hrfunc --help` work
    # even if the [gui] extras are not fully installed (e.g. user is checking
    # what version they have before pip install hrfunc[gui]).
    from nicegui import ui

    from .state import state
    from .theme import apply_theme

    if args.path is not None:
        state.preload_path = args.path.resolve()
        logger.info("Preloading path from CLI: %s", state.preload_path)

    _register_pages()

    ui.run(
        title="HRFunc",
        native=True,
        window_size=(1400, 900),
        reload=False,
        show=True,
        port=_find_free_port(),
        show_welcome_message_on_startup=False,
    )
    return 0


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hrfunc",
        description=(
            "HRFunc — fNIRS hemodynamic response function estimation and "
            "neural activity recovery. Launches the desktop GUI."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        type=Path,
        help=(
            "Optional dataset to preload — a folder of scans, a single "
            ".snirf/.fif file, or a NIRx acquisition directory."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hrfunc {_get_version()}",
    )
    return parser


def _register_pages() -> None:
    """Register all `@ui.page` handlers.

    Sprint 2.1 ships only the placeholder root page. Sprint 2.2 replaces it
    with the real welcome page; Sprint 2.3 adds /workspace. Page imports are
    lazy so reloading individual modules during development picks up changes
    without restarting the NiceGUI server.
    """
    from nicegui import ui

    from .theme import apply_theme, page_container

    @ui.page("/")
    def _placeholder_root_page() -> None:
        apply_theme()
        with page_container():
            ui.label("HRFunc").classes("text-5xl font-bold")
            ui.label(
                "Sprint 2.1 placeholder. The welcome screen lands in Sprint 2.2."
            ).classes("text-lg opacity-80")


def _find_free_port() -> int:
    """Return an unused localhost port for NiceGUI to bind.

    NiceGUI defaults to 8080, which collides if anything else on the user's
    machine is listening there (common — many dev servers default to 8080).
    Letting the OS pick avoids a confusing "address already in use" crash
    on launch.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _get_version() -> str:
    """Return the installed hrfunc version, or 'unknown' if not resolvable."""
    try:
        from importlib.metadata import version
        return version("hrfunc")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
