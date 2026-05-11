"""AppState — single source of truth for the running GUI.

NiceGUI page handlers, components, and background workers all read and write
the same `AppState` instance. A module-level `state` singleton is created at
import time so any module can ``from hrfunc.gui.state import state`` without
threading a reference through every function signature.

Lifecycle:
- One AppState per process. The `state` singleton is created on first import.
- Tests can instantiate fresh AppState() instances for isolated unit testing
  (the class itself is just a dataclass).
- The singleton holds mutable fields by design — pages bind their UI elements
  to these fields and re-render on changes.

What lives here:
- `manifest`           - last folder scan result (None until a scan completes)
- `selected_scan`      - currently inspected ScanEntry, or None
- `raw_cache`          - hot-path LRU(3) loader of MNE Raw objects
- `preload_path`       - CLI arg from `hrfunc <path>`; consumed by welcome page on first render
- `busy`               - True while a background task is running (drives spinner UI)
- `estimation_progress`- (current, total, channel_name) tuple from the latest
                         progress_callback fire; None when no estimation is in flight
- `last_error`         - last error message surfaced to the user, or None

Fields are added (not removed) as later sprints integrate more state. Keeping
the AppState surface stable across sprints means GUI components written in
Sprint 2 won't need updates as Sprint 3+ panels land.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from ..io.manifest import Manifest, ScanEntry
from ..io.raw_cache import RawCache


@dataclass
class AppState:
    """Mutable, single-process GUI state.

    All fields default to `None` / empty so a freshly-constructed AppState
    represents "no data loaded yet" — the state shown by the welcome page.
    """

    manifest: Optional[Manifest] = None
    selected_scan: Optional[ScanEntry] = None
    raw_cache: RawCache = field(default_factory=RawCache)
    preload_path: Optional[Path] = None
    busy: bool = False
    estimation_progress: Optional[Tuple[int, int, str]] = None
    last_error: Optional[str] = None

    def reset(self) -> None:
        """Return to the welcome-screen state.

        Used when the user closes the current project / switches datasets.
        Drops cached Raws so memory is released. The RawCache instance is
        kept (not reassigned) so any references held elsewhere stay valid.
        """
        self.manifest = None
        self.selected_scan = None
        self.raw_cache.clear()
        self.preload_path = None
        self.busy = False
        self.estimation_progress = None
        self.last_error = None


# Module-level singleton. Page handlers and components import this directly.
state = AppState()
