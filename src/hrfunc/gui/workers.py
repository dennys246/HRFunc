"""Background-task helpers for long-running GUI operations.

NiceGUI runs page handlers on its event loop; any synchronous work that takes
more than ~100ms blocks the UI. Scanning a large folder or estimating HRFs
take seconds to minutes — they must run off the main thread.

This module provides a thin wrapper around `asyncio.to_thread` plus a
progress-state helper for surfacing `progress_callback` events to the UI.
Sprint 2.1 ships the helper; Sprint 3 (estimate panel) wires it up to actual
estimation calls.

Design constraints:
- **Single background worker at a time.** AppState.busy is a binary flag,
  not a counter. The GUI disables long-task buttons while busy=True so the
  user can't queue overlapping work. This matches the RawCache's not-thread-
  safe contract (see hrfunc.io.raw_cache).
- **Progress is pushed, not polled.** The callback writes (current, total,
  name) into `state.estimation_progress`; UI components bind to that field
  and re-render via NiceGUI's reactivity. No timer needed.
- **Errors surface to `state.last_error`.** The worker catches exceptions
  raised in the threaded function and stores the string message; the GUI
  displays a toast / banner from there.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .state import AppState

logger = logging.getLogger(__name__)


def make_progress_callback(state: AppState) -> Callable[[int, int, str], None]:
    """Return a `progress_callback` that writes into the given AppState.

    The returned callable matches the signature expected by
    `montage.estimate_hrf` and `montage.estimate_activity`:
    `(current_index, total_channels, channel_name) -> None`.

    Each call writes a `(current, total, name)` tuple into
    `state.estimation_progress`. UI components bound to that field re-render
    automatically via NiceGUI's reactivity.
    """

    def _callback(current: int, total: int, name: str) -> None:
        state.estimation_progress = (current, total, name)

    return _callback


async def run_in_background(
    state: AppState,
    func: Callable[..., Any],
    *args: Any,
    on_done: Optional[Callable[[Any], Awaitable[None]]] = None,
    **kwargs: Any,
) -> Any:
    """Run a blocking function off the main thread, surfacing busy/error state.

    Sets `state.busy = True` before dispatch, clears it (and resets
    `estimation_progress`) when the function returns. Any exception is
    logged and stored in `state.last_error` as a string; the exception is
    NOT re-raised so the GUI stays responsive.

    Args:
        state: AppState whose `busy`, `estimation_progress`, and `last_error`
            fields will be updated.
        func: Synchronous callable to run on a worker thread.
        *args, **kwargs: Forwarded to `func`.
        on_done: Optional async callable invoked with the result after `func`
            completes successfully. Useful for "estimate, then refresh the
            HRF gallery" flows.

    Returns:
        The result of `func`, or `None` if `func` raised.
    """
    if state.busy:
        logger.warning(
            "run_in_background: state.busy is already True; refusing to "
            "start a second worker. The GUI should disable trigger buttons "
            "while busy."
        )
        return None

    state.busy = True
    state.last_error = None
    result: Any = None
    try:
        # Use run_in_executor instead of asyncio.to_thread (3.9+) so the GUI
        # works on Python 3.8 to match the library's requires-python pin.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: func(*args, **kwargs)
        )
    except Exception as exc:  # noqa: BLE001 — see module docstring
        state.last_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Background worker failed: %s", exc)
        return None
    finally:
        state.busy = False
        state.estimation_progress = None

    if on_done is not None:
        try:
            await on_done(result)
        except Exception as exc:  # noqa: BLE001
            state.last_error = f"on_done failed: {type(exc).__name__}: {exc}"
            logger.exception("on_done callback failed: %s", exc)

    return result
