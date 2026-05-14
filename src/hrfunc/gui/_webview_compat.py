"""pywebview compatibility shims for the GUI layer.

Isolates pywebview API quirks that bite our native-mode integration so the
fix lives in one place instead of being duplicated at each call site.
"""

from __future__ import annotations


def dialog_kind(webview, name: str):
    """Return a picklable file-dialog kind for ``window.create_file_dialog``.

    pywebview 6.x deprecated the module-level ``FOLDER_DIALOG`` /
    ``OPEN_DIALOG`` / ``SAVE_DIALOG`` constants in favor of the
    ``FileDialog`` enum. The deprecated constants are ``proxy_tools.Proxy``
    objects whose ``__module__`` is ``proxy_tools`` with no ``__qualname__``
    — pickle's ``save_global`` fails the identity check and raises
    ``PicklingError`` when NiceGUI's native mode tries to send them across
    its multiprocessing queue to the pywebview subprocess. The failure
    surfaces in the multiprocessing feeder thread, so the click handler
    appears to silently no-op from the user's view. The enum (proper
    Python ``Enum`` member) pickles cleanly by qualified name. Fall back to
    the legacy proxy on older pywebview where ``FileDialog`` doesn't exist.

    ``name`` is the bare kind: ``"FOLDER"``, ``"OPEN"``, or ``"SAVE"``.
    """
    file_dialog = getattr(webview, "FileDialog", None)
    if file_dialog is not None:
        return getattr(file_dialog, name)
    return getattr(webview, f"{name}_DIALOG")
