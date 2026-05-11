"""Folder scanner producing a Manifest of detected fNIRS datasets.

``scan_folder(root)`` walks a directory tree, classifies each path via
``hrfunc.io.classify_path``, and returns a frozen ``Manifest``. Results are
persisted to an XDG cache so subsequent scans of the same root are instant.

Design notes:
- NIRx directories are *not* recursed into. Once a directory classifies as
  ``nirx_dir``, its subtree is pruned. This avoids wasted work on
  ``*.wl1``/``*.wl2``/``*.hdr`` files inside the acquisition.
- Common noise directories (``.git``, ``.venv``, ``__pycache__``,
  ``node_modules``) are pruned by name. Researchers regularly drop fNIRS data
  into project trees that also contain code/build artifacts.
- BIDS metadata is parsed opportunistically via regex. No ``pybids``
  dependency; this works for ``sub-XX/ses-YY/`` style layouts even when the
  full BIDS spec is not followed.
- Cache invalidation is explicit (``force_rescan=True``) rather than
  mtime-driven. Auto-invalidation would require walking the tree to compare
  mtimes, defeating the purpose of the cache. The GUI will expose a "Rescan"
  button so users can refresh on demand.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Set, Union

from .detect import FormatHit, classify_path
from .manifest import Manifest, ScanEntry, ScanError

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 6
DEFAULT_IGNORE_NAMES: frozenset = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".mypy_cache",
    ".idea",
    ".vscode",
})

# Opportunistic BIDS parsers — match the standard sub-/ses-/task-/run- entities.
# Intentionally lenient: any path segment matching the pattern wins.
_BIDS_SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
_BIDS_SESSION_RE = re.compile(r"ses-([A-Za-z0-9]+)")
_BIDS_TASK_RE = re.compile(r"_task-([A-Za-z0-9]+)")
_BIDS_RUN_RE = re.compile(r"_run-([A-Za-z0-9]+)")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def scan_folder(
    root: Union[str, Path],
    max_depth: int = DEFAULT_MAX_DEPTH,
    ignore_names: Optional[Set[str]] = None,
    use_cache: bool = True,
    force_rescan: bool = False,
) -> Manifest:
    """Walk a folder and return a manifest of detected fNIRS datasets.

    Args:
        root: Directory to scan. Must exist and be a directory.
        max_depth: Maximum directory depth (relative to root) to descend.
            Default 6 — deep enough for ``study/sub-XX/ses-YY/nirs/file``
            layouts plus headroom, shallow enough to bound runtime.
        ignore_names: Directory names to skip wholesale. Defaults to
            DEFAULT_IGNORE_NAMES (venv/git/cache directories). Pass an
            explicit set to override.
        use_cache: If True, save the resulting manifest to the XDG cache
            and load existing cached manifests on subsequent calls.
        force_rescan: If True, ignore any cached manifest for this root
            and always walk the tree. Cache is still written (overwriting
            the old entry) when ``use_cache`` is True.

    Returns:
        A Manifest. Always returned — empty scans tuple if nothing matched.

    Raises:
        FileNotFoundError: If root does not exist.
        NotADirectoryError: If root is not a directory.
    """
    root_path = Path(root).resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Scan root does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Scan root is not a directory: {root_path}")

    if use_cache and not force_rescan:
        cached = _load_cached_manifest(root_path)
        if cached is not None:
            return cached

    ignore = ignore_names if ignore_names is not None else set(DEFAULT_IGNORE_NAMES)

    scans: List[ScanEntry] = []
    errors: List[ScanError] = []

    for hit_or_error in _walk_and_classify(root_path, max_depth, ignore):
        if isinstance(hit_or_error, ScanError):
            errors.append(hit_or_error)
        else:
            scans.append(_build_entry(hit_or_error, root_path))

    manifest = Manifest(
        root=root_path,
        scans=tuple(scans),
        errors=tuple(errors),
    )

    if use_cache:
        _save_manifest(manifest)

    return manifest


# ---------------------------------------------------------------------------
# Walk + classify
# ---------------------------------------------------------------------------


def _walk_and_classify(
    root: Path,
    max_depth: int,
    ignore_names: Set[str],
) -> Iterator:
    """Walk the tree, yielding FormatHit or ScanError for each interesting path.

    The walker:
    1. Classifies the current directory first — catches NIRx acquisition dirs.
    2. If the current dir is a NIRx hit, yields the hit and prunes its subtree.
    3. Otherwise classifies each contained file and yields any hit.

    Directories named in ``ignore_names`` are pruned wholesale before any
    classification work.
    """
    for current_dir_str, subdirs, files in os.walk(root):
        current_dir = Path(current_dir_str)

        try:
            depth = len(current_dir.relative_to(root).parts)
        except ValueError:
            depth = 0

        if depth > max_depth:
            subdirs.clear()
            continue

        # Prune well-known noise directories before recursing into them
        subdirs[:] = [d for d in subdirs if d not in ignore_names]

        # Classify the directory itself first — catches NIRx acquisition dirs
        try:
            dir_hit = classify_path(current_dir)
        except Exception as exc:
            errors_yield = ScanError(path=current_dir, reason=str(exc))
            yield errors_yield
            dir_hit = None

        if dir_hit is not None and dir_hit.format == "nirx_dir":
            yield dir_hit
            subdirs.clear()  # do not descend into the acquisition
            continue

        # Classify each file in this directory
        for filename in files:
            file_path = current_dir / filename
            try:
                file_hit = classify_path(file_path)
            except Exception as exc:
                yield ScanError(path=file_path, reason=str(exc))
                continue
            if file_hit is not None:
                yield file_hit


# ---------------------------------------------------------------------------
# Entry construction — BIDS parsing + display name
# ---------------------------------------------------------------------------


def _build_entry(hit: FormatHit, root: Path) -> ScanEntry:
    """Convert a FormatHit into a ScanEntry with BIDS metadata and display name."""
    bids = _parse_bids_metadata(hit.path, root)
    display = _make_display_name(hit, bids)
    return ScanEntry(
        format=hit.format,
        path=hit.path,
        bids_subject=bids.get("subject"),
        bids_session=bids.get("session"),
        bids_task=bids.get("task"),
        bids_run=bids.get("run"),
        display_name=display,
        n_channels=hit.n_channels,
        sfreq=hit.sfreq,
    )


def _parse_bids_metadata(path: Path, root: Path) -> dict:
    """Extract BIDS entities from a path opportunistically.

    Searches the relative path (root-anchored) plus the filename for
    standard BIDS entity patterns. Returns {subject, session, task, run},
    each value either the matched identifier or None.

    Order of search: all path components joined as a single string, so
    matches in any segment count. ``_task-`` and ``_run-`` are typically
    in the filename; ``sub-`` and ``ses-`` are typically in path segments.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path

    search_target = "/".join(rel.parts) + "/" + path.name

    def _first(pattern: re.Pattern) -> Optional[str]:
        m = pattern.search(search_target)
        return m.group(1) if m else None

    return {
        "subject": _first(_BIDS_SUBJECT_RE),
        "session": _first(_BIDS_SESSION_RE),
        "task": _first(_BIDS_TASK_RE),
        "run": _first(_BIDS_RUN_RE),
    }


def _make_display_name(hit: FormatHit, bids: dict) -> str:
    """Build a human-readable label for the GUI dataset tree.

    BIDS components win when available — researchers expect to see
    "sub-01 / ses-pre / task-flanker" rather than a filename. For
    non-BIDS paths, fall back to ``<parent>/<name>`` which is the
    most-recognizable short identifier.
    """
    parts = []
    if bids.get("subject"):
        parts.append(f"sub-{bids['subject']}")
    if bids.get("session"):
        parts.append(f"ses-{bids['session']}")
    if bids.get("task"):
        parts.append(f"task-{bids['task']}")
    if bids.get("run"):
        parts.append(f"run-{bids['run']}")

    if parts:
        return " / ".join(parts)

    return f"{hit.path.parent.name}/{hit.path.name}"


# ---------------------------------------------------------------------------
# XDG cache I/O
# ---------------------------------------------------------------------------


def _cache_root() -> Optional[Path]:
    """Return the XDG cache root for hrfunc, or None if unavailable.

    Uses ``platformdirs`` for cross-platform XDG resolution. Falls back to
    None if platformdirs is missing (declared as a core dep but a graceful
    fallback keeps the scanner usable in stripped-down environments).
    """
    try:
        import platformdirs
    except ImportError:
        logger.debug("platformdirs not installed; cache disabled")
        return None
    return Path(platformdirs.user_cache_dir("hrfunc"))


def _cache_path_for_root(root: Path) -> Optional[Path]:
    """Return the cache file path for a given scan root, or None if cache
    is unavailable.

    The filename embeds a SHA-1 of the absolute root path so multiple roots
    can coexist in the same cache directory without collision.
    """
    cache_root = _cache_root()
    if cache_root is None:
        return None
    root_hash = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:16]
    return cache_root / f"manifest_{root_hash}.json"


def _load_cached_manifest(root: Path) -> Optional[Manifest]:
    """Load a cached manifest for the given root, or None on any failure.

    Returns None for: missing cache file, JSON parse errors, schema version
    mismatches, or any other deserialization failure. The next scan will
    rebuild the cache on disk.
    """
    cache_path = _cache_path_for_root(root)
    if cache_path is None or not cache_path.exists():
        return None
    try:
        return Manifest.from_json(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Cached manifest at %s could not be loaded: %s", cache_path, exc)
        return None


def _save_manifest(manifest: Manifest) -> None:
    """Persist a manifest to the XDG cache directory.

    Silently no-ops if the cache directory cannot be written (e.g.
    read-only filesystem). The scanner returns the manifest in-memory
    either way, so cache write failures never block a scan.
    """
    cache_path = _cache_path_for_root(manifest.root)
    if cache_path is None:
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(manifest.to_json(), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not write manifest cache to %s: %s", cache_path, exc)
