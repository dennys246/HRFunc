"""Manifest dataclasses for ``hrfunc.io.scan_folder``.

A Manifest is the structured result of walking a folder for fNIRS datasets:
one ScanEntry per detected dataset, plus any ScanError encountered along the
way. Manifests are designed to be:

- **Frozen and tuple-backed** so they can be cached safely and shared across
  threads (the v1.3.0 GUI dataset tree is rendered from a cached manifest).
- **JSON-round-trippable** so the XDG cache on disk can be reloaded into the
  same shape on the next launch.

The dataclasses live in their own module (rather than inside ``scan.py``) so
they can be imported without pulling in MNE — the GUI's dataset-tree widget
needs the types but not the scanning machinery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Union

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScanEntry:
    """One detected fNIRS dataset.

    Attributes:
        format: ``"snirf"``, ``"nirx_dir"``, or ``"fif"`` — matches the values
            produced by ``hrfunc.io.classify_path``.
        path: Absolute path to the dataset (file for snirf/fif, directory for
            nirx_dir).
        bids_subject: Subject identifier parsed opportunistically from the
            path (``"sub-XX"`` segment) or None.
        bids_session: Session identifier (``"ses-YY"``) or None.
        bids_task: Task identifier (``"_task-NNN"`` filename infix) or None.
        bids_run: Run identifier (``"_run-N"`` filename infix) or None.
        display_name: Human-readable label for the GUI tree. BIDS components
            when available, otherwise ``<parent>/<filename>``.
        n_channels: Channel count when cheaply available, otherwise None.
        sfreq: Sampling frequency in Hz when cheaply available, otherwise None.
    """

    format: str
    path: Path
    bids_subject: Optional[str] = None
    bids_session: Optional[str] = None
    bids_task: Optional[str] = None
    bids_run: Optional[str] = None
    display_name: str = ""
    n_channels: Optional[int] = None
    sfreq: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "path": str(self.path),
            "bids_subject": self.bids_subject,
            "bids_session": self.bids_session,
            "bids_task": self.bids_task,
            "bids_run": self.bids_run,
            "display_name": self.display_name,
            "n_channels": self.n_channels,
            "sfreq": self.sfreq,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScanEntry":
        return cls(
            format=d["format"],
            path=Path(d["path"]),
            bids_subject=d.get("bids_subject"),
            bids_session=d.get("bids_session"),
            bids_task=d.get("bids_task"),
            bids_run=d.get("bids_run"),
            display_name=d.get("display_name", ""),
            n_channels=d.get("n_channels"),
            sfreq=d.get("sfreq"),
        )


@dataclass(frozen=True)
class ScanError:
    """A path that could not be classified during scanning.

    Errors are collected non-fatally so the GUI can surface "we couldn't read
    these N paths" without losing the rest of the scan. The reason field is a
    short human-readable string suitable for display.
    """

    path: Path
    reason: str

    def to_dict(self) -> dict:
        return {"path": str(self.path), "reason": self.reason}

    @classmethod
    def from_dict(cls, d: dict) -> "ScanError":
        return cls(path=Path(d["path"]), reason=d["reason"])


@dataclass(frozen=True)
class Manifest:
    """Frozen result of a folder scan.

    Attributes:
        root: Absolute path of the scan root.
        scans: Tuple of detected datasets, ordered by walk order.
        errors: Tuple of paths that failed classification.
        scanned_at: UTC timestamp of when the scan completed.
        schema_version: Format version of the cached representation; used to
            invalidate caches when this module's schema changes.
    """

    root: Path
    scans: Tuple[ScanEntry, ...] = field(default_factory=tuple)
    errors: Tuple[ScanError, ...] = field(default_factory=tuple)
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "root": str(self.root),
            "scans": [s.to_dict() for s in self.scans],
            "errors": [e.to_dict() for e in self.errors],
            "scanned_at": self.scanned_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Manifest":
        if d.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"Manifest schema version mismatch: cache is "
                f"{d.get('schema_version')}, current is {MANIFEST_SCHEMA_VERSION}"
            )
        return cls(
            root=Path(d["root"]),
            scans=tuple(ScanEntry.from_dict(s) for s in d.get("scans", [])),
            errors=tuple(ScanError.from_dict(e) for e in d.get("errors", [])),
            scanned_at=datetime.fromisoformat(d["scanned_at"]),
            schema_version=d["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "Manifest":
        return cls.from_dict(json.loads(text))
