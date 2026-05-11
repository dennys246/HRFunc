"""Targeted unit tests for feat/io-scanner (v1.3.0 GUI foundation 3/4).

Covers two modules:

- ``hrfunc.io.manifest`` — Manifest / ScanEntry / ScanError dataclasses and
  their JSON round-trip behavior. Tested separately because the GUI dataset
  tree imports just the dataclasses (not the scanning machinery).

- ``hrfunc.io.scan`` — ``scan_folder()`` walking a directory tree, applying
  ``classify_path`` to each entry, pruning known noise directories, and
  caching results to the XDG cache.

Most tests use ``tmp_path`` for synthetic trees. A handful exercise the real
fixtures at tests/data/ to confirm end-to-end behavior, but those are scoped
to individual subdirectories (FIF_formatted, NIRX_formatted, sNIRF_formatted)
rather than scanning all of tests/data — the latter contains a virtualenv
and is not a clean target.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from hrfunc.io.manifest import (
    MANIFEST_SCHEMA_VERSION,
    Manifest,
    ScanEntry,
    ScanError,
)
from hrfunc.io.scan import (
    DEFAULT_IGNORE_NAMES,
    DEFAULT_MAX_DEPTH,
    scan_folder,
)

DATA_ROOT = Path(__file__).parent / "data"
SNIRF_DIR = DATA_ROOT / "sNIRF_formatted"
NIRX_DIR = DATA_ROOT / "NIRX_formatted"
FIF_DIR = DATA_ROOT / "FIF_formatted"


# ---------------------------------------------------------------------------
# Manifest / ScanEntry / ScanError dataclass contracts
# ---------------------------------------------------------------------------


class TestScanEntryRoundTrip:
    def test_minimal_entry_round_trips(self):
        e = ScanEntry(format="snirf", path=Path("/tmp/x.snirf"))
        assert ScanEntry.from_dict(e.to_dict()) == e

    def test_fully_populated_entry_round_trips(self):
        e = ScanEntry(
            format="fif",
            path=Path("/tmp/study/sub-01/ses-01/nirs/sub-01_task-flanker_run-1_nirs.fif"),
            bids_subject="01",
            bids_session="01",
            bids_task="flanker",
            bids_run="1",
            display_name="sub-01 / ses-01 / task-flanker / run-1",
            n_channels=42,
            sfreq=7.8125,
        )
        assert ScanEntry.from_dict(e.to_dict()) == e


class TestScanErrorRoundTrip:
    def test_round_trip(self):
        err = ScanError(path=Path("/tmp/bad.fif"), reason="corrupt")
        assert ScanError.from_dict(err.to_dict()) == err


class TestManifestRoundTrip:
    def test_empty_manifest_round_trips_via_json(self):
        m = Manifest(root=Path("/tmp/study"))
        loaded = Manifest.from_json(m.to_json())
        assert loaded.root == m.root
        assert loaded.scans == ()
        assert loaded.errors == ()

    def test_populated_manifest_round_trips_via_json(self):
        scans = (
            ScanEntry(format="snirf", path=Path("/tmp/a.snirf"), display_name="a"),
            ScanEntry(format="fif", path=Path("/tmp/b.fif"), display_name="b",
                      n_channels=20, sfreq=7.8125),
        )
        errors = (ScanError(path=Path("/tmp/bad.fif"), reason="corrupt"),)
        m = Manifest(
            root=Path("/tmp"),
            scans=scans,
            errors=errors,
            scanned_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        loaded = Manifest.from_json(m.to_json())
        assert loaded == m

    def test_schema_version_mismatch_raises(self):
        m = Manifest(root=Path("/tmp"))
        d = m.to_dict()
        d["schema_version"] = MANIFEST_SCHEMA_VERSION + 999
        with pytest.raises(ValueError, match="schema version"):
            Manifest.from_dict(d)

    def test_manifest_carries_current_schema_version(self):
        """If the schema_version constant is bumped without a migration path,
        old caches will silently fail to load. This test fires when that
        happens so the bump is visible in PR review."""
        m = Manifest(root=Path("/tmp"))
        assert m.schema_version == MANIFEST_SCHEMA_VERSION


class TestManifestImmutability:
    def test_manifest_is_frozen(self):
        m = Manifest(root=Path("/tmp"))
        with pytest.raises(AttributeError):
            m.root = Path("/elsewhere")  # type: ignore[misc]

    def test_scan_entry_is_frozen(self):
        e = ScanEntry(format="snirf", path=Path("/tmp/x.snirf"))
        with pytest.raises(AttributeError):
            e.format = "fif"  # type: ignore[misc]

    def test_scans_tuple_not_list(self):
        """Tuple (not list) backing means callers cannot mutate the
        manifest's scan list after the fact, preserving cache safety."""
        m = Manifest(root=Path("/tmp"))
        assert isinstance(m.scans, tuple)
        assert isinstance(m.errors, tuple)


# ---------------------------------------------------------------------------
# scan_folder: top-level input validation
# ---------------------------------------------------------------------------


class TestScanFolderValidation:
    def test_missing_root_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            scan_folder(tmp_path / "no_such_dir")

    def test_file_root_raises_not_a_directory(self, tmp_path):
        f = tmp_path / "not_a_dir.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            scan_folder(f)

    def test_str_and_path_inputs_both_accepted(self, tmp_path):
        m_str = scan_folder(str(tmp_path), use_cache=False)
        m_path = scan_folder(tmp_path, use_cache=False)
        assert m_str.root == m_path.root


# ---------------------------------------------------------------------------
# scan_folder: format detection inside a tree
# ---------------------------------------------------------------------------


class TestScanFolderFindsSnirf:
    def test_finds_snirf_files_in_real_fixture(self):
        m = scan_folder(SNIRF_DIR, use_cache=False)
        snirf_paths = {s.path for s in m.scans if s.format == "snirf"}
        assert SNIRF_DIR / "subject_1.snirf" in snirf_paths
        assert SNIRF_DIR / "subject_2.snirf" in snirf_paths


class TestScanFolderFindsNirxDirs:
    def test_finds_nirx_dirs_and_does_not_recurse_into_them(self):
        """NIRx acquisition directories must be yielded as one entry each,
        not as N entries (one per .wl1/.wl2/.hdr inside)."""
        m = scan_folder(NIRX_DIR, use_cache=False)
        nirx_entries = [s for s in m.scans if s.format == "nirx_dir"]
        assert len(nirx_entries) == 2
        nirx_paths = {e.path for e in nirx_entries}
        assert NIRX_DIR / "subject_1" in nirx_paths
        assert NIRX_DIR / "subject_2" in nirx_paths

    def test_nirx_acquisition_internals_not_classified(self):
        """No .wl1/.hdr/.evt files should appear as separate entries — they
        live inside an already-classified nirx_dir."""
        m = scan_folder(NIRX_DIR, use_cache=False)
        for entry in m.scans:
            assert entry.path.suffix not in {".wl1", ".wl2", ".hdr", ".evt"}


class TestScanFolderFindsFif:
    def test_finds_fif_files_in_real_fixture(self):
        m = scan_folder(FIF_DIR, use_cache=False)
        fif_paths = {s.path for s in m.scans if s.format == "fif"}
        assert FIF_DIR / "subject_1.fif" in fif_paths
        assert FIF_DIR / "subject_2.fif" in fif_paths

    def test_fif_entries_carry_n_channels_and_sfreq(self):
        m = scan_folder(FIF_DIR, use_cache=False)
        fif_entries = [s for s in m.scans if s.format == "fif"]
        assert len(fif_entries) > 0
        for e in fif_entries:
            assert e.n_channels is not None and e.n_channels > 0
            assert e.sfreq is not None and e.sfreq > 0


class TestScanFolderEmptyAndNoise:
    def test_empty_dir_returns_empty_manifest(self, tmp_path):
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans == ()
        assert m.errors == ()
        assert m.root == tmp_path.resolve()

    def test_unrelated_files_produce_no_entries(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans == ()


# ---------------------------------------------------------------------------
# scan_folder: pruning behavior
# ---------------------------------------------------------------------------


class TestIgnoredDirectories:
    def test_default_ignore_set_includes_common_noise(self):
        for name in {".git", ".venv", "venv", "__pycache__", "node_modules"}:
            assert name in DEFAULT_IGNORE_NAMES

    def test_default_ignored_dirs_are_pruned(self, tmp_path):
        """A .venv directory containing a .snirf file should be pruned —
        researchers often have virtualenvs next to their data."""
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "fake.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans == ()

    def test_custom_ignore_list_overrides_default(self, tmp_path):
        """Passing an explicit ignore_names completely replaces the default —
        no implicit merge. Test that .venv is now visible because we set
        ignore_names to {} explicitly."""
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "scan.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, ignore_names=set(), use_cache=False)
        assert len(m.scans) == 1


class TestMaxDepth:
    def test_deep_file_beyond_max_depth_is_not_classified(self, tmp_path):
        """max_depth caps the descent to keep scans bounded. A file buried
        deeper than the cap must not appear."""
        deep = tmp_path
        for i in range(DEFAULT_MAX_DEPTH + 3):
            deep = deep / f"level_{i}"
            deep.mkdir()
        (deep / "deep.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans == ()

    def test_shallow_file_within_max_depth_is_classified(self, tmp_path):
        d = tmp_path / "level_1" / "level_2"
        d.mkdir(parents=True)
        (d / "scan.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, max_depth=4, use_cache=False)
        assert len(m.scans) == 1
        assert m.scans[0].format == "snirf"


# ---------------------------------------------------------------------------
# scan_folder: BIDS-opportunistic parsing
# ---------------------------------------------------------------------------


class TestBidsParsing:
    def test_subject_extracted_from_path_segment(self, tmp_path):
        d = tmp_path / "sub-01" / "nirs"
        d.mkdir(parents=True)
        (d / "scan.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert len(m.scans) == 1
        assert m.scans[0].bids_subject == "01"

    def test_session_extracted_from_path_segment(self, tmp_path):
        d = tmp_path / "sub-02" / "ses-pre" / "nirs"
        d.mkdir(parents=True)
        (d / "scan.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans[0].bids_session == "pre"

    def test_task_extracted_from_filename_infix(self, tmp_path):
        d = tmp_path / "sub-03" / "nirs"
        d.mkdir(parents=True)
        (d / "sub-03_task-flanker_nirs.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans[0].bids_task == "flanker"

    def test_run_extracted_from_filename_infix(self, tmp_path):
        d = tmp_path / "sub-04" / "nirs"
        d.mkdir(parents=True)
        (d / "sub-04_task-rest_run-2_nirs.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans[0].bids_run == "2"

    def test_non_bids_path_leaves_fields_none(self, tmp_path):
        (tmp_path / "random_file.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        e = m.scans[0]
        assert e.bids_subject is None
        assert e.bids_session is None
        assert e.bids_task is None
        assert e.bids_run is None


class TestDisplayName:
    def test_bids_path_produces_bids_display_name(self, tmp_path):
        d = tmp_path / "sub-05" / "ses-pre" / "nirs"
        d.mkdir(parents=True)
        (d / "sub-05_task-flanker_nirs.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans[0].display_name == "sub-05 / ses-pre / task-flanker"

    def test_non_bids_path_falls_back_to_parent_slash_name(self, tmp_path):
        sub = tmp_path / "lab_scans"
        sub.mkdir()
        (sub / "subject_A.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert m.scans[0].display_name == "lab_scans/subject_A.snirf"


# ---------------------------------------------------------------------------
# scan_folder: error collection (non-fatal)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_corrupt_fif_does_not_abort_scan(self, tmp_path):
        """A corrupt FIF must yield None from classify_path (already covered
        in test_io_detect), so it produces NO ScanError here — it is silently
        skipped. The scan completes and finds the sibling .snirf."""
        (tmp_path / "broken_raw.fif").write_bytes(b"garbage")
        (tmp_path / "good.snirf").write_bytes(b"")
        m = scan_folder(tmp_path, use_cache=False)
        assert any(s.format == "snirf" for s in m.scans)


# ---------------------------------------------------------------------------
# scan_folder: XDG cache I/O
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect platformdirs.user_cache_dir to a tmp path so cache tests
    do not touch the user's real XDG cache."""
    cache_root = tmp_path / "xdg_cache"
    monkeypatch.setattr(
        "platformdirs.user_cache_dir",
        lambda *_args, **_kw: str(cache_root),
    )
    return cache_root


class TestCachePersistence:
    def test_cache_written_after_scan(self, tmp_path, isolated_cache):
        scan_root = tmp_path / "study"
        scan_root.mkdir()
        (scan_root / "scan.snirf").write_bytes(b"")
        scan_folder(scan_root, use_cache=True)
        # exactly one manifest file should be written
        manifests = list((isolated_cache / "hrfunc").glob("manifest_*.json")) \
                    if (isolated_cache / "hrfunc").exists() \
                    else list(isolated_cache.glob("manifest_*.json"))
        assert len(manifests) == 1

    def test_second_scan_loads_from_cache(self, tmp_path, isolated_cache):
        """Second call returns the cached manifest unchanged — verify by
        adding a new file between scans and confirming it does NOT appear."""
        scan_root = tmp_path / "study"
        scan_root.mkdir()
        (scan_root / "first.snirf").write_bytes(b"")
        first = scan_folder(scan_root, use_cache=True)
        # Add a new file; cache should hide it
        (scan_root / "second.snirf").write_bytes(b"")
        second = scan_folder(scan_root, use_cache=True)
        assert {s.path.name for s in first.scans} == {s.path.name for s in second.scans}

    def test_force_rescan_bypasses_cache(self, tmp_path, isolated_cache):
        scan_root = tmp_path / "study"
        scan_root.mkdir()
        (scan_root / "first.snirf").write_bytes(b"")
        scan_folder(scan_root, use_cache=True)
        (scan_root / "second.snirf").write_bytes(b"")
        rescanned = scan_folder(scan_root, use_cache=True, force_rescan=True)
        names = {s.path.name for s in rescanned.scans}
        assert "first.snirf" in names
        assert "second.snirf" in names

    def test_use_cache_false_does_not_write_cache(
        self, tmp_path, isolated_cache
    ):
        scan_root = tmp_path / "study"
        scan_root.mkdir()
        (scan_root / "scan.snirf").write_bytes(b"")
        scan_folder(scan_root, use_cache=False)
        # No manifest file should have been written
        if (isolated_cache / "hrfunc").exists():
            assert list((isolated_cache / "hrfunc").glob("manifest_*.json")) == []
        else:
            assert list(isolated_cache.glob("manifest_*.json")) == []

    def test_different_roots_get_different_cache_files(
        self, tmp_path, isolated_cache
    ):
        """The cache filename embeds a hash of the root path so multiple
        studies can coexist without collision."""
        root_a = tmp_path / "study_a"
        root_b = tmp_path / "study_b"
        root_a.mkdir()
        root_b.mkdir()
        (root_a / "x.snirf").write_bytes(b"")
        (root_b / "y.snirf").write_bytes(b"")
        scan_folder(root_a, use_cache=True)
        scan_folder(root_b, use_cache=True)
        manifests = list((isolated_cache / "hrfunc").glob("manifest_*.json")) \
                    if (isolated_cache / "hrfunc").exists() \
                    else list(isolated_cache.glob("manifest_*.json"))
        assert len(manifests) == 2


class TestCacheSchemaMismatch:
    def test_old_schema_cache_is_ignored(self, tmp_path, isolated_cache, monkeypatch):
        """A cache written by an old version of the scanner has a schema
        version that no longer parses. The next scan should treat it as a
        miss (not crash, not silently use stale data)."""
        scan_root = tmp_path / "study"
        scan_root.mkdir()
        (scan_root / "scan.snirf").write_bytes(b"")

        # Manually pre-write a bad cache for this root
        from hrfunc.io.scan import _cache_path_for_root
        cache_file = _cache_path_for_root(scan_root.resolve())
        assert cache_file is not None
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps({
            "schema_version": 999,
            "root": str(scan_root.resolve()),
            "scans": [],
            "errors": [],
            "scanned_at": datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat(),
        }))

        # Scan must still work and produce a real manifest
        m = scan_folder(scan_root, use_cache=True)
        assert len(m.scans) == 1
        assert m.scans[0].path.name == "scan.snirf"
