"""Targeted unit tests for feat/io-detect (v1.3.0 GUI foundation 2/4).

Covers ``hrfunc.io.detect.classify_path`` — the single entry point for
deciding whether a filesystem path is an fNIRS dataset and, if so, which
of the three supported formats (snirf, nirx_dir, fif) it belongs to.

Positive tests use the bundled fixtures at tests/data/{FIF,NIRX,sNIRF}_formatted.
Negative tests use tmp_path so we can synthesize partial-marker directories,
non-fNIRS FIFs, and corrupt files without polluting the real fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hrfunc.io.detect import FormatHit, classify_path

DATA_ROOT = Path(__file__).parent / "data"
SNIRF_FILE = DATA_ROOT / "sNIRF_formatted" / "subject_1.snirf"
NIRX_DIR = DATA_ROOT / "NIRX_formatted" / "subject_1"
FIF_FILE = DATA_ROOT / "FIF_formatted" / "subject_1.fif"


# ---------------------------------------------------------------------------
# Positive cases — one per supported format against real fixtures
# ---------------------------------------------------------------------------


class TestSnirfDetection:
    def test_snirf_file_is_classified_as_snirf(self):
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        assert hit.format == "snirf"

    def test_snirf_file_is_marked_fnirs(self):
        """SNIRF is fNIRS-only by definition; no content sniff needed."""
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        assert hit.is_fnirs is True

    def test_snirf_path_is_preserved(self):
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        assert hit.path == SNIRF_FILE

    def test_snirf_suffix_case_insensitive(self, tmp_path):
        """Real-world filesystems sometimes hand us .SNIRF (uppercase) — we
        should still recognize it. Guards against losing files from a scan
        because of case sensitivity."""
        upper = tmp_path / "scan.SNIRF"
        upper.write_bytes(b"")  # content does not matter for classification
        hit = classify_path(upper)
        assert hit is not None
        assert hit.format == "snirf"


class TestNirxDirDetection:
    def test_nirx_dir_is_classified_as_nirx_dir(self):
        hit = classify_path(NIRX_DIR)
        assert hit is not None
        assert hit.format == "nirx_dir"

    def test_nirx_dir_is_marked_fnirs(self):
        hit = classify_path(NIRX_DIR)
        assert hit is not None
        assert hit.is_fnirs is True

    def test_nirx_dir_path_is_preserved(self):
        """The returned path must be the directory itself, not a file inside,
        because mne.io.read_raw_nirx consumes the directory as its argument."""
        hit = classify_path(NIRX_DIR)
        assert hit is not None
        assert hit.path == NIRX_DIR


class TestFifDetection:
    def test_fif_with_fnirs_channels_is_classified(self):
        hit = classify_path(FIF_FILE)
        assert hit is not None
        assert hit.format == "fif"
        assert hit.is_fnirs is True

    def test_fif_n_channels_populated(self):
        """The FIF code path runs a cheap info read, so we get channel count
        and sfreq for free — populate them so callers don't need a second
        read. Regression guard: if a future refactor stops populating these,
        the scanner will need an extra MNE call per file."""
        hit = classify_path(FIF_FILE)
        assert hit is not None
        assert hit.n_channels is not None and hit.n_channels > 0

    def test_fif_sfreq_populated(self):
        hit = classify_path(FIF_FILE)
        assert hit is not None
        assert hit.sfreq is not None and hit.sfreq > 0


# ---------------------------------------------------------------------------
# Negative cases — paths that should classify as None
# ---------------------------------------------------------------------------


class TestNonExistentPath:
    def test_missing_file_returns_none(self, tmp_path):
        assert classify_path(tmp_path / "does_not_exist.snirf") is None

    def test_missing_directory_returns_none(self, tmp_path):
        assert classify_path(tmp_path / "no_such_dir") is None


class TestUnrelatedFiles:
    def test_text_file_returns_none(self, tmp_path):
        f = tmp_path / "notes.txt"
        f.write_text("not an fNIRS file")
        assert classify_path(f) is None

    def test_unknown_extension_returns_none(self, tmp_path):
        f = tmp_path / "data.bin"
        f.write_bytes(b"\x00\x01\x02")
        assert classify_path(f) is None


class TestPartialNirxMarkers:
    """NIRx classification requires BOTH a probeInfo.mat and a wl1 file. Either
    alone is insufficient — guards against false positives in shared-probe
    directories that don't contain recordings."""

    def test_only_probe_info_returns_none(self, tmp_path):
        (tmp_path / "study_probeInfo.mat").write_bytes(b"")
        assert classify_path(tmp_path) is None

    def test_only_wl1_returns_none(self, tmp_path):
        (tmp_path / "study.wl1").write_bytes(b"")
        assert classify_path(tmp_path) is None

    def test_neither_marker_returns_none(self, tmp_path):
        (tmp_path / "random.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b,c")
        assert classify_path(tmp_path) is None

    def test_both_markers_classifies_as_nirx_dir(self, tmp_path):
        """Smoke check that the AND-of-markers rule works on a synthetic
        folder, independent of the real fixture."""
        (tmp_path / "scan_probeInfo.mat").write_bytes(b"")
        (tmp_path / "scan.wl1").write_bytes(b"")
        hit = classify_path(tmp_path)
        assert hit is not None
        assert hit.format == "nirx_dir"


class TestNonFnirsFif:
    """FIF files can carry MEG, EEG, or fNIRS — only fNIRS-containing files
    should classify. Synthesizes an EEG-only FIF to verify the sniff
    rejects it."""

    def _write_eeg_fif(self, tmp_path: Path) -> Path:
        import mne
        info = mne.create_info(
            ch_names=["Fp1", "Fp2"], sfreq=256.0, ch_types="eeg"
        )
        raw = mne.io.RawArray(np.zeros((2, 50)), info, verbose="ERROR")
        out = tmp_path / "eeg_only_raw.fif"
        raw.save(out, overwrite=True, verbose="ERROR")
        return out

    def test_eeg_only_fif_returns_none(self, tmp_path):
        fif = self._write_eeg_fif(tmp_path)
        assert classify_path(fif) is None

    def test_corrupt_fif_returns_none(self, tmp_path):
        """A garbage .fif must not crash classify_path — folder scans iterate
        over many files and one corrupt file should not abort the scan."""
        f = tmp_path / "corrupt_raw.fif"
        f.write_bytes(b"not a real fif")
        assert classify_path(f) is None


# ---------------------------------------------------------------------------
# Input shape — both str and Path should work, both absolute and relative
# ---------------------------------------------------------------------------


class TestInputShape:
    def test_string_path_accepted(self):
        hit = classify_path(str(SNIRF_FILE))
        assert hit is not None
        assert hit.format == "snirf"

    def test_pathlib_path_accepted(self):
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        assert hit.format == "snirf"

    def test_relative_path_accepted(self, tmp_path, monkeypatch):
        f = tmp_path / "scan.snirf"
        f.write_bytes(b"")
        monkeypatch.chdir(tmp_path)
        hit = classify_path("scan.snirf")
        assert hit is not None
        assert hit.format == "snirf"


# ---------------------------------------------------------------------------
# FormatHit dataclass contract
# ---------------------------------------------------------------------------


class TestFormatHitContract:
    def test_format_hit_is_frozen(self):
        """Hits are immutable so they can be cached safely (the v1.3.0 scanner
        keeps them in a per-folder manifest). A future refactor that drops
        the frozen=True would silently break cache invariants.

        dataclasses raises FrozenInstanceError (a subclass of AttributeError)
        on assignment to a frozen instance."""
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        with pytest.raises(AttributeError):
            hit.format = "fif"  # type: ignore[misc]

    def test_format_hit_carries_required_fields(self):
        hit = classify_path(SNIRF_FILE)
        assert hit is not None
        assert hasattr(hit, "format")
        assert hasattr(hit, "path")
        assert hasattr(hit, "is_fnirs")
        assert hasattr(hit, "n_channels")
        assert hasattr(hit, "sfreq")
