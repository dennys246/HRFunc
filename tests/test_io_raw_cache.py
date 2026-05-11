"""Targeted unit tests for feat/io-raw-cache (v1.3.0 GUI foundation 4/4).

Covers ``hrfunc.io.raw_cache.RawCache`` — the in-memory LRU cache that holds
loaded MNE Raw objects for the GUI's dataset-tree navigation. The cache:

- evicts least-recently-used entries when over capacity
- promotes cache hits to most-recently-used
- dispatches loading by format (snirf / nirx_dir / fif) via classify_path
- accepts ScanEntry, Path, or str inputs

Most LRU/dispatch behavior is exercised via a monkeypatched fake loader so
tests stay fast and deterministic. A handful of smoke tests load real
fixtures end-to-end to confirm the MNE dispatch wiring actually works.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hrfunc.io.manifest import ScanEntry
from hrfunc.io.raw_cache import DEFAULT_MAXSIZE, RawCache

DATA_ROOT = Path(__file__).parent / "data"
SNIRF_FILE = DATA_ROOT / "sNIRF_formatted" / "subject_1.snirf"
SNIRF_FILE_2 = DATA_ROOT / "sNIRF_formatted" / "subject_2.snirf"
NIRX_DIR = DATA_ROOT / "NIRX_formatted" / "subject_1"
FIF_FILE = DATA_ROOT / "FIF_formatted" / "subject_1.fif"


class _FakeRaw:
    """Stand-in for mne.io.BaseRaw — lets us assert identity without loading MNE."""

    def __init__(self, path):
        self.path = path

    def __repr__(self) -> str:
        return f"_FakeRaw({self.path})"


@pytest.fixture
def fake_loader(monkeypatch, tmp_path):
    """Replace RawCache._load_from_disk with a counting fake.

    Returns a callable that creates fake snirf files in tmp_path for tests
    that need real on-disk paths. Tests can inspect ``calls`` to verify how
    many disk loads occurred.
    """
    calls = []

    def _fake_load(path):
        calls.append(path)
        return _FakeRaw(path)

    monkeypatch.setattr(
        "hrfunc.io.raw_cache.RawCache._load_from_disk",
        staticmethod(_fake_load),
    )

    def _make_fake_snirf(name="scan.snirf"):
        """Create a real .snirf file (empty content is fine — _extract_path
        only calls .resolve(), which doesn't read content) so the path
        actually exists on disk."""
        p = tmp_path / name
        p.write_bytes(b"")
        return p

    return calls, _make_fake_snirf


# ---------------------------------------------------------------------------
# Construction & basic state
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_maxsize_is_three(self):
        c = RawCache()
        assert c.maxsize == DEFAULT_MAXSIZE == 3

    def test_custom_maxsize_accepted(self):
        c = RawCache(maxsize=10)
        assert c.maxsize == 10

    def test_zero_maxsize_raises(self):
        with pytest.raises(ValueError, match="maxsize must be >= 1"):
            RawCache(maxsize=0)

    def test_negative_maxsize_raises(self):
        with pytest.raises(ValueError, match="maxsize must be >= 1"):
            RawCache(maxsize=-1)

    def test_new_cache_is_empty(self):
        c = RawCache()
        assert len(c) == 0


# ---------------------------------------------------------------------------
# Cache hits and misses
# ---------------------------------------------------------------------------


class TestCacheHitsAndMisses:
    def test_first_get_is_a_miss_and_loads_from_disk(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(p)
        assert len(calls) == 1

    def test_second_get_on_same_path_is_a_hit(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(p)
        cache.get(p)
        assert len(calls) == 1  # no second load

    def test_cache_hit_returns_same_object(self, fake_loader):
        _, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        first = cache.get(p)
        second = cache.get(p)
        assert first is second

    def test_contains_returns_true_after_get(self, fake_loader):
        _, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        assert p not in cache
        cache.get(p)
        assert p in cache


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------


class TestLruEviction:
    def test_over_capacity_evicts_oldest(self, fake_loader):
        _, make_snirf = fake_loader
        a = make_snirf("a.snirf")
        b = make_snirf("b.snirf")
        c = make_snirf("c.snirf")
        d = make_snirf("d.snirf")
        cache = RawCache(maxsize=3)
        cache.get(a)
        cache.get(b)
        cache.get(c)
        assert len(cache) == 3
        cache.get(d)  # forces eviction
        assert len(cache) == 3
        assert a not in cache
        assert d in cache

    def test_cache_hit_promotes_to_most_recently_used(self, fake_loader):
        """A access an old entry → it should NOT be the next evicted.

        Sequence: load A, B, C (cache full). Access A (promotes it to MRU).
        Load D (forces eviction). B should be evicted, not A."""
        _, make_snirf = fake_loader
        a = make_snirf("a.snirf")
        b = make_snirf("b.snirf")
        c = make_snirf("c.snirf")
        d = make_snirf("d.snirf")
        cache = RawCache(maxsize=3)
        cache.get(a)
        cache.get(b)
        cache.get(c)
        cache.get(a)  # promote A
        cache.get(d)  # force eviction
        assert a in cache  # A survived because it was promoted
        assert b not in cache  # B was the actual LRU
        assert c in cache
        assert d in cache

    def test_maxsize_one_keeps_only_latest(self, fake_loader):
        _, make_snirf = fake_loader
        a = make_snirf("a.snirf")
        b = make_snirf("b.snirf")
        cache = RawCache(maxsize=1)
        cache.get(a)
        cache.get(b)
        assert a not in cache
        assert b in cache
        assert len(cache) == 1


# ---------------------------------------------------------------------------
# Explicit eviction and clearing
# ---------------------------------------------------------------------------


class TestExplicitEviction:
    def test_evict_removes_present_entry(self, fake_loader):
        _, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(p)
        assert cache.evict(p) is True
        assert p not in cache

    def test_evict_returns_false_for_missing_entry(self, fake_loader, tmp_path):
        # Entry never inserted
        cache = RawCache()
        missing = tmp_path / "never_loaded.snirf"
        missing.write_bytes(b"")
        assert cache.evict(missing) is False

    def test_clear_drops_everything(self, fake_loader):
        _, make_snirf = fake_loader
        a = make_snirf("a.snirf")
        b = make_snirf("b.snirf")
        cache = RawCache()
        cache.get(a)
        cache.get(b)
        cache.clear()
        assert len(cache) == 0
        assert a not in cache
        assert b not in cache


# ---------------------------------------------------------------------------
# Input shape — ScanEntry, Path, str
# ---------------------------------------------------------------------------


class TestInputShape:
    def test_str_path_accepted(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(str(p))
        assert len(calls) == 1

    def test_pathlib_path_accepted(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(p)
        assert len(calls) == 1

    def test_scan_entry_accepted(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        entry = ScanEntry(format="snirf", path=p)
        cache = RawCache()
        cache.get(entry)
        assert len(calls) == 1

    def test_str_path_and_path_object_share_cache_entry(self, fake_loader):
        """Same scan loaded via str and Path should hit the same cache slot —
        otherwise the cache would double-fill on path-style mismatches."""
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        cache = RawCache()
        cache.get(str(p))
        cache.get(p)
        assert len(calls) == 1
        assert len(cache) == 1

    def test_scan_entry_and_path_share_cache_entry(self, fake_loader):
        calls, make_snirf = fake_loader
        p = make_snirf("a.snirf")
        entry = ScanEntry(format="snirf", path=p)
        cache = RawCache()
        cache.get(entry)
        cache.get(p)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unrecognized_path_raises(self, tmp_path):
        """A text file is not an fNIRS dataset — RawCache should raise
        rather than caching some sentinel that would mask the error later."""
        bad = tmp_path / "notes.txt"
        bad.write_text("not fNIRS")
        cache = RawCache()
        with pytest.raises(ValueError, match="not a recognized fNIRS dataset"):
            cache.get(bad)

    def test_failed_load_does_not_pollute_cache(self, tmp_path):
        bad = tmp_path / "notes.txt"
        bad.write_text("not fNIRS")
        cache = RawCache()
        with pytest.raises(ValueError):
            cache.get(bad)
        assert len(cache) == 0


# ---------------------------------------------------------------------------
# Real-fixture integration smoke tests
# ---------------------------------------------------------------------------


class TestRealFixtureIntegration:
    """One end-to-end test per supported format. These actually load through
    MNE so they're slower than the mocked tests above — keep the count small."""

    def test_loads_snirf_file(self):
        import mne
        cache = RawCache()
        raw = cache.get(SNIRF_FILE)
        assert isinstance(raw, mne.io.BaseRaw)
        assert SNIRF_FILE in cache

    def test_loads_nirx_directory(self):
        import mne
        cache = RawCache()
        raw = cache.get(NIRX_DIR)
        assert isinstance(raw, mne.io.BaseRaw)
        assert NIRX_DIR in cache

    def test_loads_fif_file(self):
        import mne
        cache = RawCache()
        raw = cache.get(FIF_FILE)
        assert isinstance(raw, mne.io.BaseRaw)
        assert FIF_FILE in cache

    def test_real_cache_hit_skips_second_load(self):
        """Two loads of the same SNIRF on a real fixture — second call
        must return the same object as the first (cache hit). Without
        caching, MNE would reparse the SNIRF and return a different
        in-memory object."""
        cache = RawCache()
        first = cache.get(SNIRF_FILE)
        second = cache.get(SNIRF_FILE)
        assert first is second
