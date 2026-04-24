# HRFunc Known Issues

**Status as of:** 2026-04-14
**Active work:** v1.2.0 correctness release (10 branches in flight, 6 merged to main, 4 pushed pending merge)
**Audience:** Contributors — see [plan_v1_2_0.md](../plans/plan_v1_2_0.md) and [plan_v2_0_0.md](../plans/plan_v2_0_0.md) for where each open issue is scoped.

Issues are grouped by status. File:line references are to the codebase as of this revision.

---

## RESOLVED in v1.2.0 (in flight)

The following issues have been fixed on feature branches in the v1.2.0 correctness chain. Some are merged to main, others are pushed but pending merge. See [v1_2_0_changelog_summary.md](v1_2_0_changelog_summary.md) for per-branch attribution.

### Critical — silent data corruption, wrong scientific results

| ID | Summary | Fixed in |
|----|---------|----------|
| KI-001 | `tree.filter()` condition inverted — deleted matching nodes instead of non-matching | `fix/critical-bugs-phase1a` |
| KI-002 | Mutable default args in `HRF.__init__` (`estimates=[], locations=[], context=[]`) | `fix/tree-hrf-correctness` (3.7) |
| KI-003 | `hasher.contexts = [[]] * capacity` shared list references in three locations | `fix/critical-bugs-phase1bc` |
| KI-004 | `ValueError` returned instead of raised in multiple validators | `fix/critical-bugs-phase1a` |
| KI-005 | `estimate_hrf` loaded data before preprocessing | `fix/critical-bugs-phase1bc` |
| KI-006 | `estimate_activity` preprocess return value not captured | `fix/critical-bugs-phase1bc` |
| KI-007 | `load_montage` discarded inserted HRF nodes via duplicate tree re-init | `fix/critical-bugs-phase1bc` |
| NE-001 | `tree.insert` stamped `context['method']='canonical'` on first user HRF (clobbered user data) | `fix/tree-hrf-correctness` |
| NE-002 | `load_hrfs`/`load_montage` populated hasher by dict KEYS; `tree.branch` searched by VALUES | `fix/hasher-branch-correctness` |
| S4 | Hardcoded canonical HRF at `t_r=0.128` regardless of scan sfreq | `fix/canonical-hrf-sfreq` |
| H1 | Zero-trace HRFs produced silent NaN in `estimate_activity` (division by `max(abs(zeros))`) | `fix/input-validation` |
| — | `calc_snr` PSD variable swap — each band read from the signal filtered to the WRONG band, noise power ≈ 0, SNR effectively infinite | `fix/observer-and-typos` (lint-sweep bonus) |
| — | `load_montage` dropped `channel['context']` when constructing HRF nodes — post-NE-002 this broke `compare_context` / branch / filter on loaded montages | `fix/tree-edge-cases` (cross-branch audit) |

### High — crashes or wrong algorithm behavior

| ID | Summary | Fixed in |
|----|---------|----------|
| KI-008 | `hasher.probe_count` never initialized in `__init__` | `fix/critical-bugs-phase1bc` |
| KI-009 | `tree._delete_recursive` 5-arg-to-3-arg signature mismatch + non-existent `node.hrf_data` | `fix/tree-delete-filter` |
| KI-010 | `compare_context` signature mismatch with callers | `fix/critical-bugs-phase1bc` |
| KI-011 | `compare_context` wrong denominator (`len(first_context)` vs `len(values)`) | `fix/critical-bugs-phase1bc` |
| KI-012 | Divide-by-zero guards in `hasher.__repr__` and `compare_context` | `fix/critical-bugs-phase1bc`, `fix/hasher-branch-correctness` |
| KI-013 / 3.5 | Jitter loop variable bug — pre-fix mutated a temp var, not `hrf.x/y/z` | `fix/tree-hrf-correctness` |
| KI-014 | `hasher.search` could infinite-loop on full table | `fix/critical-bugs-phase1bc` |
| KI-015 / 3.8 | `tree.gather(None)` crashed on `.left` access | `fix/tree-delete-filter` |
| KI-016 | `double_probe(key, hashkey, False)` passed `False` as the linear coefficient | `fix/critical-bugs-phase1a` |
| KI-017 / 3.9 | `lens.__init__` didn't initialize `self.sfreq`; `calc_snr` AttributeError'd | `fix/observer-and-typos` |
| KI-018 / 3.10 | `HRF.build` referenced the never-set `self.type` | `fix/tree-hrf-correctness` |
| NE-003 | `HRF.process_options=[]` made `build()` zip zero-iterate; `spline_interp` took no trace arg | `fix/tree-hrf-correctness` |
| NE-004 | `HRF.smooth` called `self.gaussian_filter1d` which was never imported | `fix/tree-hrf-correctness` |
| NE-006 | `tree.merge` inserted source node references, sharing mutable child pointers | `fix/tree-edge-cases` |
| NE-007 | `tree.nearest_neighbor` fallback returned `self.root.right` which crashed on empty tree | `fix/canonical-hrf-sfreq` + explicit guard in `fix/tree-edge-cases` |
| H2 | Same as KI-009 (delete path) | `fix/tree-delete-filter` |
| H3 | `montage.__repr__` crashed on unconfigured montage | `fix/state-lifecycle` |
| H4 | `hasher.search` returned `False` on miss; `tree.branch` TypeError'd on `for node in False:` | `fix/hasher-branch-correctness` |

### Medium — behavioral issues

| ID | Summary | Fixed in |
|----|---------|----------|
| KI-019 / ND-003 | `success` variable scope broken in `estimate_activity` closure (non-local write) | `fix/estimate-activity-threading` |
| KI-020 / ND-004 | `timeout=1500` was seconds (25 minutes), not the intended ~seconds | `fix/estimate-activity-threading` (now default 30) |
| KI-025 | `preprocess_fnirs` crashed on missing `subject_info` | `fix/critical-bugs-phase1bc` |
| KI-026 | f-string nested-quote syntax error in `montage.__repr__` | `fix/critical-bugs-phase1a` |
| KI-027 | Missing `scipy.stats` import | `fix/critical-bugs-phase1a` |
| KI-028 | `hasher.fill()` broken signature (dead code) | `fix/hasher-branch-correctness` (deleted) |
| M1 | `standardize_name` / `_is_oxygenated` crashed on short/non-str inputs | `fix/input-validation` + `fix/oxygenation-guard` |
| M2 | Missing `lmbda≤0`, `duration≤0`, empty-events validation | `fix/input-validation` |
| M3 | `load_montage` cryptic `KeyError` on malformed JSON | `fix/input-validation` |
| M4 | `estimate_activity` left orphaned channel entries after drop | `fix/state-lifecycle` |
| M5 | `load_montage` partial-failure could return half-populated montage | `fix/state-lifecycle` |
| M6 | `configure()` committed state before `_merge_montages` could fail | `fix/tree-delete-filter` (moved from `fix/state-lifecycle` because it needed working `tree.delete`) |
| L1 | "Cannonical" → "Canonical" (4 occurrences) | `fix/observer-and-typos` |
| L2 | "intiialized" → "initialized" | `fix/observer-and-typos` |
| L3 | "ommited" → "omitted" | `fix/observer-and-typos` |

### Newly discovered, also resolved

| ID | Summary | Fixed in |
|----|---------|----------|
| ND-001 | `filter()` threshold comparison inverted (`>` → `<`) | `fix/critical-bugs-phase1a` |
| ND-002 | `compare_context` denominator (same as KI-011) | `fix/critical-bugs-phase1bc` |

---

## OPEN — scheduled for v1.2.0

### `experiment/ppf-validation` (S1)

| File:Line | Issue |
|-----------|-------|
| `hrfunc.py` `preprocess_fnirs` | `ppf=0.1` vs MNE default `ppf=6.0`. Likely a bug from early exploration, but Denny will validate experimentally on a real dataset before deciding. Blocks `release/1.2.0`. |

### `release/1.2.0`

Packaging and release mechanics:
- `pyproject.toml` dependency lower-bound pins
- `pyproject.toml` `[tool.setuptools.package-data]` for `hrfs/*.json`
- `MANIFEST.in` sdist inclusion
- Version bump to `1.2.0`
- CHANGELOG.md entry
- Tag + PyPI upload

---

## OPEN — scheduled for v2.0.0

### Architectural refactors

| ID | File | Issue | v2.0.0 branch |
|----|------|-------|----------------|
| KI-024 | `hrfunc.py` | `class montage(tree)` inheritance clash with `hbo_tree`/`hbr_tree` composition | `refactor/composition` |
| — | `hrfunc.py` `estimate_activity` | Return type is `mne.io.Raw` with `hbo`/`hbr` channel types, but data is deconvolved neural activity — type-dispatched downstream tools silently treat it as haemo. Introduce `NeuralActivity` wrapper. | `refactor/neural-activity-output-type` |
| KI-031 | All modules | No logging infrastructure — `print()` everywhere | `feat/logging` |
| — | All modules | No type hints; mypy cannot verify much structure | `feat/type-hints` |
| — | All modules | Magic numbers throughout (`0.01`, `0.128`, `7.81`, `30.0`, `0.1`, `0.2`, `1e-3`, `1e-4`) | `feat/constants` |
| KI-032 | All modules | Class names don't follow PascalCase (`montage`, `tree`, `lens`, `hasher`) | `refactor/composition` (pick up naming while we're in there) |
| KI-033 | `tests/test_estimation.py`, `tests/test_localization.py` | Module-level code crashes pytest on collection | `feat/test-suite-restructure` |
| KI-034 | `tests/test_hashing.py` | Empty test file | `feat/test-suite-restructure` |
| KI-022 | `hrfunc.py` `correlate_hrf` | Hardcoded `correlation_matrix.json` write to CWD regardless of working directory | `refactor/shared-helpers` |
| — | `hrtree.py` `tree.insert` | Absolute `1e-10` jitter loses relative precision at large coordinate scales (flagged in `fix/tree-hrf-correctness` data-integrity review) | v2.0.0 magnitude-relative jitter |
| KI-021 | `hrtree.py` `HRF.copy()` | `dict(self.context)` shallow copy — nested lists/dicts in context still shared across branched trees | `refactor/shared-helpers` |

### Performance

| ID | Issue | v2.0.0 branch |
|----|-------|----------------|
| — | `estimate_hrf` solves one channel at a time; batch solve via `np.linalg.solve` gives ~20–40× speedup | `perf/estimate-hrf-batch` |
| — | `estimate_activity` solves one channel at a time; parallel solve with `ThreadPoolExecutor(max_workers=N)` gives ~N× speedup | `perf/estimate-activity-parallel` |

### Nice-to-haves

| ID | Issue | v2.0.0 branch |
|----|-------|----------------|
| — | `estimate_hrf` silently drops events that fall outside the scan timeframe after edge expansion; should count and report | `feat/event-edge-warning` |
| — | Nicer error messages for JSON load/save edge cases | `feat/robustness-io` |
| — | Adaptive `lmbda` scaling with matrix norm | `feat/adaptive-regularization` (needs design work) |
| KI-029 | No dependency version pins in `pyproject.toml` | `release/1.2.0` (moved from v2.0.0 — needed for PyPI release) |
| KI-030 | Bundled HRF JSON not declared in `[tool.setuptools.package-data]` | `release/1.2.0` |

### Documentation

| ID | Issue | v2.0.0 branch |
|----|-------|----------------|
| KI-023 | README code examples had syntax errors and typos | Fixed directly in a README PR (or bundle into release/1.2.0) |

---

## Issue ID key

| Prefix | Source |
|--------|--------|
| `KI-NNN` | Original "known issues" catalog from the first code review (2026-04-13) |
| `NE-NNN` | "Newly discovered" during the 2026-04-13 code review |
| `ND-NNN` | Non-deterministic / threading issues discovered during `fix/estimate-activity-threading` scoping |
| `S1–S4` | Scientific correctness issues from the 2026-04-14 parallel review |
| `H1–H4` | High-priority issues from the same review |
| `M1–M6` | Medium issues from the same review |
| `L1–L3` | Low-priority typos from the same review |

All issue IDs are stable once assigned. Fixed issues are preserved in this file for traceability; do not delete them.
