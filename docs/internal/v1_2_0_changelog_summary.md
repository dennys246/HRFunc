# HRFunc v1.2.0 — Change Summary

**Release type:** Correctness release (minor version bump 1.1.2 → 1.2.0)
**Strategy:** Fix every crash, silent-wrong-result, and input-validation hole without changing architecture. See [../plans/plan_v1_2_0.md](../plans/plan_v1_2_0.md) for the detailed plan.

This document is the authoritative per-branch change summary for use when writing the CHANGELOG.md entry, PyPI release notes, and GitHub release.

---

## Branch chain (in dependency order)

1. `fix/critical-bugs-phase1a`
2. `fix/critical-bugs-phase1bc`
3. `refactor/circular-imports-phase2`
4. `fix/estimate-activity-threading`
5. `fix/input-validation`
6. `fix/state-lifecycle`
7. `fix/oxygenation-guard` (mini-branch)
8. `fix/tree-delete-filter`
9. `fix/hasher-branch-correctness`
10. `fix/tree-hrf-correctness`
11. `fix/canonical-hrf-sfreq`
12. `fix/tree-edge-cases`
13. `fix/observer-and-typos`
14. `experiment/ppf-validation` *(pending Denny's experimental validation)*
15. `release/1.2.0` *(pending)*

---

## What changed, grouped by theme

### Bug fixes — pipeline correctness

- **Pipeline ordering** in `estimate_hrf`: data loaded AFTER preprocessing instead of before, so deconvolution runs on HbO/HbR data instead of raw intensity (1.8)
- **`preprocess_fnirs` return capture** in `estimate_activity`: the preprocessed haemo object is now assigned back, preventing deconvolution on unprocessed data (1.9)
- **`load_montage` tree overwrite**: removed the duplicate `hbo_tree = tree(...)` / `hbr_tree = tree(...)` re-init at the end of load_montage that was discarding all just-inserted user HRFs (1.10)
- **`preprocess_fnirs` subject_info guard**: no longer crashes on `raw_od.info['subject_info']['his_id']` when subject_info is None (1.11)
- **`_is_oxygenated` fallthrough**: unrecognized channel names now raise a clean `ValueError` instead of falling through silently (1.12)
- **`filter()` threshold comparison**: inverted `>` to `<` so nodes BELOW the similarity threshold are deleted instead of nodes ABOVE it (ND-001)
- **`compare_context` denominator**: divides by `len(values)` (number of values for the current key) instead of `len(first_context)` (total keys) so multi-value contexts aren't underweighted (ND-002)
- **`estimate_activity` threading scope**: `nonlocal success` declaration added inside the deconvolution closure so the outer drop-channel check actually sees the closure's write (ND-003)
- **`estimate_activity` timeout default**: changed from `1500` (seconds = 25 minutes) to `30` seconds (ND-004)
- **`estimate_activity` return value**: returns `nirx_obj` (or `None` when all channels were bad) instead of implicit `None` (1.16)
- **`polynomial_detrend` order**: reverted from 1 back to 3 to match documentation (Q2 resolved)

### Bug fixes — data structures

- **`hasher.probe_count`** initialized in `__init__` — first probe call no longer AttributeErrors (KI-008, 1.3)
- **`hasher.contexts`** now uses `[[] for _ in range(capacity)]` instead of `[[]] * capacity` in three locations (init, `fill`, `resize`) so slots don't share list references (KI-003, 1.14)
- **`hasher.__repr__`** ZeroDivisionError guard when `size == 0` (KI-012, 1.15)
- **`hasher.search`** cycle detection stops the probe loop at capacity steps instead of infinite-looping on a full table (KI-014, 1.17)
- **`hasher.add`** now appends pointers to a per-slot list with identity deduplication (3.3)
- **`hasher.search`** returns `list[pointer]` (empty on miss) instead of scalar-or-False (H4)
- **`hasher.fill` / `hasher.double_check`** deleted as broken / dead code (3.4)
- **`tree._delete_recursive`** rewritten with a new `_copy_payload` helper. Pre-fix had a 5-arg-to-3-arg recursive call mismatch and referenced the non-existent `node.hrf_data` attribute. (KI-009 / H2)
- **`_copy_payload`** copies all HRF payload fields except `left`/`right` children, `np.copy`s trace/trace_std to avoid aliasing, copies the `built` flag, and intentionally does NOT copy `hrf_processes`/`process_names`/`process_options` (bound methods that would cross-reference `src` into `dst`)
- **`tree.gather(None)`** returns `{}` instead of crashing on `node.left` access (3.8)
- **`tree.merge`** inserts `node.copy()` so the merged tree's nodes are independent from the source; recursion still walks the SOURCE's children to traverse the full subtree (NE-006)
- **`tree.nearest_neighbor`** explicit `if self.root is None: return None, float("inf")` early return, plus tightened `== None` / `if best:` to `is None` / `is not None` (NE-007)
- **`tree.insert`** jitter branch directly mutates `hrf.x/y/z` instead of a loop variable; refreshes `h_val` for the current axis routing (3.5, KI-013)
- **`tree.insert`** first-insert path no longer stamps `context['method'] = 'canonical'` on the user's first HRF (NE-001)
- **`tree.insert`** no longer creates an eager canonical sentinel at `root.right` (S4 — see "canonical HRF" section below)
- **`HRF.__init__`** mutable default args (`estimates=[], locations=[], context=[]`) replaced with `None` sentinels and per-instance materialization (3.7, KI-002)
- **`HRF.__init__`** `process_options = [None]` so `build()`'s zip produces one iteration (pre-fix was `[]` causing zero iterations) (NE-003)
- **`HRF.build`** derives `hrf_type` from `self.oxygenation` instead of reading the never-set `self.type` (3.10, KI-018)
- **`HRF.spline_interp`** accepts an optional trace argument so `build()` can call it via the `process(self.trace)` pipeline pattern (NE-003 complement)
- **`HRF.smooth`** imports `scipy.ndimage.gaussian_filter1d` at module level and calls it as a free function instead of the non-existent `self.gaussian_filter1d` (NE-004)

### Bug fixes — API surface

- **`montage.__repr__`** safe on unconfigured instances via `getattr` fallback; reports configured vs unconfigured state (H3)
- **`montage.estimate_activity`** zero-trace HRF fallback: detects `None`, empty, or all-zero traces and falls back to the canonical HRF with a warning instead of silently producing NaN (H1)
- **`montage.estimate_activity`** drops orphaned channel entries from `self.channels` / `hbo_channels` / `hbr_channels` after a failed channel drop, so downstream iterators (`correlate_hrf`, `generate_distribution`) don't trip on stale pointers (M4)
- **`montage.estimate_activity`** deconvolution closure catches generic `Exception` (not just `TimeoutError`) so every solve failure routes through the drop-and-cleanup path
- **`montage.configure`** transactional: commits scalar/list state and `_merge_montages` result atomically. First-time configure rolls back via `self.root = None`; re-configure rolls back via `tree.delete` on newly-inserted nodes (M6 — moved here from `fix/state-lifecycle` to sit on top of working `tree.delete`)
- **`montage.correlate_canonical`** raises a clean `ValueError` on unconfigured montage instead of AttributeError on `self.root.trace` (lint-sweep finding)
- **`load_montage`** per-entry schema validation with `try/except` wrapping — raises `ValueError` naming the offending entry and field, preserves original exception via `__cause__`, and never returns a half-populated montage (M3, M5)
- **`load_montage`** passes `channel['context']` to the `HRF` constructor so loaded nodes carry their actual metadata. Pre-NE-002 this was hidden because the hasher was populated by dict keys; post-NE-002 the regression would have caused silent compare_context / branch / filter mismatches. Caught by the cross-branch audit.
- **`estimate_hrf`** rejects `duration <= 0`, non-list `events`, empty events, and `lmbda <= 0` at the top of the function (M2a, M2b)
- **`estimate_activity`** rejects `lmbda <= 0` at the top of the function (M2b)

### Bug fixes — helpers

- **`_utils.standardize_name`** entry guard raises `TypeError` on non-str and `ValueError` on strings shorter than 3 characters (M1)
- **`_utils._is_oxygenated`** matching entry guard plus a structure guard for the `ch_name[-2] == 'b'` branch — pre-fix strings like `'abc'` / `'abb'` passed the length check but crashed `split[1][0]` with IndexError (mini-branch)

### Bug fixes — canonical HRF

- **S4**: canonical HRFs are now lazily generated at the calling scan's actual sample rate via a new `tree.get_canonical_hrf(oxygenation, sfreq, duration)` helper with a per-`(ox, sfreq, duration)` cache. Pre-fix the canonical was hardcoded to `t_r=0.128` (7.81 Hz) and eagerly constructed during the first `tree.insert`, which meant scans at any other sample rate got a kernel of the wrong length and downstream deconvolution math silently produced wrong results.
- `tree.insert` no longer creates a canonical sentinel at `root.right`. The tree is a pure kd-tree of user HRFs.
- `tree.nearest_neighbor` fallback returns `(None, inf)` on miss instead of `(root.right, inf)`. Callers handle `None` explicitly by calling `get_canonical_hrf` with their own sfreq.
- `montage.estimate_activity`, `montage.localize_hrfs`, and `montage._merge_montages` all updated to the new pattern.

### Bug fixes — lint sweep (v1.2.0 in-scope)

These were caught by a targeted `ruff` + `mypy` pass during `fix/observer-and-typos` and fixed in the same branch:

- **`lens.calc_snr` PSD variable swap**: pre-fix `psd_noise_slow = preproc_noise_FAST.compute_psd(slow_band)` — the filter had removed all energy from the requested band, so `noise_power_slow` and `noise_power_fast` were both near zero and `SNR = signal / ~0` came out effectively infinite. Silent wrong scientific result.
- **`lens.calc_snr` mutable default**: `noise_bands=[(0.0, 0.01), (0.1, 0.5)]` → `None` sentinel + per-call materialization
- **`montage.correlate_canonical` None-root guard**: raises clean `ValueError` on unconfigured montage instead of AttributeError
- **`lens.__init__`** initializes `self.sfreq` and `self.channels` so direct `calc_snr` calls don't AttributeError (3.9)

### Bug fixes — infrastructure

- **f-string syntax error** in `montage.__repr__`: nested quotes inside f-string was invalid Python < 3.12, extracted to a local variable (1.1, KI-026)
- **Missing `scipy.stats` import** added to `hrfunc.py` (1.2, KI-027)
- **`return ValueError` → `raise ValueError`** in multiple validators (1.4, KI-004)
- **Bare `except:` → specific exception types** in `_is_oxygenated` wavelength parsing (1.5)
- **`double_probe(key, hashkey, False)`** invalid third positional arg removed — `False` was being passed as the linear probe coefficient `a`, clustering every probe to slot 1 (1.6, KI-016)
- **"Configureding" → "Configuring"** (1.7)
- **"Cannonical" → "Canonical"** in correlate_canonical plot labels (4 occurrences) (L1)
- **"intiialized" → "initialized"** in tree init warning (L2)
- **"ommited" → "omitted"** in estimate_hrf edge-expansion warning (L3)

### Refactor — circular imports

- New `src/hrfunc/_utils.py` holds `standardize_name`, `_is_oxygenated`, and `_LIB_DIR`
- `hrfunc.py` no longer imports itself (was importing `hrfunc.__file__` for the library path)
- `hrtree.py` no longer imports from `hrfunc.py`
- Breaks the circular import that previously only worked by accident of function-body references instead of module-level references

---

## Architecture state at 1.2.0

At the end of the correctness release the library is structurally the same as 1.1.2 — no API rename, no composition refactor, no new top-level modules except `_utils.py`. All v2.0.0-class architectural concerns (composition, type hints, logging, magic number extraction, test suite restructure, performance optimizations) were deliberately deferred.

**What is different:**
- Every reachable user-facing code path either works correctly or raises a clean exception naming the problem
- All documented pytest targets pass (213 tests, 0 xfailed) after the full in-flight stack
- All bundled test fixtures survive a `load_montage` round-trip
- `tree.filter` + `tree.delete` + `tree.branch` work end-to-end for the first time
- Canonical HRFs match the calling scan's actual sample rate
- Input validation fires at API boundaries with `ValueError` / `TypeError` naming the offending parameter
- Malformed JSON files fail load with a clear error and don't leave partial state behind
- Mutable default args across the codebase (HRF, lens) no longer leak state between instances
- Observer SNR metric is no longer silently infinite

**What is NOT different:**
- `montage` still inherits from `tree` (composition refactor is v2.0.0)
- Channel solves are still sequential (parallelization is v2.0.0)
- Logging is still `print()` (structured logging is v2.0.0)
- No type hints (v2.0.0)
- Magic numbers are still scattered in the code (v2.0.0)
- `experiment/ppf-validation` is still open — `ppf=0.1` vs `ppf=6.0` in `preprocess_fnirs` needs Denny's experimental test on real data before release

---

## Testing

Each branch added its own `tests/test_<branch>.py` file. The cumulative targeted gate across the full in-flight stack is:

```bash
pytest \
  tests/test_phase1a.py \
  tests/test_phase1bc.py \
  tests/test_phase2.py \
  tests/test_threading.py \
  tests/test_input_validation.py \
  tests/test_state_lifecycle.py \
  tests/test_oxygenation_guard.py \
  tests/test_tree_delete_filter.py \
  tests/test_hasher_branch.py \
  tests/test_tree_hrf.py \
  tests/test_canonical_hrf.py \
  tests/test_tree_edge_cases.py \
  tests/test_observer_and_typos.py \
  -q
```

Result: **213 passed, 0 xfailed**.

The integration tests `tests/test_estimation.py` and `tests/test_localization.py` are still excluded because their module-level code crashes pytest on collection (KI-033). Fixing that is scoped to `feat/test-suite-restructure` in v2.0.0.

---

## Release gate for `release/1.2.0`

See [../plans/plan_v1_2_0.md](../plans/plan_v1_2_0.md) `release/1.2.0` section for the packaging checklist:

1. `pyproject.toml` dependency lower-bound pins (numpy, mne, mne_nirs, scipy, nilearn, matplotlib)
2. `pyproject.toml` `[tool.setuptools.package-data]` for `hrfs/*.json`
3. `MANIFEST.in` sdist inclusion
4. Version bump to `1.2.0`
5. CHANGELOG.md entry (use this file as the source)
6. `python -m build` + `twine check dist/*`
7. Wheel contents check: `unzip -l dist/hrfunc-1.2.0-py3-none-any.whl | grep hrfs`
8. Fresh-venv install test
9. Git tag + GitHub release
10. `twine upload dist/*`
