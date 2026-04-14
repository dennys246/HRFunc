# HRFunc Release Roadmap

**Current version:** 1.1.2
**Next release:** 1.2.0 — Correctness Release (PyPI)
**Release after that:** 2.0.0 — Architectural Refactor

**Last updated:** 2026-04-14

---

## Release Strategy

HRFunc is a scientific library that researchers rely on for publishable data. We split refactoring into two releases with distinct goals:

### **1.2.0 — Correctness Release**
Every user-facing path works without crashes, silent wrong results, or data corruption. No architectural changes. No breaking API changes. This is a **minor version bump** so existing users can upgrade without code changes and gain stability. Detailed plan: [plan_v1_2_0.md](plan_v1_2_0.md).

### **2.0.0 — Architectural Refactor**
Code cleanliness, performance, type safety, and breaking API improvements (notably the composition refactor that removes `montage(tree)` inheritance). This is a **major version bump** because `isinstance(m, tree)` behavior changes. Ships after 1.2.0 has been validated in Denny's own pipeline. Detailed plan: [plan_v2_0_0.md](plan_v2_0_0.md).

**Why this split:** Researchers need stability more urgently than cleanliness. A bug-fix release can ship in weeks; the composition refactor needs more design thought and a breaking-change release cycle.

---

## Working Practices (Apply to Every Branch)

Every branch in either release follows the review protocol defined in the "Working Practices for HRFunc Sessions" memory:

1. **Feature-branch-per-change** — each branch addresses one cohesive fix or refactor, stacks on the previous branch in the dependency chain
2. **Targeted unit tests** — add a `tests/test_<branch-name>.py` file exercising the specific behavior changed. Must run in under a few seconds and require no fNIRS data files
3. **Pre-PR review protocol** — always run two reviews before pushing:
   - Architecture & Execution Flow (primary)
   - A secondary lens chosen for the change type (data integrity, API surface, threading, etc.)
4. **Pre-PR test gate** — `python -m pytest tests/test_*.py -q` must pass (integration tests still deferred to v2.0.0 per KI-033)
5. **Editable install** — `pip install -e .` before running tests. Verify imports resolve to `src/hrfunc/__init__.py`, not site-packages
6. **Conventional commits** — branches named `fix/...`, `refactor/...`, `feat/...`, `perf/...`, `experiment/...`, `release/...`

---

## Status of Work Completed to Date (2026-04-14)

The v1.2.0 chain is nearly complete. Most branches are merged to main; the remaining four are pushed and stacked waiting for merge. The only un-started work is `experiment/ppf-validation` (waiting on Denny's experimental test) and `release/1.2.0` (packaging).

| Branch | Status | Scope |
|---|---|---|
| `fix/critical-bugs-phase1a` | ✅ Merged | Syntax errors, missing imports, return-vs-raise, filter() inversion (ND-001), double_probe args, typos |
| `fix/critical-bugs-phase1bc` | ✅ Merged | Pipeline ordering in estimate_hrf, preprocess_fnirs return capture, load_montage tree overwrite, subject_info guard, _is_oxygenated fallthrough, hasher probe_count/shared-list/repr/search, estimate_activity return, context similarity denominator (ND-002), polynomial_detrend order=3 |
| `refactor/circular-imports-phase2` | ✅ Merged | `_utils.py` created, `hrfunc.py` self-import removed, `hrtree.py` no longer imports `hrfunc`, `_LIB_DIR` constant |
| `fix/estimate-activity-threading` | ✅ Merged | ND-003 (success variable scope via `nonlocal`), ND-004 (default `timeout=30`) |
| `fix/input-validation` | ✅ Merged | M1 standardize_name guard, M2a/M2b duration/events/lmbda validation, M3 load_montage schema validation, H1 zero-trace canonical fallback |
| `fix/state-lifecycle` | ✅ Merged | H3 `__repr__` unconfigured guard, M4 drop-channel orphaning cleanup + deconvolution closure generic-exception catch, M5 `load_montage` partial-load wrapping. (M6 moved to `fix/tree-delete-filter`.) |
| `fix/oxygenation-guard` | ✅ Merged | Mini-branch. `_is_oxygenated` entry guards and `[-2]=='b'` structure guard preventing `split[1][0]` IndexError on short/unstructured inputs |
| `fix/tree-delete-filter` | ✅ Merged | KI-009 `_delete_recursive` rewrite with `_copy_payload` helper, 3.8 `gather(None)` guard, `test_filter_removes_low_similarity_nodes` xfail → pass, M6 configure commit-on-success with tree rollback |
| `fix/hasher-branch-correctness` | ⏳ Pushed | 3.3/H4 hasher list contract, NE-002 populate by values, 3.1 `compare_context` scalar wrap, 3.2 `tree.branch` AND-on-kwargs, dead code removal, filter no-op cleanup |
| `fix/tree-hrf-correctness` | ⏳ Pushed | NE-001 canonical label, 3.5 jitter, NE-003 process_options + spline_interp signature, NE-004 gaussian_filter1d import, 3.7 mutable defaults, 3.10 `hrf_type` |
| `fix/canonical-hrf-sfreq` | ⏳ Pushed | S4 lazy `tree.get_canonical_hrf` with cache, no more eager canonical sentinel, callers updated |
| `fix/tree-edge-cases` | ⏳ Pushed | NE-006 `merge` deep copy, NE-007 empty-tree early return, cross-branch fix for `load_montage` dropping `channel['context']` |
| `fix/observer-and-typos` | ⏳ Pushed | 3.9 `lens.sfreq` init, L1-L3 typos, lint-sweep bonuses (`calc_snr` PSD swap, `noise_bands` mutable default, `correlate_canonical` None-root guard) |

Full gate after the complete stack: **213 passed, 0 xfailed**.

---

## v1.2.0 Branch Decomposition

Full detail and sub-issue mapping in [plan_v1_2_0.md](plan_v1_2_0.md). Dependency chain (each stacks on the previous):

```
main
├── fix/critical-bugs-phase1a                    ✅ pushed
├── fix/critical-bugs-phase1bc                   ✅ pushed
├── refactor/circular-imports-phase2             ✅ pushed
├── fix/estimate-activity-threading              ✅ pushed
├── fix/input-validation                         ⏳ next
│   └── fix/state-lifecycle
│       └── fix/tree-delete-filter
│           └── fix/hasher-branch-correctness
│               └── fix/tree-hrf-correctness
│                   └── fix/canonical-hrf-sfreq
│                       └── fix/tree-edge-cases
│                           └── fix/observer-and-typos
│                               └── experiment/ppf-validation
│                                   └── release/1.2.0
```

### v1.2.0 Branch Summary

See [plan_v1_2_0.md](plan_v1_2_0.md) for detailed per-branch scope and the Completed Branches section for what each one actually landed. The two remaining branches are:

| Branch | Addresses | Depends on |
|---|---|---|
| `experiment/ppf-validation` | **S1**: `ppf=0.1` vs MNE default `ppf=6.0` in `preprocess_fnirs`. Denny tests both values experimentally on a real dataset and decides which is scientifically defensible before release. | observer-and-typos |
| `release/1.2.0` | Phase 11 packaging (`package-data` for `hrfs/*.json`, dependency version pins in `pyproject.toml`, `MANIFEST.in`), CHANGELOG, version bump to 1.2.0, tag, PyPI upload | ppf-validation |

All of the correctness-fix branches that preceded these two are complete (merged or pushed pending merge — see the status table above).

---

## v2.0.0 Branch Decomposition

Full detail in [plan_v2_0_0.md](plan_v2_0_0.md). These branches have fewer hard dependencies and can be sequenced more flexibly after 1.2.0 ships.

| Branch | Scope | Notes |
|---|---|---|
| `refactor/composition` | Phase 4: `montage` → composition (`_tree` attribute, delegation methods); fixes ND-005 global HRFs | Breaking change; needs Denny's explicit sign-off |
| `perf/estimate-hrf-batch` | Batch-solve all channels in one `np.linalg.solve` call (20–40× speedup) | Low risk, high payoff; doable as early 2.0 work |
| `refactor/shared-helpers` | Phase 8 + ND-006: extract `_build_toeplitz`, `_solve_regularized` | Unblocks further parallelism |
| `perf/estimate-activity-parallel` | Multi-worker ThreadPool for per-channel deconvolution | Depends on shared-helpers |
| `feat/logging` | Phase 5: replace `print()` with structured logging | Helps debug everything downstream |
| `feat/type-hints` | Phase 7: type annotations, `from __future__ import annotations` | Low risk cleanup |
| `feat/constants` | Phase 9: magic numbers → named constants | Low risk cleanup |
| `feat/test-suite-restructure` | Phase 12: fix KI-033 (integration tests crash pytest collection), unify targeted + integration suites | Final gate for 2.0.0 |
| `feat/robustness-io` | Phase 6 spillover: nicer error messages for JSON load/save edge cases | Nice-to-have |
| `feat/event-edge-warning` | Count and report silently dropped events in `estimate_hrf` edge expansion | Correctness-adjacent |
| `feat/adaptive-regularization` | Investigate `lmbda` scaling with matrix norm | Design question, needs Denny's judgment |

---

## Testing Gate for v1.2.0 Release

Before tagging 1.2.0 and pushing to PyPI, verify:

- [ ] All targeted test files pass: `pytest tests/test_phase1a.py tests/test_phase1bc.py tests/test_phase2.py tests/test_threading.py tests/test_input_validation.py tests/test_state_lifecycle.py tests/test_tree_delete_filter.py tests/test_hasher_branch.py tests/test_tree_hrf.py tests/test_canonical_hrf.py tests/test_tree_edge_cases.py tests/test_observer_typos.py`
- [ ] `import hrfunc` works cleanly with no warnings
- [ ] `from hrfunc import montage, tree, load_montage, localize_hrfs, preprocess_fnirs` all resolve
- [ ] `montage()` → `montage.estimate_hrf(...)` → `montage.estimate_activity(...)` → `montage.save(...)` round-trips on bundled test data
- [ ] `load_montage(bundled_hrfs_path)` succeeds without partial-state or tree overwrite
- [ ] `tree.filter()` remove-path no longer xfailed (KI-009 unblocked)
- [ ] Built wheel includes `src/hrfunc/hrfs/*.json` (`python -m build && unzip -l dist/*.whl | grep hrfs`)
- [ ] Denny validates experimentally on real data whether `ppf=0.1` or `ppf=6.0` is correct (S1 decision) before tagging
- [ ] Version bumped in `pyproject.toml` to `1.2.0`
- [ ] `CHANGELOG.md` updated with all bug fixes, grouped by category
- [ ] `python -m build` succeeds; `twine check dist/*` passes
- [ ] Tag `v1.2.0`, push, GitHub release notes, `twine upload dist/*`

---

## Testing Gate for v2.0.0 Release

Before tagging 2.0.0, in addition to all 1.2.0 gates:

- [ ] `refactor/composition` merged, `isinstance(m, tree)` removed (documented in CHANGELOG as breaking change)
- [ ] Integration test suite (KI-033) restructured and runs in CI
- [ ] `perf/estimate-hrf-batch` benchmarked against 1.2.0 baseline — should demonstrate ~20× speedup on a realistic fNIRS dataset
- [ ] `feat/logging` replaces all `print()` calls; `logging.basicConfig(level=logging.WARNING)` silences routine output
- [ ] Type hints pass `mypy --strict` on `src/hrfunc/`
- [ ] Migration guide written for users upgrading from 1.x to 2.0
- [ ] Version bumped to `2.0.0`
- [ ] Tag, release notes, upload

---

## Related Documents

- [plan_v1_2_0.md](plan_v1_2_0.md) — detailed v1.2.0 branch-by-branch work
- [plan_v2_0_0.md](plan_v2_0_0.md) — detailed v2.0.0 branch-by-branch work
- [../internal/known_issues.md](../internal/known_issues.md) — authoritative issue catalogue (KI-001 through KI-034, NE-001 through NE-007, ND-001 through ND-006, S1-S4, H1-H4, M1-M6, L1-L3)
- [../internal/architecture.md](../internal/architecture.md) — module map and execution paths