# HRFunc v1.2.0 — Correctness Release Plan

**Release type:** Minor version bump (1.1.2 → 1.2.0)
**Goal:** Every user-facing path works without crashes, silent wrong results, or data corruption. No breaking API changes. No architectural refactors. Ship to PyPI.

**High-level roadmap:** [phase_breakdown.md](phase_breakdown.md)
**Architectural refactors:** [plan_v2_0_0.md](plan_v2_0_0.md) (post-1.2.0)

---

## Scope Philosophy

**IN scope for 1.2.0:**
- Crashes on valid user inputs
- Silent wrong results (e.g. canonical HRF at wrong sample rate, filter() inverted)
- Missing input validation that allows degenerate states
- Packaging bugs that make `pip install hrfunc` unusable
- Obvious typos in user-visible output

**OUT of scope for 1.2.0 (→ 2.0.0):**
- Architectural refactors (composition, shared helpers, inheritance untangle)
- Performance optimizations
- Type hints
- Structured logging
- Magic-number extraction
- Test suite restructuring
- Breaking API changes

**Rule of thumb:** if a fix changes observable behavior for a scientist who's already using HRFunc correctly, it belongs in 2.0.0 unless the current behavior is scientifically wrong.

---

## Completed Branches (2026-04-13 → 2026-04-14)

These are pushed to origin and pending merge to main:

### `fix/critical-bugs-phase1a` ✅
- **1.1** f-string syntax error in `montage.__repr__` (extracted `context_str` variable)
- **1.2** Added `scipy.stats` import
- **1.4** `return ValueError` → `raise ValueError` in estimate_hrf input validation
- **1.5** Narrowed bare `except:` and added proper `raise LookupError` in `_is_oxygenated`
- **1.6** Removed invalid `False` positional arg in `double_probe` → `quad_probe` call
- **1.7** Typo fix: "Configureding" → "Configuring"
- **ND-001** `filter()` threshold comparison inverted (`>` → `<`)
- **Tests:** `tests/test_phase1a.py` (22 passed, 1 xfailed on filter remove path)

### `fix/critical-bugs-phase1bc` ✅
- **1.8** `estimate_hrf`: moved `load_data()`/`get_data()` to AFTER `preprocess_fnirs` so deconvolution runs on HbO/HbR data
- **1.9** `estimate_activity`: captured `preprocess_fnirs` return value, added None guard
- **1.10** `load_montage`: removed duplicate tree re-init that was wiping inserted user HRFs (init happens in `montage.__init__`)
- **1.11** `preprocess_fnirs`: guarded `subject_info=None` before accessing `his_id`
- **1.12** `_is_oxygenated`: final else-raise for unrecognized channel names
- **1.3** `hasher.__init__`: initialized `probe_count = 0`
- **1.13** `compare_context`: added `context_weights=None` parameter
- **1.14** `hrhash.py`: `[[]]*n` → `[[] for _ in range(n)]` in three locations
- **1.15** `hasher.__repr__`: ZeroDivisionError guard when size==0
- **1.16** `estimate_activity`: added `return nirx_obj`
- **1.17** `hasher.search`: cycle detection prevents infinite loop on full tables
- **ND-002** `compare_context`: denominator corrected from `len(first_context)` to `len(values)`, empty-list guard
- **Poly order** `polynomial_detrend`: reverted from 1 back to 3 (Q2 resolved)
- **Tests:** `tests/test_phase1bc.py` (23 passed)

### `refactor/circular-imports-phase2` ✅
- New `src/hrfunc/_utils.py` with `standardize_name`, `_is_oxygenated`, `_LIB_DIR`
- `hrfunc.py`: removed `import hrfunc` self-import, replaced `hrfunc.__file__` with `_LIB_DIR`, renamed local variable in `load_montage` from `montage` → `_montage` to avoid `UnboundLocalError`, deleted duplicated helper definitions
- `hrtree.py`: removed `from . import hrfunc`, imports helpers from `_utils`
- **Tests:** `tests/test_phase2.py` (12 passed)

### `fix/estimate-activity-threading` ✅
- **ND-003** Added `nonlocal success` declaration to deconvolution closure; hoisted `success = None` to estimate_activity scope before closure definition
- **ND-004** Changed default `timeout=1500` → `timeout=30` (Option A from Q1 tradeoff analysis)
- Cleanup: `success == False` → `success is False`; TimeoutError print message now includes timeout value
- **Tests:** `tests/test_threading.py` (9 passed)

**Full suite after all four merged branches:** 66 passed, 1 xfailed (unblocked later by `fix/tree-delete-filter`)

---

## Remaining Branches for 1.2.0

Each branch stacks on the previous. Each gets its own targeted test file and the standard two-agent pre-PR review.

---

### `fix/input-validation`

**Depends on:** `fix/estimate-activity-threading`
**Goal:** Reject degenerate inputs at API boundaries with clear errors instead of cryptic crashes deep in the math.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| M1 | `src/hrfunc/_utils.py` `standardize_name` | Add length check: if `len(ch_name) < 3`, raise `ValueError` with a clear message. Current `ch_name[:-3]` crashes with `IndexError` on short inputs. |
| M2a | `src/hrfunc/hrfunc.py` `montage.estimate_hrf` | After existing type checks, add `if duration <= 0: raise ValueError("duration must be > 0")` and `if len(events) == 0: raise ValueError("events list must not be empty")`. |
| M2b | `src/hrfunc/hrfunc.py` `montage.estimate_hrf` and `estimate_activity` | Add `if lmbda <= 0: raise ValueError("lmbda must be > 0 for Tikhonov regularization")` at top of both methods. |
| M3 | `src/hrfunc/hrfunc.py` `load_montage` | Wrap JSON-schema access (`channel['hrf_mean']`, `channel['sfreq']`, `channel['context']['duration']`, `channel['location']`) in a per-entry try/except that raises a clean `ValueError` with the offending key name when fields are missing. Do NOT proceed with partial load — raise and let the caller retry. |
| H1 | `src/hrfunc/hrfunc.py` `montage.estimate_activity` deconvolution closure | Before `hrf_kernel = hrf.trace / np.max(np.abs(hrf.trace))`, check `if hrf.trace is None or len(hrf.trace) == 0 or np.max(np.abs(hrf.trace)) == 0`: fall back to canonical HRF (same code path as `hrf_model == 'canonical'`) and emit a warning. Currently produces silent NaN. |

**Tests:** `tests/test_input_validation.py` — boundary cases for each validator, confirm old crashes are now clean exceptions.

**Exit criteria:**
- All existing tests still pass
- New test file has ≥5 tests per fix above, each triggering the exact error path
- Error messages name the offending parameter clearly

---

### `fix/state-lifecycle`

**Depends on:** `fix/input-validation`
**Goal:** Montage and tree objects cannot land in a half-configured or inconsistent state after partial failures.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| H3 | `src/hrfunc/hrfunc.py` `montage.__repr__` | Guard against unconfigured state. Currently references `self.sfreq`, `self.hbo_channels`, `self.hbr_channels` which are only set by `configure()`. Use `getattr(self, 'sfreq', None)` and similar; produce a useful repr for both configured and unconfigured montages. |
| M4 | `src/hrfunc/hrfunc.py` `montage.estimate_activity` | When `success is False` and `drop_channels` is called, also pop the entry from `self.channels[ch_name]` and from the `hbo_channels` / `hbr_channels` lists. The tree cleanup (also removing the node from `hbo_tree` / `hbr_tree`) is deferred to `fix/tree-delete-filter` because `tree.delete` is broken. Also broadens the deconvolution closure exception catch from `TimeoutError` only to `Exception` so any solve failure triggers the cleanup path. |
| M5 | `src/hrfunc/hrfunc.py` `load_montage` | Wrap the per-entry insertion loop so that if any entry raises, the entire `_montage` is discarded and the original exception re-raised with a clear message. Snapshot-and-restore is fine (keep a list of inserted entries, clear the trees on failure). |

**Note on M6:** The configure() commit-on-success fix was moved to `fix/tree-delete-filter`. M6 requires rolling back the spatial tree on a failed re-configure, which needs a working `tree.delete`. Since `tree.delete` (KI-009) is fixed in `fix/tree-delete-filter`, M6 will be implemented there on top of the working delete path. Shipping a partial M6 here would silently leak orphan nodes into the tree on re-configure failures — worse than no rollback.

**Tests:** `tests/test_state_lifecycle.py` — construct a montage, call each method with a failure-inducing input, verify state is either unchanged or cleanly re-raised.

**Exit criteria:**
- `print(montage())` works on unconfigured instances
- A `load_montage` on a JSON with one bad entry raises cleanly AND the returned montage is not half-populated (either ValueError before return or full montage)
- After a dropped channel in `estimate_activity`, calling `correlate_hrf` or `generate_distribution` does not iterate orphan entries

---

### `fix/tree-delete-filter`

**Depends on:** `fix/state-lifecycle`
**Goal:** Fix the `_delete_recursive` signature mismatch (KI-009 / H2) so `tree.delete()` and the `tree.filter()` remove path actually work. Unblocks the xfailed test from phase1a. Then land M6 on top of the working delete path.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| KI-009 / H2 | `src/hrfunc/hrtree.py` `_delete_recursive` | Current signature takes `(node, hrf, depth)` but recursive calls pass `(node.right, min_node.x, min_node.y, min_node.z, depth+1)` — 5 positional args to a 3-arg function. Rewrite to pass the HRF consistently. Also replace any reference to `node.hrf_data` with the correct attribute. |
| 3.6 | `src/hrfunc/hrtree.py` `tree.delete` | Ensure top-level `delete(hrf)` passes an HRF object to `_delete_recursive`, not coordinates. Verify `left → right` assignment is not swapped. |
| M6 | `src/hrfunc/hrfunc.py` `montage.configure` | Only commit `self.sfreq`, `self.hbo_channels`, `self.hbr_channels`, `self.channels`, `self.root` AFTER successful `_merge_montages()`. On failure, roll back scalar/list attributes AND undo the partial tree inserts via the now-working `tree.delete`. (Deferred here from `fix/state-lifecycle` because proper tree rollback requires a working `tree.delete`.) |
| xfail cleanup | `tests/test_phase1a.py::TestFilterInversion::test_filter_removes_low_similarity_nodes` | Remove the `@pytest.mark.xfail` decorator once delete works. The test should pass cleanly. |

**Tests:** `tests/test_tree_delete_filter.py` — insert several HRFs, delete by position, verify tree structure preserved for remaining nodes. Verify filter() remove-path works end-to-end. Add re-configure rollback test that inserts some channels before raising and verifies `self.root` / `self.channels` / scalar attrs all snap back.

**Exit criteria:**
- `tree.delete(hrf)` works for leaf, single-child, and two-child nodes
- The xfailed filter test becomes a normal passing test
- Deleting the canonical HRF doesn't corrupt the rest of the tree (or is explicitly prevented)
- A failed re-configure leaves every montage attribute AND the spatial tree in their pre-call state

---

### `fix/hasher-branch-correctness`

**Depends on:** `fix/tree-delete-filter`
**Goal:** Fix the `tree.branch()` path end-to-end — currently broken in three overlapping ways.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| NE-002 | `src/hrfunc/hrtree.py` `load_hrfs` and `tree.branch` | Currently `load_hrfs` adds context *keys* to the hasher while `branch` searches by context *values*. Fundamental mismatch. Change `load_hrfs` to add context VALUES as keys (e.g., add every task name, stimulus name, etc. that a user might search by). Keys map to lists of node pointers that match that value. |
| H4 | `src/hrfunc/hrhash.py` `hasher.search` | Currently returns `False` on miss. `tree.branch` does `for node in context_references:` which TypeErrors on `False`. Change `search` to return an empty list `[]` on miss. Update all existing callers. |
| 3.1 | `src/hrfunc/hrtree.py` `compare_context` | Accept scalar context values by auto-wrapping in a list. Skip `None` values cleanly. Already partially fixed in phase1bc but needs extension to handle scalar inputs. |
| 3.2 | `src/hrfunc/hrtree.py` `tree.branch` | Normalize context values before iterating. Remove any `branch.channels` references or wire `channels` through properly. Initialize the sub-tree fresh instead of reloading from disk. |
| 3.3 | `src/hrfunc/hrhash.py` `hasher.add` / `search` | Support multiple pointers per key: `add` appends to the list at that slot; `search` returns the full list. Dedupe on insertion. |
| 3.4 | `src/hrfunc/hrhash.py` `hasher.fill` | Either rewrite to accept `(key, pointer)` pairs consistently, or delete the method if it has no valid use case. |

**Tests:** `tests/test_hasher_branch.py` — add context values, search by value returns correct pointer list, branch() produces a valid sub-tree filtered to the requested context.

**Exit criteria:**
- `hasher.search('nonexistent')` returns `[]`, not `False`
- `tree.branch(task='flanker')` returns a populated tree when flanker nodes exist and an empty tree otherwise — no TypeErrors
- `montage.branch()` continues to work (it already does, but verify no regression from the hasher changes)

**Review note:** This branch has the highest blast radius of the 1.2.0 fixes because it changes a return-type contract (`hasher.search` False → []). All call sites must be audited.

---

### `fix/tree-hrf-correctness`

**Depends on:** `fix/hasher-branch-correctness`
**Goal:** Fix remaining correctness bugs in `tree.insert` and the `HRF` node class.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| NE-001 | `src/hrfunc/hrtree.py` `tree.insert` | Currently on first insert: `self.root.context['method'] = 'canonical'` — corrupts the FIRST inserted node's context (user data) instead of labeling the canonical HRF. Change to `self.root.right.context['method'] = 'canonical'`. |
| 3.5 | `src/hrfunc/hrtree.py` `tree.insert` jitter branch | `for val in (hrf.x, hrf.y, hrf.z): val += 1e-10` mutates a loop variable, not the HRF's coordinates. Rewrite to directly assign `hrf.x += 1e-10; hrf.y += 1e-10; hrf.z += 1e-10`. |
| NE-003 | `src/hrfunc/hrtree.py` `HRF.__init__` | `self.process_options = []` is zero-length while `hrf_processes`/`process_names` are length 1, so `zip()` in `build()` does zero iterations and spline_interp is never called. Initialize `process_options = [None]` so the zip produces one pair. |
| NE-004 | `src/hrfunc/hrtree.py` `HRF.smooth` | Calls `self.gaussian_filter1d(...)` which is neither imported nor defined. Import `from scipy.ndimage import gaussian_filter1d` at module top; change the call to `gaussian_filter1d(self.trace, a)`. |
| 3.7 | `src/hrfunc/hrtree.py` `HRF.__init__` | Replace mutable default args (`estimates=[]`, `locations=[]`, `context={}`) with `None` defaults and initialize inside `__init__`. Standard Python bug class. |
| 3.10 | `src/hrfunc/hrtree.py` `HRF.build` | References `self.type` which is never set. Replace with `hrf_type = "hbo" if self.oxygenation else "hbr"` and use that in the plot filename. |

**Tests:** `tests/test_tree_hrf.py` — test each fix in isolation. For NE-003 verify that after `build(new_sfreq)`, `self.trace` is actually resampled. For NE-001 verify that after inserting a node, its `context['method']` is NOT clobbered.

**Exit criteria:**
- `HRF.build(new_sfreq)` actually runs the spline_interp process and updates `self.trace` length
- `HRF.smooth(sigma)` no longer AttributeErrors
- Inserting a new node preserves its original context
- Two independent `HRF()` instances have independent `estimates` / `locations` / `context` lists (mutable defaults fixed)

---

### `fix/canonical-hrf-sfreq`

**Depends on:** `fix/tree-hrf-correctness`
**Goal:** **S4** — remove the hardcoded `t_r=0.128` canonical HRF. Generate on demand using the calling context's sfreq (Option B). Relies on `HRF.build()` now working after NE-003 fix.

**Design approach (Option B):**

1. Remove the eager canonical construction in `tree.insert()` when `root is None`. The first-inserted node becomes root without any sibling.
2. Add a new method `tree.get_canonical_hrf(oxygenation, sfreq, duration)` that:
   - Generates a fresh Glover HRF with `t_r = 1/sfreq, time_length=duration`
   - For HbR: negate trace
   - Wraps in an `HRF` node with sentinel location (keep `[359, 359, 359]` for the sentinel since it's out of realistic MNE meter-scale head coordinates)
   - Cache keyed on `(oxygenation, sfreq, duration)` to avoid regenerating on every channel
3. Update `nearest_neighbor` fallback path (hrtree.py:370 area) to call `get_canonical_hrf` with the tree's own `self.sfreq` / context duration.
4. Update `estimate_activity` canonical branch (hrfunc.py:424-427): instead of `hrf = self.hbo_tree.root.right`, call `hrf = self.hbo_tree.get_canonical_hrf(True, self.sfreq, self.context['duration'])`. Same for HbR.
5. Update `localize_hrfs` (hrfunc.py:190) to use the same helper rather than generating its own canonical inline — eliminates the duplicate generation path.
6. Update any existing references to `root.right` as the canonical HRF to go through the new helper.

**Fixes:**

| ID | Fix |
|---|---|
| S4 | Lazy canonical HRF generation per scan sfreq as described above |
| Dedup | Consolidate the two canonical generation sites (hrtree.py:156 hardcoded + hrfunc.py:190 correct) into one helper |

**Tests:** `tests/test_canonical_hrf.py` — generate canonical at 5 Hz, 7.81 Hz, 10 Hz, verify the trace length matches `sfreq * duration`. Verify HbR canonical is the negation of HbO canonical. Verify caching: two calls with the same `(oxygenation, sfreq, duration)` return the same object.

**Exit criteria:**
- No `t_r=0.128` hardcoded anywhere in the codebase
- `estimate_activity(scan, hrf_model='canonical')` on scans at any sfreq produces a canonical kernel matching that scan's sample rate
- `localize_hrfs` fallback uses the same canonical generator as `estimate_activity`
- No regression in existing tests

---

### `fix/tree-edge-cases`

**Depends on:** `fix/canonical-hrf-sfreq`
**Goal:** Edge-case crashes in tree traversal methods.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| NE-006 | `src/hrfunc/hrtree.py` `tree.merge` | Currently inserts original node references, so `left`/`right` pointers are shared between trees. Rewrite to insert `node.copy()` (HRF already has a `copy()` method). |
| NE-007 | `src/hrfunc/hrtree.py` `tree.nearest_neighbor` | Fallback path `return self.root.right, float("inf")` crashes when `self.root is None`. Add early return `if self.root is None: return None, float("inf")`. Update callers to handle None. |
| 3.8 | `src/hrfunc/hrtree.py` `tree.gather` | Add `if node is None: return {}` at top of method. Currently crashes accessing `node.left`. |

**Tests:** `tests/test_tree_edge_cases.py` — empty tree nearest_neighbor, gather(None), merge two disjoint trees and then mutate one to verify the other is unaffected.

**Exit criteria:**
- Empty-tree queries return cleanly instead of crashing
- Merged trees have fully independent node objects

---

### `fix/observer-and-typos`

**Depends on:** `fix/tree-edge-cases`
**Goal:** Small cleanup items that don't fit other branches.

**Fixes:**

| ID | File | Fix |
|---|---|---|
| 3.9 | `src/hrfunc/observer.py` `lens.__init__` | Add `sfreq` parameter (with a sensible default like 7.81) and `self.sfreq = sfreq`. Currently `self.sfreq` is referenced without being set. |
| L1 | `src/hrfunc/hrfunc.py` | "Cannonical" → "Canonical" (×3 in plot titles/labels around lines 590, 604, 606) |
| L2 | `src/hrfunc/hrtree.py:74` | "intiialized" → "initialized" |
| L3 | `src/hrfunc/hrfunc.py:280` | "ommited" → "omitted" |

**Tests:** `tests/test_observer_typos.py` — construct a `lens` without error; grep source for the typos and assert absence.

**Exit criteria:**
- `lens()` can be instantiated and used without AttributeError
- All known typos removed

---

### `experiment/ppf-validation`

**Depends on:** `fix/observer-and-typos`
**Goal:** **S1** — resolve the `ppf=0.1` vs `ppf=6.0` question by experimental test. Denny runs both values on real data and decides which is correct before release.

**Process:**

1. Create the branch with the change `ppf=0.1` → `ppf=6.0` in `preprocess_fnirs` at hrfunc.py:800.
2. Do NOT merge until Denny confirms the correct value.
3. Denny runs the preprocessing pipeline with both values on at least one real dataset and compares HRF amplitudes against expected fMRI BOLD-scale references.
4. Whichever value produces scientifically defensible outputs wins.
5. If `ppf=6.0` is correct: keep the branch as-is, merge.
6. If `ppf=0.1` was correct: close the branch, add a comment in `preprocess_fnirs` documenting WHY `0.1` is used and what the scaling correction relationship is (Denny mentioned he does scaling at critical points but couldn't remember if it relates to this).

**Exit criteria:**
- Denny has explicitly confirmed the correct ppf value
- If the value changed from the current code, a test exists verifying the new default
- An inline comment in `preprocess_fnirs` explains the choice regardless of which value wins

---

### `release/1.2.0`

**Depends on:** `experiment/ppf-validation` (resolved, one way or the other)
**Goal:** Ship 1.2.0 to PyPI.

**Work items (Phase 11 packaging + release mechanics):**

1. **`pyproject.toml` — pin dependency lower bounds:**
    ```toml
    dependencies = [
        "numpy>=1.20",
        "mne>=1.0",
        "mne_nirs>=0.4",
        "scipy>=1.7",
        "nilearn>=0.9",
        "matplotlib>=3.5",
    ]
    ```
    Leave upper bounds open to avoid over-constraining. Consider adding upper bounds only if we hit confirmed breakage with a future major version.

2. **`pyproject.toml` — include bundled HRFs in wheels:**
    ```toml
    [tool.setuptools.package-data]
    hrfunc = ["hrfs/*.json"]
    ```
    This is critical — without it the library is unusable after pip install. Denny reports his install works, but this guards against it silently breaking for others.

3. **`MANIFEST.in` — sdist inclusion:**
    ```
    recursive-include src/hrfunc/hrfs *.json
    include README.md
    include CHANGELOG.md
    include LICENSE
    ```

4. **`pyproject.toml` — bump version:**
    ```toml
    version = "1.2.0"
    ```

5. **`CHANGELOG.md` — document all fixes** grouped under:
    - **Bug fixes — pipeline correctness** (1.8, 1.9, 1.10, 1.11, 1.12, ND-001, ND-002, ND-003, S4, canonical HRF sfreq)
    - **Bug fixes — data structures** (1.3, 1.13, 1.14, 1.15, 1.17, NE-001, NE-002, NE-003, NE-004, NE-006, NE-007, KI-009, 3.1-3.10)
    - **Bug fixes — API surface** (1.16, H1, H3, M1-M6, input validation)
    - **Bug fixes — infrastructure** (1.1, 1.2, 1.4, 1.5, 1.6, 1.7, circular imports, threading, typos)
    - **Packaging** (package-data, dependency pins)
    - **Scientific correction** (polynomial_detrend order=3, S1 ppf decision with rationale)

6. **Verification steps:**
    ```bash
    python -m build
    twine check dist/*
    unzip -l dist/hrfunc-1.2.0-py3-none-any.whl | grep hrfs
    # Should list hrfs/hbo_hrfs.json and hrfs/hbr_hrfs.json
    ```
    Then in a fresh venv:
    ```bash
    pip install dist/hrfunc-1.2.0-py3-none-any.whl
    python -c "import hrfunc; m = hrfunc.montage(); print('OK')"
    python -c "import hrfunc; m = hrfunc.load_montage(hrfunc.__file__.replace('__init__.py', 'hrfs/hbo_hrfs.json')); print('OK', len(m.channels))"
    ```

7. **Git tag and GitHub release:**
    ```bash
    git tag -a v1.2.0 -m "HRFunc 1.2.0 — Correctness Release"
    git push origin v1.2.0
    gh release create v1.2.0 --notes-file CHANGELOG_1_2_0.md
    ```

8. **PyPI upload:**
    ```bash
    twine upload dist/*
    ```

**Exit criteria:**
- All 1.2.0 branches merged to main
- `pip install hrfunc==1.2.0` works from a clean environment
- Library is usable end-to-end from the installed wheel (not from a dev checkout)
- Release notes published on GitHub

---

## Summary

**13 branches total for 1.2.0**, 4 already complete, 9 still to do plus the release branch. Each remaining branch is small enough to review and test in one sitting. The full chain ships Denny a library that won't crash or silently corrupt results on any documented user path.

After 1.2.0 ships and Denny validates it against his own pipeline, work moves to [plan_v2_0_0.md](plan_v2_0_0.md).