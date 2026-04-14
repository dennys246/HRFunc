# HRFunc v2.0.0 — Architectural Refactor Plan

**Release type:** Major version bump (1.2.0 → 2.0.0)
**Prerequisite:** 1.2.0 shipped and validated in Denny's pipeline
**Goal:** Code cleanliness, performance, type safety, and breaking API improvements. The composition refactor is the major-version trigger.

**High-level roadmap:** [phase_breakdown.md](phase_breakdown.md)
**Correctness release (prerequisite):** [plan_v1_2_0.md](plan_v1_2_0.md)

---

## Scope Philosophy

**IN scope for 2.0.0:**
- Architectural refactors (composition, shared helpers)
- Performance optimizations (batch solves, parallel channels)
- Type hints and static analysis
- Structured logging replacing `print()`
- Magic-number extraction to named constants
- Test suite restructuring (fixing KI-033)
- Breaking API changes where they materially improve ergonomics
- Nice-to-have robustness (better error messages, edge-case warnings)

**OUT of scope for 2.0.0:**
- Scientific model changes (HRF model selection, deconvolution math)
- Major new features that expand the library's charter
- Anything already shipped in 1.2.0

**Rule of thumb:** 2.0.0 should make the library nicer to *maintain* and *extend* without changing what it computes.

---

## Branch Decomposition

Fewer hard dependencies than 1.2.0 — most of these can be sequenced flexibly. The only hard ordering is that `refactor/composition` should land before `perf/estimate-activity-parallel` and before `feat/type-hints` (to avoid retyping delegate methods).

---

### `refactor/composition`

**Priority:** First. This is the major-version trigger.

**Scope — Phase 4 from the original plan + ND-005:**

Replace `class montage(tree)` inheritance with composition:

```python
class montage:
    # Class-level constants (moved from magic numbers)
    HBO_HRF_PATH = "hrfs/hbo_hrfs.json"
    HBR_HRF_PATH = "hrfs/hbr_hrfs.json"
    DEFAULT_DURATION = 30.0
    DEFAULT_LAMBDA = 1e-3
    DEFAULT_EDGE_EXPANSION = 0.15
    DEFAULT_MAX_DISTANCE = 0.01

    def __init__(self, nirx_obj=None, **kwargs):
        self._tree = tree()  # Internal tree, not inherited
        self.hbo_tree = tree(f"{_LIB_DIR}/{self.HBO_HRF_PATH}", **kwargs)
        self.hbr_tree = tree(f"{_LIB_DIR}/{self.HBR_HRF_PATH}", **kwargs)
        ...

    # Delegation methods
    @property
    def root(self):
        return self._tree.root

    def insert(self, hrf, depth=0, node=None):
        return self._tree.insert(hrf, depth, node)

    def gather(self, node=None, oxygenation=None):
        return self._tree.gather(node if node is not None else self._tree.root, oxygenation)

    def nearest_neighbor(self, optode, max_distance, **kw):
        return self._tree.nearest_neighbor(optode, max_distance, **kw)

    def delete(self, hrf):
        return self._tree.delete(hrf)

    def filter(self, similarity_threshold=0.95, **kw):
        return self._tree.filter(similarity_threshold, **kw)
```

**Also fixes ND-005** (the hidden third tree): `generate_distribution()` currently inserts global HRFs into `self.insert()`, which goes to `montage.root` (the inherited tree root) — a tree that nothing else searches. After composition, `self._tree` is explicitly the "global HRFs for this montage" tree and is searched by `nearest_neighbor()` fallbacks.

**Breaking changes:**
- `isinstance(m, tree)` will return `False` for montages — document in migration guide
- Any user code that directly accessed tree methods via inheritance still works through delegation but the MRO is different

**Review protocol:**
- Architecture & Execution Flow review — every `self.insert`, `self.root`, `self.gather`, `self.filter`, `self.nearest_neighbor`, `self.delete` call inside `montage` must be traced and verified against the delegation layer
- API surface review — confirm no method signatures change observably for existing users

**Tests:** `tests/test_composition.py` — verify every delegated method works, verify `isinstance(m, tree) == False`, verify `generate_distribution` inserts globals into `self._tree` and `nearest_neighbor` finds them.

**Estimated complexity:** Medium — well-understood refactor, but lots of call sites to audit. Worth splitting into multiple commits within the branch: (a) add `_tree` alongside inheritance, (b) add delegation methods, (c) switch `class montage(tree)` → `class montage`, (d) remove inheritance-based fallbacks.

---

### `perf/estimate-hrf-batch`

**Priority:** Early — biggest performance win, low risk, standalone.

**Scope:**

In `estimate_hrf`, the Toeplitz matrix `X` built from events is **the same for every channel**. Only the signal `Y` differs. Currently the code recomputes `lhs = X.T @ X + lmbda*I` and calls `lstsq` per channel — O(N) solves. Replace with one solve over all channels stacked:

```python
X = scipy.linalg.toeplitz(events, np.zeros(hrf_len))
lhs = X.T @ X + lmbda * np.eye(X.shape[1])

# Z-score every channel along the time axis
Y_matrix = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
Y_matrix = Y_matrix.T  # shape: (n_time, n_channels)

# One solve, all channels at once
rhs = X.T @ Y_matrix
HRF_matrix = np.linalg.solve(lhs, rhs)  # shape: (hrf_len, n_channels)
```

Then iterate the resulting columns to assign per-optode estimates. Same math, same numerical result (up to floating-point), but N solves → 1 solve.

**Expected speedup:** 20–40× for typical montages (20–40 channels). Tested empirically in the review branch.

**Risks:**
- Numerical stability should be identical since `np.linalg.solve` is just as well-conditioned as `lstsq` on a square positive-definite LHS
- Memory: `Y_matrix` is `(n_time × n_channels)` — for a 10-min scan at 10 Hz with 40 channels that's 240k floats, negligible
- Any per-channel processing that can't be batched (e.g., per-channel `cond_thresh` pinv fallback) needs a fallback path

**Review protocol:**
- Data Integrity review — verify batch output is numerically equivalent to per-channel output within machine epsilon on a synthetic test case
- Performance benchmark — measure baseline vs batched on a realistic dataset

**Tests:** `tests/test_estimate_hrf_batch.py` — synthetic dataset where the answer is known analytically; assert batched and per-channel outputs agree.

---

### `refactor/shared-helpers`

**Priority:** Before `perf/estimate-activity-parallel`.

**Scope — Phase 8 + ND-006:**

Extract shared math into private module-level helpers:

```python
def _build_toeplitz_matrix(kernel_or_events, n_time):
    """Build a Toeplitz convolution/deconvolution design matrix."""
    n_k = len(kernel_or_events)
    first_col = np.r_[kernel_or_events, np.zeros(n_time - n_k)]
    first_row = np.r_[kernel_or_events[0], np.zeros(n_time - 1)]
    return scipy.linalg.toeplitz(first_col, first_row)


def _solve_regularized_lstsq(lhs, rhs, cond_thresh=None):
    """Solve regularized least squares with optional pinv fallback."""
    if cond_thresh is not None and np.linalg.cond(lhs) >= cond_thresh:
        return scipy.linalg.pinv(lhs) @ rhs
    result, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return result
```

`estimate_hrf` and the `estimate_activity` deconvolution closure both delegate to these. The MNE `apply_function` single-argument constraint (ND-006) still forces the closure pattern in `estimate_activity`, but the closure body becomes thin.

**Also extract:** `montage._plot_matrix` (Phase 8.1/8.2) — the plotting code duplicated between `correlate_hrf` and `correlate_canonical` gets factored out.

**Review protocol:**
- Architecture review — confirm both callers reach identical numerical output through the shared helpers

**Tests:** `tests/test_shared_helpers.py` — direct unit tests for `_build_toeplitz_matrix` and `_solve_regularized_lstsq` with known inputs.

---

### `perf/estimate-activity-parallel`

**Priority:** After `refactor/shared-helpers`.

**Scope:**

Currently `estimate_activity` creates a `ThreadPoolExecutor(max_workers=1)` per channel inside the deconvolution closure and calls `apply_function` sequentially per channel. Two problems:
1. `max_workers=1` is an executor with no parallelism — pure overhead
2. Channels are processed sequentially — a 40-channel montage runs 40 serial solves

Proposal: break out of the `apply_function` per-channel pattern. Pre-compute all deconvolved channel signals using a shared `ThreadPoolExecutor(max_workers=os.cpu_count())`, then assign them back to the Raw object's `_data` array in one pass.

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    futures = {}
    for ch_idx, (ch_name, hrf) in enumerate(self.channels.items()):
        if 'global' in ch_name:
            continue
        Y = nirx_obj._data[ch_idx]
        futures[executor.submit(_deconvolve_channel, Y, hrf, lmbda, cond_thresh)] = ch_idx

    for future in as_completed(futures):
        ch_idx = futures[future]
        try:
            nirx_obj._data[ch_idx] = future.result(timeout=timeout)
        except TimeoutError:
            logger.warning(f"Channel {ch_idx} timed out after {timeout}s, dropping")
            dropped_channels.append(ch_idx)

nirx_obj.drop_channels([nirx_obj.ch_names[i] for i in dropped_channels])
```

NumPy's `lstsq` releases the GIL, so threading is real parallelism.

**Expected speedup:** 4–8× on a typical multi-core laptop.

**Risks:**
- Writing directly to `nirx_obj._data` bypasses `apply_function`'s safety checks. Verify that MNE handles this cleanly or use a thin wrapper.
- Ordering: `drop_channels` must come AFTER all futures resolve to avoid index invalidation mid-loop.
- Edge case: if some channels drop, the remaining channel indices shift — test carefully.

**Review protocol:**
- Threading & concurrency review (primary)
- Data integrity review — confirm per-channel output matches sequential output

**Tests:** `tests/test_estimate_activity_parallel.py` — construct a montage with a mix of fast-converging and slow-converging HRF kernels (mocked), verify parallel version produces same result as sequential and drops the right channels.

---

### `feat/logging`

**Priority:** Anytime — standalone cleanup.

**Scope — Phase 5:**

Replace every `print(...)` call in `src/hrfunc/` with `logger.info/debug/warning/error` as appropriate:

| Current | Log level |
|---|---|
| `print(f"Deconvolving channel {ch_name}...")` | `logger.debug` |
| `print(f"Tree initialized...")` | `logger.info` |
| `print(f"WARNING: Using canonical HRF...")` | `logger.warning` |
| `print(f"lstsq exceeded {timeout}s...")` | `logger.warning` |
| Error paths | `logger.error` |

Add to each module:
```python
import logging
logger = logging.getLogger(__name__)
```

Scientists can then:
```python
import logging
logging.basicConfig(level=logging.WARNING)  # silence routine output
```

**Risks:** None — pure substitution.

**Review protocol:**
- Quick grep review to confirm every `print` has been replaced with the right log level

**Tests:** `tests/test_logging.py` — use `caplog` fixture to verify key messages are emitted at the right level.

---

### `feat/type-hints`

**Priority:** After `refactor/composition` (to avoid typing methods that get removed) and after `feat/logging` (fewer surface changes).

**Scope — Phase 7:**

Add `from __future__ import annotations` to every module. Add type hints to all public function signatures. Aim for `mypy --strict` compliance on `src/hrfunc/`.

Key signatures:

```python
# hrfunc.py
def localize_hrfs(
    nirx_obj: mne.io.Raw,
    max_distance: float = 0.01,
    verbose: bool = False,
    **kwargs
) -> montage: ...

def load_montage(
    json_filename: str,
    rich: bool = False,
    **kwargs
) -> montage: ...

class montage:
    def __init__(self, nirx_obj: mne.io.Raw | None = None, **kwargs) -> None: ...
    def estimate_hrf(
        self,
        nirx_obj: mne.io.Raw,
        events: list[int] | np.ndarray,
        duration: float = 30.0,
        lmbda: float = 1e-3,
        edge_expansion: float = 0.15,
        preprocess: bool = True,
    ) -> None: ...
    def estimate_activity(
        self,
        nirx_obj: mne.io.Raw,
        lmbda: float = 1e-4,
        hrf_model: str = 'toeplitz',
        preprocess: bool = True,
        cond_thresh: float | None = None,
        timeout: float = 30,
    ) -> mne.io.Raw | None: ...
    def save(self, filename: str = 'montage_hrfs.json') -> None: ...

def preprocess_fnirs(
    scan: mne.io.Raw,
    deconvolution: bool = False,
) -> mne.io.Raw | None: ...
```

**Add a `py.typed` marker file** in `src/hrfunc/` so downstream users get type checking.

**Risks:**
- MNE types may not be fully annotated — may need `# type: ignore` in a few places
- Protocol types for duck-typed arguments (e.g. `events` accepts lists and ndarrays)

**Tests:** Run `mypy --strict src/hrfunc/` as part of the test gate.

---

### `feat/constants`

**Priority:** Anytime after `refactor/composition`.

**Scope — Phase 9:**

Replace magic numbers with module-level named constants:

```python
# hrtree.py
DEFAULT_DURATION: float = 30.0
DEFAULT_SFREQ: float = 7.81
CANONICAL_LOCATION: list[float] = [359.0, 359.0, 359.0]

# hrfunc.py
DEFAULT_LAMBDA_HRF: float = 1e-3
DEFAULT_LAMBDA_ACTIVITY: float = 1e-4
DEFAULT_EDGE_EXPANSION: float = 0.15
DEFAULT_MAX_DISTANCE: float = 0.01
DEFAULT_TIMEOUT_SECONDS: float = 30.0
SCI_THRESHOLD: float = 0.95
CANONICAL_HRF_PEAK_RATIO: float = 1.0
HBR_WAVELENGTH_LOW: int = 760
HBR_WAVELENGTH_HIGH: int = 780
HBO_WAVELENGTH_LOW: int = 830
HBO_WAVELENGTH_HIGH: int = 850
```

**Risks:** Low — mechanical substitution.

**Tests:** None specific; existing tests should pass unchanged.

---

### `feat/test-suite-restructure`

**Priority:** Near the end of 2.0.0 work, after other refactors land.

**Scope — Phase 12 + KI-033:**

The `tests/test_estimation.py` and `tests/test_localization.py` files currently execute code at module-import time, which crashes pytest collection (KI-033). Move the module-level code into proper test functions, add `@pytest.mark.slow` or `@pytest.mark.integration` markers, and wire them into the CI pipeline.

**Work:**

1. **Convert existing scripts to pytest tests:**
    ```python
    # Currently:
    print("TEST: Estimating HRF...")
    for filetype_load, filetype in zip(...):
        ...

    # Should become:
    @pytest.mark.integration
    def test_hrf_estimation_nirx():
        ...
    ```

2. **Add fixtures for common data loading:**
    ```python
    @pytest.fixture
    def events():
        return load_events()

    @pytest.fixture(scope='session')
    def nirx_scans():
        return load_nirx_test_data()
    ```

3. **Mark data-heavy tests so they can be skipped in fast runs:**
    ```python
    @pytest.mark.slow
    @pytest.mark.integration
    def test_full_pipeline_on_real_dataset():
        ...
    ```

4. **Guard optional data files:**
    ```python
    @pytest.fixture
    def test_data_path():
        path = Path('tests/data/sNIRF_formatted')
        if not path.exists():
            pytest.skip("Test data not available")
        return path
    ```

5. **Unify the test gate:**
    - Fast gate (CI, pre-commit): `pytest -m "not slow and not integration"`
    - Full gate (pre-release): `pytest`

6. **Fix test_localization.py RuntimeError-not-raised bug:**
    ```python
    # Currently (line 33, 50):
    RuntimeError(f"ERROR: True channel {ch_name} not found...")

    # Fix:
    raise RuntimeError(f"ERROR: True channel {ch_name} not found...")
    ```

7. **Ensure the targeted test files from 1.2.0 are preserved and still run** as part of the fast gate.

**Review protocol:**
- Verify pytest collection succeeds on the full `tests/` directory
- Verify fast and full gates each produce sensible output

---

### `feat/robustness-io`

**Priority:** Anytime — standalone.

**Scope — Phase 6 spillover:**

Robustness guards that weren't critical enough for 1.2.0 but improve error messages:

1. **JSON load:**
    ```python
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"HRFunc montage file not found: {filename}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {filename}: {e}")
    ```

2. **JSON save:**
    ```python
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w") as file:
            json.dump(data, file, indent=4)
    except (IOError, OSError) as e:
        raise IOError(f"Failed to save montage to {filename}: {e}")
    ```

3. **Plot paths:** `os.makedirs(os.path.dirname(plot_path), exist_ok=True)` before any `plt.savefig`.

**Risks:** None.

**Tests:** `tests/test_io_robustness.py` — trigger each error path with synthetic bad inputs.

---

### `feat/event-edge-warning`

**Priority:** Anytime — standalone.

**Scope:**

In `estimate_hrf`, the edge expansion logic shifts events backward by `timeshift` samples and silently `continue`s any event that falls off the beginning of the scan. This biases HRF estimation toward late-session events if early events are preferentially lost.

**Fix:**
Track the count of dropped events and emit a `logger.warning` (or `UserWarning` via `warnings.warn`) when nonzero, including the count and the suggestion to reduce `edge_expansion` if this happens frequently.

```python
dropped_events = 0
for ind in range(events.shape[0]):
    if events[ind] != 0:
        if (ind - timeshift) < 0:
            dropped_events += 1
            continue
        new_events[ind - timeshift] = 1

if dropped_events > 0:
    logger.warning(
        f"Dropped {dropped_events} event(s) during edge expansion "
        f"(timeshift={timeshift} samples). Consider reducing edge_expansion "
        f"or ensuring events are not at the scan start."
    )
```

**Tests:** `tests/test_event_edge.py` — construct events near the boundary, verify warning is emitted with correct count.

---

### `feat/adaptive-regularization`

**Priority:** Last, or deferred to 2.1.0 — design question, not a pure refactor.

**Scope:**

Currently `lmbda` is a fixed default (`1e-3` for estimate_hrf, `1e-4` for estimate_activity) regardless of the matrix magnitude. Tikhonov regularization works best when `lmbda` is scaled relative to the norm of the LHS (e.g. to a fraction of the largest eigenvalue or the Frobenius norm).

**Investigation (not a fix):**

1. Profile `np.linalg.cond(X.T @ X)` on realistic fNIRS datasets.
2. Compare fixed `lmbda=1e-3` vs `lmbda = 1e-3 * np.linalg.norm(X, 'fro')` on several test datasets.
3. Survey the fMRI/fNIRS literature for standard lmbda-scaling conventions.
4. Report findings to Denny; decide whether to change the default or expose a `lmbda_scale='norm'` option alongside the fixed mode.

**This is a design question, not a bug.** May not ship in 2.0.0 — document findings and defer if unclear.

---

## Sequencing Recommendation

```
1.2.0 ships, validated in real pipeline
│
├── refactor/composition              ← start here, biggest architectural change
│   ├── refactor/shared-helpers
│   │   ├── perf/estimate-hrf-batch
│   │   └── perf/estimate-activity-parallel
│   ├── feat/logging
│   ├── feat/constants
│   ├── feat/robustness-io
│   └── feat/event-edge-warning
│
├── feat/type-hints                   ← after composition, before release
│
├── feat/test-suite-restructure       ← final gate
│
└── release/2.0.0                     ← tag, migration guide, PyPI
```

Branches in the same tier can proceed in parallel if someone wants to split the work across multiple sessions.

---

## Migration Guide for v1.2.0 → v2.0.0 Users

To be written during `release/2.0.0`. Must cover:

- **`isinstance(m, tree)` returns False now**: use duck-typed methods instead of type checks
- **All `print` output is now logging**: users who want the old verbose behavior call `logging.basicConfig(level=logging.DEBUG)`
- **`estimate_hrf` performance improvement** — same results, much faster
- **Type hints available** — downstream projects can now `mypy` HRFunc usages
- Any other observable changes from the refactors

---

## Summary

11 branches across architectural, performance, cleanliness, and tooling work. Estimated work unit: 2–4 sessions for the heavy branches (`composition`, `parallel`, `test-suite-restructure`), 1 session each for the rest. Total scope is larger than 1.2.0 but lower risk per branch.