# HRFunc Internal Architecture

**Version:** 1.2.0 (correctness release in flight)
**Last reviewed:** 2026-04-14
**Audience:** Contributors and maintainers

---

## Overview

HRFunc estimates hemodynamic response functions (HRFs) and neural activity from fNIRS (functional near-infrared spectroscopy) brain imaging data. It wraps MNE-Python's Raw fNIRS objects and applies Toeplitz deconvolution with Tikhonov regularization to recover HRF shapes and neural activity timeseries.

The library is used by neuroscientists to:
1. Estimate channel-wise HRFs from event-related fNIRS scans
2. Store and retrieve those HRFs from a community-sourced spatial library (the HRtree)
3. Use those HRFs to deconvolve neural activity from new fNIRS scans

---

## Module Structure

```
src/hrfunc/
├── __init__.py      — public exports
├── hrfunc.py        — montage (main API), standalone functions
├── hrtree.py        — tree (k-d tree), HRF (node/data container)
├── hrhash.py        — hasher (open-addressed hash table)
├── observer.py      — lens (signal quality metrics)
├── _utils.py        — standardize_name, _is_oxygenated, _LIB_DIR
└── hrfs/
    ├── hbo_hrfs.json — bundled HbO HRF library
    └── hbr_hrfs.json — bundled HbR HRF library
```

---

## Module Dependency Graph

```
__init__.py
├── hrfunc.py  ──→  hrtree.py  ──→  hrhash.py
│                 ─→  _utils.py (standardize_name, _is_oxygenated, _LIB_DIR)
└── observer.py  (standalone, no internal imports)

hrtree.py
├── from . import hrhash
├── from ._utils import standardize_name, _is_oxygenated, _LIB_DIR
└── _flatten_context_value  (module-level helper exported to hrfunc.py)

hrfunc.py
├── from .hrtree import tree, HRF, _flatten_context_value
└── from ._utils import standardize_name, _is_oxygenated, _LIB_DIR
```

The old circular import between `hrfunc.py` and `hrtree.py` was broken during the `refactor/circular-imports-phase2` branch by extracting shared helpers into `_utils.py`. Neither `hrtree.py` nor `_utils.py` imports from `hrfunc.py` anymore.

---

## Class Responsibilities

### `montage` (hrfunc.py)

The primary user-facing object. Currently **inherits from `tree`** while also **owning two separate `tree` instances** (`self.hbo_tree`, `self.hbr_tree`). This inheritance/composition clash is the central architectural problem addressed by `refactor/composition` in v2.0.0.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `channels` | `dict[str, HRF]` | ch_name → HRF node pointer |
| `hbo_tree` | `tree` | Spatial library of HbO HRFs (from hbo_hrfs.json) |
| `hbr_tree` | `tree` | Spatial library of HbR HRFs (from hbr_hrfs.json) |
| `hbo_channels` | `list[str]` | HbO channel names from nirx_obj |
| `hbr_channels` | `list[str]` | HbR channel names from nirx_obj |
| `sfreq` | `float` | Sampling frequency extracted from nirx_obj |
| `context` | `dict` | Metadata applied during HRF tree filtering |
| `context_weights` | `dict` | Per-key weights for context similarity scoring |
| `configured` | `bool` | Whether configure() has been called with a nirx_obj |
| `lib_dir` | `str` | Path to hrfunc package directory (for bundled JSON) |

**Key methods:**
- `configure(nirx_obj)` — transactional: commits sfreq/channel lists only on successful `_merge_montages()`; rolls back scalar/list state AND undoes tree inserts on failure (via `tree.delete`)
- `_merge_montages(nirx_obj)` — create empty HRF nodes for each channel, link to channels dict
- `estimate_hrf(nirx_obj, events, duration, lmbda, edge_expansion, preprocess)` — Toeplitz HRF estimation with M2a/M2b input validation at the top (duration > 0, events non-empty, lmbda > 0)
- `estimate_activity(nirx_obj, lmbda, hrf_model, preprocess, cond_thresh, timeout)` — neural activity deconvolution. Snapshot-iterates self.channels so orphaned entries can be popped on drop. Catches generic `Exception` from the solve (not just `TimeoutError`) so every failure routes through the drop-and-cleanup path
- `localize_hrfs(max_distance, verbose)` — k-d tree nearest-neighbor search for each channel; falls back to `tree.get_canonical_hrf` at the scan's own sfreq on miss
- `generate_distribution()` — compute mean/std across subject estimates per channel
- `save(filename)` — serialize montage HRFs to JSON
- `correlate_hrf(plot_filename)` — Spearman correlation matrix across channels
- `correlate_canonical(plot_filename, duration)` — correlation vs. double-gamma canonical HRF (raises `ValueError` on unconfigured montage)
- `branch(**kwargs)` — reimplements sub-tree construction by iterating `self.channels` and deep-copying matching nodes. The user-facing branch API.

---

### `tree` (hrtree.py)

A 3D k-d tree that indexes HRF estimates by optode spatial position (x, y, z in MNE's coordinate space). The splitting axis cycles through x → y → z by depth.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `root` | `HRF\|None` | Root node of the tree |
| `hasher` | `hasher` | Context-based fast lookup table, keyed by context VALUES |
| `context` | `dict` | Default context for filtering/branching |
| `context_weights` | `dict` | Per-key similarity weights |
| `branched` | `bool` | Whether branch() has been called |
| `hrf_filename` | `str\|None` | Source JSON path if loaded from file |
| `_canonical_cache` | `dict` | Lazy Glover HRF cache keyed on `(oxygenation, sfreq, duration)` |

**Canonical HRFs are lazy (S4):** `tree.insert` does NOT create a sentinel node. The tree is a pure kd-tree of user HRFs. When a canonical is needed, callers invoke `tree.get_canonical_hrf(oxygenation, sfreq, duration)` which generates a Glover HRF at the requested sample rate, caches it, and returns an HRF node at sentinel location `[359, 359, 359]`. `nearest_neighbor` returns `(None, float("inf"))` on miss; callers handle `None` explicitly and fetch the canonical via the helper.

**Key methods:**
- `insert(hrf, depth, node)` — recursive k-d insertion; jitter branch directly mutates `hrf.x/y/z` (3.5 fix) and refreshes `h_val` for the axis routing
- `get_canonical_hrf(oxygenation, sfreq, duration)` — lazy canonical generator with cache
- `nearest_neighbor(optode, max_distance, ...)` — recursive k-d search; explicit early return on empty tree; returns `(HRF, distance)` or `(None, inf)` on miss
- `radius_search(optode, radius, ...)` — collect all nodes within Euclidean radius
- `filter(similarity_threshold, node, **kwargs)` — delete nodes below context similarity threshold. Iterates post-order so child mutations don't affect parent traversal
- `branch(**kwargs)` — AND-on-kwargs / OR-within-kwarg semantics. Empty kwargs returns an empty sub-tree. Populates the sub-tree's hasher keyed by copied node context VALUES
- `compare_context(first_context, second_context, context_weights)` — returns 0.0–1.0 similarity. Auto-wraps scalar values, handles missing keys in second_context, uses `weights.get(key, 1.0)` for partial weights dicts
- `gather(node, oxygenation)` — collect all nodes into JSON-serializable dict. Returns `{}` on `node=None`
- `delete(hrf)` / `_delete_recursive(node, hrf, depth)` — standard kd-tree delete-by-copy using `_copy_payload` helper
- `_copy_payload(src, dst)` — copies all HRF payload fields except `left`/`right`. Uses `np.copy` for trace/trace_std and dict/list copies for context/estimates/locations to avoid aliasing. Copies the `built` flag but NOT `hrf_processes`/`process_names`/`process_options` (bound methods point to `src`)
- `merge(tree, node)` — walks the SOURCE tree but inserts fresh `node.copy()` so merged tree is independent

### Module-level helper

- `_flatten_context_value(value)` — generator that yields hashable keys from a context dict value. Flattens one-level lists/tuples, skips None. Used by `load_hrfs`, `load_montage`, and `tree.branch` to bridge HRF context values (scalars or lists like `age_range=[20, 30]`) to the hasher's single-key requirement.

---

### `HRF` (hrtree.py)

Tree node AND data container. Serves dual roles: spatial tree node (has `left`, `right` pointers and `x`, `y`, `z` coordinates) and HRF data record.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `doi` | `str` | Publication DOI for provenance tracking |
| `ch_name` | `str` | Standardized channel name (e.g., `s1_d1_hbo`) |
| `sfreq` | `float` | Sampling frequency of source scan |
| `length` | `int` | Expected trace length (sfreq × duration) |
| `oxygenation` | `bool` | True=HbO, False=HbR |
| `trace` | `np.ndarray` | Mean HRF timeseries |
| `trace_std` | `np.ndarray\|None` | Standard deviation over time |
| `x, y, z` | `float` | 3D optode coordinates |
| `estimates` | `list` | Per-subject HRF estimates (list of lists) — per-instance, not shared |
| `locations` | `list` | Per-subject optode locations — per-instance, not shared |
| `context` | `dict` | Experimental metadata — per-instance, not shared |
| `left`, `right` | `HRF\|None` | k-d tree child pointers |
| `hrf_processes` | `list` | Pipeline step bound methods (currently `[self.spline_interp]`) |
| `process_names` | `list` | Human-readable names for each step |
| `process_options` | `list` | Per-step options (one entry per process; `[None]` default triggers the zero-arg path) |
| `built` | `bool` | Whether `build()` has already run |

Mutable default args (`estimates=[]`, `locations=[]`, `context=[]`) were replaced with None sentinels + per-instance materialization in `fix/tree-hrf-correctness` (3.7).

**Key methods:**
- `build(new_sfreq, plot, show)` — apply the pipeline in `hrf_processes` to resample `trace` to `new_sfreq × duration` samples. Derives `hrf_type` from `self.oxygenation` locally instead of reading the never-set `self.type` (3.10).
- `spline_interp(trace=None)` — resample `trace` (defaults to `self.trace`) via cubic spline to `self.target_length`. Accepts a trace argument so `build()` can call it through the pipeline pattern (NE-003 fix)
- `smooth(a)` — Gaussian smoothing via the module-level `scipy.ndimage.gaussian_filter1d` (NE-004 fix — was previously calling a non-existent `self.gaussian_filter1d`)
- `update_centroid()` — set x,y,z to mean of all locations[]
- `copy()` — deep copy of HRF. `context`/`estimates`/`locations` are fresh, `trace`/`trace_std` are `np.copy`'d. Left/right are None.
- `plot(plot_path, show_legend, show)` — save HRF plot PNG

---

### `hasher` (hrhash.py)

Open-addressed hash table mapping **context values** (not keys) to lists of HRF node pointers. Used by `tree.branch()` to avoid full tree traversal when filtering by context.

- Hash function: SHA3-512 (via `hashlib`)
- Collision resolution: linear probe `(5*hashkey + 1) % capacity`
- Resize triggers: shrink at <1/3 fill, grow at >2/3 fill
- Tombstone deletion: removed slots marked `'!tombstone!'`
- Storage: each slot holds a **list** of pointers (multiple HRFs can share a context value)

**Contract (post-fix/hasher-branch-correctness):**
- `add(key, pointer)` — appends pointer to the slot's list, deduplicated by identity
- `search(key)` — returns `list[pointer]`. Empty list on miss (not `False`). Returns a shallow copy so callers can iterate safely.
- `remove(key)` — removes the key's slot and replaces with an empty list

`fill` and `double_check` methods were dead code with broken signatures; deleted in `fix/hasher-branch-correctness`.

---

### `lens` (observer.py)

Signal quality observer for comparing preprocessed vs. deconvolved data. Standalone from the core pipeline — used for QA/QC.

- Tracks: kurtosis, skewness, SNR (Welch PSD) per channel per subject
- Outputs: CSV files (SNR, kurtosis, skewness), PNG plots
- `__init__` initializes `self.sfreq` (default 7.81) and `self.channels = []` so direct `calc_snr` calls don't AttributeError (3.9 fix)
- `calc_snr` takes `noise_bands=None` sentinel and materializes the default list inside the function (3.7-class fix for observer). The pre-fix variable swap between `psd_noise_slow` / `preproc_noise_fast` is fixed.

---

## Execution Paths

### Path 1: Localize pre-trained HRFs to optodes
```
localize_hrfs(nirx_obj, max_distance=0.01, **kwargs)
  → montage.__init__(nirx_obj, **kwargs)
      → self.hbo_tree = tree("hbo_hrfs.json")   # loads pre-trained library
      → self.hbr_tree = tree("hbr_hrfs.json")
      → self.configure(nirx_obj)
          → extract sfreq, hbo_channels, hbr_channels
          → _merge_montages(nirx_obj): for each channel, nearest_neighbor(empty_hrf, 1e-9)
          → commit on success; roll back on failure (including tree.delete of newly-inserted nodes)
  → montage.localize_hrfs(max_distance)
      → for ch_name in self.channels:
          hrf, dist = hbo_tree.nearest_neighbor(optode, max_distance)  # or hbr_tree
          if hrf is not None: attach hrf.trace, hrf.trace_std to optode
          else: attach canonical trace from hbo_tree.get_canonical_hrf(True, self.sfreq, duration)
→ return montage
```

### Path 2: Estimate HRFs from events
```
montage.estimate_hrf(nirx_obj, events, duration=30.0, lmbda=1e-3, edge_expansion=0.15)
  → M2a/M2b validation: duration > 0, events non-empty list, lmbda > 0
  → configure(nirx_obj) if not configured
  → if preprocess: nirx_obj = preprocess_fnirs(nirx_obj, deconvolution=True)
      (if preprocess_fnirs returns None → early return)
  → nirx_obj.load_data()  (AFTER preprocessing — 1.8 pipeline ordering fix)
  → data = nirx_obj.get_data()
  → Expand events by edge_expansion to account for Toeplitz edge artifacts
  → X = scipy.linalg.toeplitz(events, zeros(hrf_len))
  → for each channel:
      Y = (signal - mean) / std                           # z-score normalize
      lhs = X.T @ X + lmbda * I
      rhs = X.T @ Y
      hrf_estimate = solve_lstsq(lhs, rhs)
      hrf_estimate = hrf_estimate[timeshift:-timeshift]   # remove edge artifacts
      optode.estimates.append(hrf_estimate)
      optode.locations.append(channel_loc)
```

### Path 3: Deconvolve neural activity
```
montage.estimate_activity(nirx_obj, lmbda=1e-4, hrf_model='toeplitz', timeout=30)
  → M2b validation: lmbda > 0
  → configure(nirx_obj) if not configured
  → nirx_obj = preprocess_fnirs(nirx_obj, deconvolution=True)   # captures return
      (return None early if all channels bad)
  → success = None  (outer-scope declaration for nonlocal closure write)
  → dropped_channels = []
  → for ch_name in list(self.channels.keys()):  (snapshot iteration)
      hrf = self.channels[ch_name]
      success = None
      if 'global' in ch_name: continue

      # H1: zero-trace guard
      trace_invalid = (hrf.trace is None or len(hrf.trace) == 0
                       or np.max(np.abs(hrf.trace)) == 0)
      if trace_invalid and hrf_model != 'canonical':
          print warning; fall through to canonical branch

      if hrf_model == 'canonical' or trace_invalid:
          # S4: canonical generated at the scan's own sfreq
          hrf = self.hbo_tree.get_canonical_hrf(True, self.sfreq, duration)  # or hbr

      Closure deconvolution(nirx):         # captures hrf, lmbda, cond_thresh, timeout
        Y = (nirx - mean) / std
        hrf_kernel = hrf.trace / np.max(np.abs(hrf.trace))
        A = scipy.linalg.toeplitz(hrf_kernel, ...)
        lhs = A.T @ A + lmbda * I
        rhs = A.T @ Y
        with ThreadPoolExecutor(max_workers=1):
          try: deconvolved = solve_lstsq(lhs, rhs, cond_thresh); success = True
          except TimeoutError: deconvolved = nirx; success = False
          except Exception: deconvolved = nirx; success = False   # M4: catch-all
        return deconvolved
      nirx_obj.apply_function(deconvolution, picks=[ch_name])

      if success is False:
          nirx_obj.drop_channels([ch_name])
          dropped_channels.append(ch_name)

  # M4 post-loop cleanup: pop dropped channels from self.channels and lists
  for ch_name in dropped_channels:
      self.channels.pop(ch_name, None)
      if ch_name in self.hbo_channels: self.hbo_channels.remove(ch_name)
      if ch_name in self.hbr_channels: self.hbr_channels.remove(ch_name)
  return nirx_obj
```

**Why the closure pattern**: `mne.Raw.apply_function()` requires a single-argument callable `f(channel_data) → channel_data`. Extra parameters (hrf kernel, lambda, etc.) must be captured via closure. This constraint prevents clean abstraction of deconvolution logic between `estimate_hrf` and `estimate_activity`. (See `project_deconvolution_quirk.md` memory for the full explanation.)

### Path 4: Load montage from JSON
```
load_montage(json_filename, rich=False, **kwargs)
  → parse JSON
  → M3 validation: json_contents must be non-empty dict; first entry must contain sfreq
  → montage(**kwargs)   # fresh montage with library hbo_tree/hbr_tree
  → for each JSON entry (M5 per-entry try/except):
      → M3 per-entry schema validation (hrf_mean, hrf_std, sfreq, location, context, context.duration)
      → create HRF with channel['context']  (cross-branch fix: pre-fix omitted this argument)
      → insert into hbo_tree or hbr_tree
      → populate target tree's hasher keyed by channel context VALUES (NE-002)
  → on any exception: raise ValueError naming the entry key; local _montage dropped
  → return _montage
```

---

## Data Preprocessing Pipeline (preprocess_fnirs)

```
mne.Raw (optical density)
  1. mne.preprocessing.nirs.optical_density()
  2. mne.preprocessing.nirs.scalp_coupling_index() → mark bad channels (threshold < 0.5)
  3. mne.interpolate_bads()
  4. mne.preprocessing.nirs.tddr()                  # motion correction
  5. [deconvolution=True] polynomial detrend (order 3)
  6. mne.preprocessing.nirs.beer_lambert_law(ppf=0.1)  # experimental ppf value — see plan
  7. raw.apply_baseline((None, 0))
  8. [deconvolution=False] raw.filter(0.01, 0.2)     # hemodynamic bandpass
→ mne.Raw (hemoglobin concentrations) — NEW OBJECT, caller must capture
```

Note: the `ppf=0.1` value is under experimental validation in `experiment/ppf-validation`. MNE's default is `6.0`. The resolution of this question is pending Denny's experimental test on real data.

---

## File I/O Schema

### HRF JSON format
```json
{
  "{ch_name}-{doi}": {
    "hrf_mean": [float, ...],
    "hrf_std": [float, ...],
    "location": [x, y, z],
    "oxygenation": bool,
    "sfreq": float,
    "context": {
      "method": "toeplitz",
      "doi": "...",
      "study": "...",
      "task": "...",
      "conditions": [...],
      "stimulus": "...",
      "intensity": 1.0,
      "duration": 30.0,
      "protocol": "...",
      "age_range": [...],
      "demographics": "..."
    },
    "estimates": [[float, ...], ...],
    "locations": [[x, y, z], ...]
  }
}
```

### Channel name standardization
Channel names are standardized via `_utils.standardize_name()` before any dict lookup:
1. Short-name and type guards (raises `TypeError`/`ValueError` on bad input)
2. Replace separators `[-_\s]+` → `_`
3. Lowercase
4. Normalize suffix: `hbo` or `hbr`

`_utils._is_oxygenated` has its own matching guards so direct callers (in configure, load_montage, etc.) also get clean errors on degenerate input.

---

## Hasher Data Model

Each slot in the `hasher.contexts` array holds a `list` of HRF node pointers. `add` appends, `search` returns the full list (shallow copy). `load_hrfs` and `load_montage` populate the hasher with each channel's context VALUES (e.g., `'flanker'`, `'checkerboard'`, a DOI string, or each element of `age_range`). `tree.branch(task='flanker')` then searches for `'flanker'` as a hasher key and gets back every node whose context dict contained it anywhere.

The `_flatten_context_value` helper bridges the gap between HRF context values (scalars, lists, or None) and the hasher's single-hashable-key requirement.

---

## Units Convention (intentional)

HRFunc intentionally outputs HRF traces and deconvolved neural activity in **arbitrary units (a.u.)**, matching the fMRI BOLD analysis convention:

- `estimate_hrf` z-scores the input signal before the least-squares solve and does not denormalize back to absolute hemoglobin concentrations. Returned HRF traces are in z-score space.
- `estimate_activity` peak-normalizes the HRF kernel (`hrf.trace / np.max(np.abs(hrf.trace))`) before using it as a deconvolution template. Returned activity is relative deviation from baseline.

This is NOT a bug — past code reviews have flagged it as one. Do not "fix" it. See the `project_unit_conventions.md` memory for the full rationale.

---

## Known Architectural Problems (scope: v2.0.0)

The v1.2.0 correctness release fixed all the crash-class, silent-wrong-result, and input-validation issues. Remaining architectural problems are all scoped to v2.0.0:

1. **`class montage(tree)` inheritance clash** — montage both inherits from tree (has its own root, insert, etc.) AND owns hbo_tree/hbr_tree. Fixed by `refactor/composition`.
2. **No structured logging** — `print()` everywhere. Fixed by `feat/logging`.
3. **No type hints** — mypy can't give us much structure. Fixed by `feat/type-hints`.
4. **Magic numbers scattered through the code** — Fixed by `feat/constants`.
5. **Integration tests crash pytest on collection (KI-033)** — module-level code in `test_estimation.py` and `test_localization.py`. Fixed by `feat/test-suite-restructure`.
6. **Sequential channel solves in `estimate_activity`** — one ThreadPoolExecutor per channel. Fixed by `perf/estimate-activity-parallel` after `refactor/shared-helpers`.
7. **Sequential batch solves in `estimate_hrf`** — one lstsq per channel. Fixed by `perf/estimate-hrf-batch`.
8. **Absolute 1e-10 jitter in `insert`** — effective at MNE NIRX meter scale but loses relative precision at much larger coordinate scales. Flagged for v2.0.0 magnitude-relative jitter.

---

## References

- Plan roadmap: [docs/plans/phase_breakdown.md](../plans/phase_breakdown.md)
- v1.2.0 detailed plan: [docs/plans/plan_v1_2_0.md](../plans/plan_v1_2_0.md)
- v2.0.0 detailed plan: [docs/plans/plan_v2_0_0.md](../plans/plan_v2_0_0.md)
- Known issues (open + resolved catalog): [known_issues.md](known_issues.md)
- v1.2.0 change summary: [v1_2_0_changelog_summary.md](v1_2_0_changelog_summary.md)
