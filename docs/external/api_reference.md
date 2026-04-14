# HRFunc API Reference

**Version:** 1.2.0 (correctness release, in flight)
**Python:** ≥ 3.8
**Depends on:** MNE-Python, NumPy, SciPy, nilearn, matplotlib

For installation and quickstart, see the [README](../../README.md).
For in-depth guides, visit [hrfunc.org](https://www.hrfunc.org).

---

## Quick Import

```python
import hrfunc as hrf
import mne

# Load your scan via MNE
scan = mne.io.read_raw_snirf("subject_1.snirf")

# Core objects
montage = hrf.montage(scan)                 # Main estimation object
montage = hrf.load_montage("hrfs.json")     # Load saved HRFs

# Standalone functions
montage = hrf.localize_hrfs(scan)           # Attach pre-trained HRFs
scan = hrf.preprocess_fnirs(scan)           # Preprocess fNIRS data
```

---

## Return-value capture pattern

Several MNE NIRS preprocessing steps internally call `.copy().load_data()` and return a new object rather than mutating in place. HRFunc follows the same convention:

```python
# WRONG — scan is still unpreprocessed
hrf.preprocess_fnirs(scan)

# RIGHT — capture the return value
scan = hrf.preprocess_fnirs(scan)
if scan is None:
    # preprocess_fnirs returns None when every channel was flagged bad
    return
```

`montage.estimate_activity()` also returns an MNE Raw object (or `None` if all channels were bad), and must be captured the same way when you need the deconvolved data.

---

## `montage`

The central object for HRF estimation and neural activity deconvolution.

### `montage(nirx_obj=None, **kwargs)`

Create a new montage, optionally configured to a specific fNIRS scan.

**Parameters:**
- `nirx_obj` (`mne.io.Raw`, optional) — fNIRS scan loaded through MNE. Accepted formats: NIRX, sNIRF, FIF. If provided, the montage is immediately configured to the scan's channels and sampling frequency.
- `**kwargs` — Context metadata as keyword arguments. See [Context System](#context-system) below for the full list of supported keys.

**Attributes after construction:**
- `channels` (`dict`) — Maps standardized channel name → HRF node
- `hbo_channels` / `hbr_channels` (`list`) — Channel names by oxygenation
- `sfreq` (`float`) — Sampling frequency in Hz (only set after `configure()`)
- `context` (`dict`) — Active context metadata
- `configured` (`bool`) — Whether the montage has been configured to a scan
- `hbo_tree` / `hbr_tree` (`tree`) — Bundled HRF libraries

**Unconfigured instances are safe to print and introspect** — `__repr__` handles missing `sfreq` / channel lists gracefully.

**Example:**
```python
import mne
import hrfunc as hrf

scan = mne.io.read_raw_snirf("subject_1.snirf")
montage = hrf.montage(scan, doi="10.1000/xyz", task="flanker")
```

---

### `montage.configure(nirx_obj, **kwargs)`

Configure the montage against an MNE Raw object. Called automatically by `__init__` if `nirx_obj` is provided.

**Failure behavior:** configure is transactional. If anything fails partway through (channel-name error, `_merge_montages` exception), the montage is rolled back to its pre-call state via the spatial tree. A re-configure attempt that fails leaves the previous configuration intact.

**Parameters:**
- `nirx_obj` (`mne.io.Raw`) — fNIRS scan
- `**kwargs` — Additional context overrides

---

### `montage.estimate_hrf(nirx_obj, events, duration=30.0, lmbda=1e-3, edge_expansion=0.15, preprocess=True)`

Estimate channel-wise HRFs from a single subject's fNIRS scan and event series using Toeplitz deconvolution with Tikhonov regularization.

Call this once per subject. Multiple calls accumulate estimates in `montage.channels[ch_name].estimates`; call `generate_distribution()` after all subjects have been processed to compute means and standard deviations.

**Parameters:**
- `nirx_obj` (`mne.io.Raw`) — fNIRS scan
- `events` (`list[int]`) — Binary event impulse series (0 or 1 per sample), same length as scan
- `duration` (`float`) — HRF duration to estimate in seconds, default `30.0`. Must be `> 0`.
- `lmbda` (`float`) — Tikhonov regularization strength, default `1e-3`. Must be `> 0`. Increase to smooth the HRF estimate; decrease for sharper (potentially noisier) estimates.
- `edge_expansion` (`float`) — Fraction of `duration` used to pad edges to remove Toeplitz artifacts, default `0.15` (15%)
- `preprocess` (`bool`) — If `True`, run `preprocess_fnirs()` before estimation

**Raises:**
- `ValueError` — `duration` not numeric or `<= 0`
- `ValueError` — `events` not a list, empty, or length mismatch with scan
- `ValueError` — `lmbda <= 0`

**Returns:** `None` (HRF estimates accumulated in `montage.channels[ch_name].estimates`)

**Example:**
```python
with open("events.txt") as f:
    events = [int(line.strip()) for line in f]

for scan in scans:
    montage.estimate_hrf(scan, events, duration=30.0)

montage.generate_distribution()
```

---

### `montage.generate_distribution(plot_dir=None)`

Compute the channel-wise mean HRF and standard deviation across all accumulated subject estimates. Call this after `estimate_hrf()` has been run for every subject.

**Parameters:**
- `plot_dir` (`str`, optional) — Directory path to save per-channel HRF plots. Directory must exist.

**Returns:** `None` (updates `montage.channels[ch_name].trace` and `.trace_std`)

---

### `montage.estimate_activity(nirx_obj, lmbda=1e-4, hrf_model='toeplitz', preprocess=True, cond_thresh=None, timeout=30)`

Deconvolve neural activity from a fNIRS scan using the estimated HRFs stored in the montage. Returns a new MNE Raw object with deconvolved channel data — **capture the return value**.

**Parameters:**
- `nirx_obj` (`mne.io.Raw`) — fNIRS scan to deconvolve
- `lmbda` (`float`) — Tikhonov regularization strength, default `1e-4`. Must be `> 0`.
- `hrf_model` (`str`) — `'toeplitz'` to use localized HRFs, or `'canonical'` to use the Glover canonical HRF for all channels (generated at the scan's actual sample rate).
- `preprocess` (`bool`) — If `True`, run `preprocess_fnirs()` before deconvolution
- `cond_thresh` (`float`, optional) — Condition number threshold; if exceeded, uses pseudoinverse instead of lstsq
- `timeout` (`int`) — Maximum seconds to wait for lstsq convergence per channel, default `30`. Realistic fNIRS inputs solve in tens of milliseconds; this fires only on pathological matrices.

**Behavior on failure:** If a channel's solve times out or raises any exception, the channel is dropped from both the output `nirx_obj` and the montage's `channels` / `hbo_channels` / `hbr_channels` lists, so downstream calls like `correlate_hrf()` and `generate_distribution()` don't iterate orphan entries.

**Zero-trace fallback:** If a channel's HRF has an empty, `None`, or all-zero trace, `estimate_activity` transparently falls back to the canonical HRF and emits a warning. This avoids silent `NaN` propagation from dividing by `max(abs(zeros))`.

**Canonical HRF matches scan sample rate:** When `hrf_model='canonical'` (or zero-trace fallback is triggered), the canonical is generated at the scan's actual `sfreq`, not a hardcoded 7.81 Hz. Each `(sfreq, duration)` pair is generated once and cached.

**Raises:**
- `ValueError` — `lmbda <= 0`

**Returns:** `mne.io.Raw` — the deconvolved scan, or `None` if all channels were bad.

**Example:**
```python
for scan in scans:
    deconvolved = montage.estimate_activity(scan)
    if deconvolved is not None:
        deconvolved.save(f"neural_{scan.filenames[0]}")
```

---

### `montage.localize_hrfs(max_distance=0.01, verbose=False)`

Attach pre-trained HRFs from the bundled library to each channel using spatial nearest-neighbor search. Channels without a match within `max_distance` fall back to the Glover canonical HRF generated at the montage's own `sfreq`.

**Parameters:**
- `max_distance` (`float`) — Maximum Euclidean distance (in MNE coordinate units, meters) for a library HRF to be attached to an optode
- `verbose` (`bool`) — Print search details for each channel

**Returns:** `None`

---

### `montage.branch(**kwargs)`

Return a new montage containing only channels whose HRF context matches **all** of the requested keyword filters. Values within a single kwarg are ORed.

```python
# Match any channel whose HRF was tagged task='flanker'
flanker_montage = montage.branch(task='flanker')

# Match channels tagged both task='flanker' AND stimulus='checkerboard'
filtered = montage.branch(task='flanker', stimulus='checkerboard')
```

**Parameters:**
- `**kwargs` — Context key/value pairs to match against each channel's HRF `context` dict

**Returns:** A new `montage` with deep-copied HRF nodes for each matching channel.

---

### `montage.save(filename='montage_hrfs.json')`

Save the montage's HRF estimates to a JSON file for later loading via `load_montage()`.

**Parameters:**
- `filename` (`str`) — Output file path, default `'montage_hrfs.json'`

---

### `montage.correlate_hrf(plot_filename='montage_correlation.png')`

Compute Spearman rank correlations between all HbO and HbR channel HRF estimates, saving a correlation matrix plot (PNG) and the raw matrix as `correlation_matrix.json` in the current working directory.

**Returns:** `np.ndarray` — correlation matrix, shape `(n_hbo, n_hbr, 2)` where `[:,:,0]` is the correlation coefficient and `[:,:,1]` is the p-value

---

### `montage.correlate_canonical(plot_filename='canonical_correlation.png', duration=30.0)`

Correlate each channel's HRF with a double-gamma canonical HRF to assess fit quality. Requires a configured montage with at least one channel.

**Raises:**
- `ValueError` — montage has no configured channels (call `configure()` or `estimate_hrf()` first)

---

## `localize_hrfs(nirx_obj, max_distance=0.01, verbose=False, **kwargs)`

Convenience function that creates a `montage`, attaches context kwargs, and immediately runs spatial HRF localization. Use this when you have no subject-specific HRF estimates and want to rely on the community library.

**Parameters:**
- `nirx_obj` (`mne.io.Raw`) — fNIRS scan
- `max_distance` (`float`) — Maximum search radius for library HRF attachment
- `verbose` (`bool`) — Print localization details
- `**kwargs` — Context metadata (same keys as `montage.__init__`)

**Returns:** Configured `montage` with library HRFs attached

**Example:**
```python
montage = hrf.localize_hrfs(scan, max_distance=0.01, task="flanker", age_range=[5, 10])
```

---

## `load_montage(json_filename, rich=False, **kwargs)`

Load a previously saved montage from a JSON file. Per-entry schema validation runs on load; if any entry is malformed the whole load aborts and raises a `ValueError` naming the offending entry key and field. The caller never receives a half-populated montage.

**Parameters:**
- `json_filename` (`str`) — Path to JSON file created by `montage.save()`
- `rich` (`bool`) — If `True`, load full per-subject estimates and locations. If `False` (default), load only mean and std.
- `**kwargs` — Context metadata to apply to the loaded montage

**Raises:**
- `ValueError` — JSON is empty, not a dict, or first entry missing `sfreq`
- `ValueError` — A per-entry field (`hrf_mean`, `hrf_std`, `sfreq`, `location`, `context`, `context.duration`, and with `rich=True` also `estimates`, `locations`) is missing. The wrapped exception preserves the original cause via `__cause__`.

**Returns:** Configured `montage`

**Example:**
```python
try:
    montage = hrf.load_montage("flanker_study_hrfs.json", rich=False)
except ValueError as e:
    print(f"Failed to load: {e}")
    print(f"Underlying cause: {e.__cause__}")
```

---

## `preprocess_fnirs(scan, deconvolution=False)`

Preprocess a raw fNIRS scan using the standard HRFunc pipeline. Returns a **new** MNE Raw object — capture the return value.

**Pipeline steps:**
1. Convert to optical density
2. Scalp coupling index → mark bad channels (threshold < 0.5)
3. Interpolate bad channels
4. TDDR motion correction
5. *(deconvolution=True only)* Polynomial detrend, order 3
6. Beer-Lambert Law → hemoglobin concentrations
7. Baseline correction
8. *(deconvolution=False only)* Bandpass filter 0.01–0.2 Hz

**Parameters:**
- `scan` (`mne.io.Raw`) — Raw fNIRS scan in optical density or raw intensity format
- `deconvolution` (`bool`) — If `True`, apply detrending instead of bandpass filtering (appropriate for deconvolution pipelines)

**Returns:** Preprocessed `mne.io.Raw` (new object), or `None` if all channels were marked bad

**Example:**
```python
preprocessed = hrf.preprocess_fnirs(scan, deconvolution=True)
if preprocessed is None:
    print("All channels bad — skipping subject")
```

---

## `tree`

The underlying k-d tree spatial data structure indexed by optode (x, y, z) coordinates. Most users interact with it indirectly through `montage`. Use directly when building custom HRF libraries.

### `tree(hrf_filename=None, **kwargs)`

**Parameters:**
- `hrf_filename` (`str`, optional) — JSON file to load HRFs from on initialization
- `**kwargs` — Context metadata for filtering

**Key methods:**

| Method | Description |
|--------|-------------|
| `insert(hrf)` | Insert an HRF node into the tree. No canonical sentinel is created; the tree is a pure kd-tree of user HRFs. |
| `get_canonical_hrf(oxygenation, sfreq, duration)` | Lazily generate and cache a Glover canonical HRF at the requested sample rate and duration. Sentinel location `[359, 359, 359]`. |
| `nearest_neighbor(optode, max_distance, verbose=False)` | Find the spatially nearest HRF. Returns `(HRF, distance)` on match, `(None, float("inf"))` on miss. Callers wanting a canonical fallback should call `get_canonical_hrf()`. |
| `radius_search(optode, radius)` | Find all HRFs within Euclidean radius |
| `filter(similarity_threshold=0.95, **kwargs)` | Remove HRFs below context similarity threshold |
| `branch(**kwargs)` | Return a new sub-tree containing only nodes matching all supplied context kwargs (AND semantics, values-within-kwarg ORed). Empty kwargs returns an empty sub-tree. |
| `gather(node, oxygenation=None)` | Collect all nodes as a JSON-serializable dict. Returns `{}` on `node=None`. |
| `save(filename)` | Save tree to JSON |
| `split_save(hbo_filename, hbr_filename)` | Save HbO and HbR HRFs to separate files |
| `merge(other_tree)` | Merge another tree into this one. Nodes are deep-copied so the source and destination remain independent after the merge. |
| `delete(hrf)` | Delete a node by spatial position using the standard kd-tree delete-by-copy algorithm |
| `compare_context(first, second, context_weights=None)` | Similarity score `0.0–1.0` between two context dicts. Scalar values are auto-wrapped; missing keys in the second context count as zero matches. |

---

## `hasher`

Open-addressed hash table used internally by `tree.branch()` to look up nodes by their context values (not keys).

**Contract:**
- `hasher.add(key, pointer)` — append pointer to the key's slot list. Duplicate `(key, pointer)` pairs are deduplicated by identity.
- `hasher.search(key)` — returns a `list` of pointers. Empty list on miss. The returned list is a shallow copy so callers can mutate without affecting the hasher.

Populated by `load_hrfs` and `load_montage` with the VALUES from each channel's context dict (e.g., `'flanker'`, `'checkerboard'`, a DOI string). Not by context dict KEYS.

---

## `lens`

Signal quality observer. Use to compare preprocessed and deconvolved fNIRS data.

### `lens(working_directory=None, sfreq=7.81)`

**Parameters:**
- `working_directory` (`str`, optional) — Directory for saving plots and CSV output. Defaults to CWD.
- `sfreq` (`float`) — Fallback sampling frequency used if `compare_subject` has not been called. Overwritten with the actual scan's `info['sfreq']` on first compare.

**Key methods:**

| Method | Description |
|--------|-------------|
| `compare_subject(subject_id, raw_nirx, preproc_nirx, deconv_nirx, events)` | Run all QC metrics for one subject |
| `compare_subjects()` | Aggregate metrics across subjects and generate group plots |
| `calc_snr(subject_id, nirx, state, signal_band=(0.03, 0.1), noise_bands=None)` | Compute signal-to-noise ratio via Welch PSD. `noise_bands` defaults to `[(0.0, 0.01), (0.1, 0.5)]`. |
| `calc_skewness_and_kurtosis(subject_id, nirx, state)` | Compute skewness and kurtosis per channel |
| `calc_sci(subject_id, data, state)` | Compute Scalp Coupling Index |
| `save(output_dir)` | Save all metrics to CSV files |

---

## Context System

All context keys filter which pre-trained HRFs are retrieved from the community library. Internally the hasher keys on **values** (task name, DOI, etc.) so `tree.branch(task='flanker')` returns every HRF whose context dict contained `'flanker'`.

| Key | Type | Description |
|-----|------|-------------|
| `doi` | `str` | Study DOI |
| `study` | `str` | Study name/identifier |
| `task` | `str` | Task name (e.g., `'flanker'`, `'stroop'`, `'rest'`) |
| `conditions` | `list` | Experimental conditions |
| `stimulus` | `str` | Stimulus modality |
| `intensity` | `float` | Stimulus intensity |
| `duration` | `float` | HRF duration (seconds) |
| `protocol` | `str` | Protocol description |
| `age_range` | `list` | `[min_age, max_age]` |
| `demographics` | `str` | Demographic descriptor |

Context keys with `None` values are excluded from similarity comparisons. Pass context as kwargs to `montage()`, `localize_hrfs()`, or `load_montage()`.

---

## Units Convention

**HRFunc outputs estimated HRFs and deconvolved neural activity in arbitrary units (a.u.)**, matching the fMRI BOLD analysis convention. Specifically:

- `estimate_hrf` z-scores the signal before the least-squares solve and does **not** denormalize back to absolute hemoglobin concentrations. Returned HRF traces are in z-score space.
- `estimate_activity` peak-normalizes the HRF kernel before using it as a deconvolution template. Returned neural activity is a relative deviation from baseline, not an absolute metric.

This is intentional. Treat HRF shapes and neural activity traces as relative templates for comparison across conditions and subjects, not as absolute magnitudes.

---

## Error Handling

| Exception | When raised |
|-----------|-------------|
| `ValueError` | Invalid duration (must be > 0), empty events, lmbda <= 0, malformed JSON schema, unconfigured montage on `correlate_canonical` |
| `TypeError` | Non-string passed to `standardize_name` or `_is_oxygenated` |
| `FileNotFoundError` | JSON file not found |
| `LookupError` | Cannot determine channel oxygenation from name/wavelength |

`preprocess_fnirs()` and `montage.estimate_activity()` both return `None` when all channels are bad — **always check the return value**.

---

## Supported fNIRS Formats

HRFunc accepts any `mne.io.Raw` object. Load your data through MNE:

```python
import mne

# sNIRF
scan = mne.io.read_raw_snirf("subject.snirf")

# NIRX
scan = mne.io.read_raw_nirx("path/to/nirx_directory")

# FIF
scan = mne.io.read_raw_fif("subject.fif")
```

---

## Channel Naming

Channel names are standardized internally to the format `{source}_{detector}_{hbo|hbr}` (e.g., `s1_d1_hbo`). Any separator (`-`, `_`, space) and any case is accepted on input. Inputs shorter than 3 characters raise `ValueError`.

Oxygenation is inferred from:
- The `hbo`/`hbr` suffix in the channel name, OR
- The wavelength suffix (760–780 nm → HbR, 830–850 nm → HbO)
