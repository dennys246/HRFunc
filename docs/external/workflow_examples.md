# HRFunc Workflow Examples

Practical end-to-end examples for the most common HRFunc workflows. Assumes you've already installed HRFunc and have a working MNE-Python environment.

For reference-level documentation of each function, see [api_reference.md](api_reference.md).

---

## Prerequisites

```bash
pip install hrfunc mne mne-nirs
```

```python
import mne
import hrfunc as hrf
import numpy as np
```

---

## Workflow 1 — Localize pre-trained HRFs to a new scan

Use this when you don't have subject-specific HRF estimates and want to rely on HRFunc's bundled community library. The library is spatially indexed by optode location, so each channel gets the nearest match.

```python
# Load a scan
scan = mne.io.read_raw_snirf("subject_01.snirf")

# Attach nearest-neighbor HRFs from the bundled library.
# Context kwargs narrow the match (e.g., only flanker-task HRFs for
# subjects aged 18-35).
montage = hrf.localize_hrfs(
    scan,
    max_distance=0.01,         # meters, in MNE coordinates
    task="flanker",
    age_range=[18, 35],
)

# Channels without a library match within max_distance fall back to a
# Glover canonical HRF generated at the scan's own sample rate.
for ch_name, optode in montage.channels.items():
    print(f"{ch_name}: method={optode.context['method']}, "
          f"trace_length={len(optode.trace)}")
```

---

## Workflow 2 — Estimate per-subject HRFs, then deconvolve neural activity

Use this when you have event-aligned fNIRS scans and want to recover the HRFs first, then use them to deconvolve neural activity from the same (or different) scans.

```python
# --- Phase 1: estimate HRFs ---
montage = hrf.montage(doi="10.1000/my-study", task="flanker")

# Per subject: load scan + events, call estimate_hrf
for subject_path, events_path in zip(subjects, events_files):
    scan = mne.io.read_raw_snirf(subject_path)

    # Events must be a binary list (0 or 1 per sample, same length as scan)
    with open(events_path) as f:
        events = [int(line.strip()) for line in f]

    montage.estimate_hrf(
        scan, events,
        duration=30.0,
        lmbda=1e-3,
    )

# Compute per-channel means and std dev across all subjects
montage.generate_distribution(plot_dir="plots/hrfs")

# Save the estimated HRFs for future reuse
montage.save("flanker_study_hrfs.json")


# --- Phase 2: deconvolve neural activity ---
for subject_path in subjects:
    scan = mne.io.read_raw_snirf(subject_path)

    # estimate_activity returns a NEW MNE Raw with deconvolved channels.
    # Capture the return value — the original scan is not mutated.
    deconvolved = montage.estimate_activity(scan, lmbda=1e-4)
    if deconvolved is None:
        print(f"{subject_path}: all channels bad, skipping")
        continue

    deconvolved.save(f"neural_{subject_path.replace('.snirf', '.fif')}")
```

**Important gotchas:**
- `estimate_hrf` and `estimate_activity` both return a new MNE Raw object (or `None`). Always capture the return value.
- If `estimate_activity` drops a channel (timeout, solve failure, zero-trace HRF), that channel is removed from both the output scan AND the montage's `channels` / `hbo_channels` / `hbr_channels` lists. Subsequent calls to `correlate_hrf()` or `generate_distribution()` will only see the remaining channels.
- `estimate_activity` silently falls back to the canonical HRF for any channel whose stored HRF has an empty, `None`, or all-zero trace (with a warning printed). This avoids silent `NaN` propagation.

---

## Workflow 3 — Load a saved montage and apply it to a new scan

```python
# Load previously saved HRFs
montage = hrf.load_montage("flanker_study_hrfs.json")

# Apply to a new scan from the same study
new_scan = mne.io.read_raw_snirf("new_subject.snirf")
deconvolved = montage.estimate_activity(new_scan)
```

**If the JSON is malformed:**
```python
try:
    montage = hrf.load_montage("maybe_broken.json")
except ValueError as e:
    print(f"Failed to load: {e}")
    # The per-entry error message names the offending entry key and field
    print(f"Underlying cause: {e.__cause__}")
```

`load_montage` is transactional: if any entry fails schema validation, the whole load aborts and nothing is returned. You'll never get a half-populated montage.

---

## Workflow 4 — Filter HRFs by experimental context

Use `montage.branch()` (or `tree.branch()`) to get a filtered sub-montage containing only channels whose stored HRFs match the requested context.

```python
# Load a large multi-task montage
all_hrfs = hrf.load_montage("multi_task_library.json")

# Get a sub-montage with only flanker HRFs
flanker_only = all_hrfs.branch(task="flanker")

# AND multiple kwargs
filtered = all_hrfs.branch(task="flanker", age_range=[18, 35])

# Values within a single kwarg are ORed
flanker_or_gonogo = all_hrfs.branch(task=["flanker", "gonogo"])
```

The sub-montage is a deep copy — mutations on the branch don't affect the parent.

---

## Workflow 5 — QC metrics with `lens`

Compare preprocessed and deconvolved data for a subject:

```python
observer = hrf.lens(working_directory="qc_outputs/")

for subject_id, raw_path in subjects.items():
    raw = mne.io.read_raw_snirf(raw_path)
    preproc = hrf.preprocess_fnirs(raw.copy(), deconvolution=False)
    montage = hrf.load_montage("hrfs.json")
    deconv = montage.estimate_activity(preproc.copy())

    events = load_events_for_subject(subject_id)
    observer.compare_subject(
        subject_id,
        raw_nirx=raw,
        preproc_nirx=preproc,
        deconv_nirx=deconv,
        events=events,
    )

observer.compare_subjects()  # Generate group-level plots
```

---

## Workflow 6 — Minimal sanity check (no real data)

Useful for verifying your installation works end-to-end without needing an actual fNIRS scan.

```python
import hrfunc as hrf

# A bare montage loads the bundled library into hbo_tree and hbr_tree
m = hrf.montage()
print(m)  # Reports "unconfigured" state cleanly

# Fetch a canonical HRF directly from the tree (doesn't need a scan)
canonical = m.hbo_tree.get_canonical_hrf(
    oxygenation=True,
    sfreq=10.0,
    duration=30.0,
)
print(f"Canonical HbO at 10 Hz, 30 s: {len(canonical.trace)} samples")
```

---

## Error-handling cheat sheet

| You see | What's going on |
|---------|-----------------|
| `ValueError: duration must be > 0` | Passed `duration=0` or negative to `estimate_hrf`. Pass a positive float. |
| `ValueError: events list must not be empty` | Passed `events=[]`. Your event series has no events. |
| `ValueError: lmbda must be > 0` | Regularization strength must be positive. Use `1e-3` to `1e-4` for typical fNIRS. |
| `ValueError: load_montage: entry 'X' is missing required field 'Y'` | Your saved JSON is missing a field. Check the entry named in the error. |
| `TypeError: standardize_name expected a str` | Passed `None` or a non-string as a channel name. Check your MNE `ch_names`. |
| `ValueError: Channel name '...' is too short` | Channel name is under 3 characters. MNE channel names should carry at least an `hbo`/`hbr` suffix or wavelength digits. |
| `ValueError: correlate_canonical requires a configured montage` | You called `correlate_canonical()` before any channels were added. Call `configure()` or `estimate_hrf()` first. |
| `WARNING: HRF trace for channel X is empty or all-zero` | `estimate_activity` detected a degenerate HRF and fell back to the canonical. Not an error — the deconvolution proceeds. |
| `lstsq exceeded 30s timeout, dropping channel` | A single channel's solve timed out. The channel is dropped from the output scan and the montage. |
| `preprocess_fnirs()` returns `None` | All channels were flagged bad by the scalp coupling index check. Check your data quality. |
| `estimate_activity()` returns `None` | `preprocess_fnirs` inside `estimate_activity` returned `None`. Same cause as above. |

---

## Units reminder

HRFunc intentionally outputs HRF traces and deconvolved neural activity in **arbitrary units (a.u.)**, matching the fMRI BOLD analysis convention. Traces are z-score-normalized inputs and peak-normalized kernels; the absolute magnitude is not meaningful. Treat HRF shapes and activity traces as relative templates for comparison across conditions and subjects. See [api_reference.md](api_reference.md#units-convention) for the full rationale.

---

## Supported scan formats

```python
# sNIRF
scan = mne.io.read_raw_snirf("subject.snirf")

# NIRX
scan = mne.io.read_raw_nirx("path/to/nirx_directory")

# FIF (already-loaded or previously saved)
scan = mne.io.read_raw_fif("subject.fif")
```

HRFunc accepts any `mne.io.Raw` object.
