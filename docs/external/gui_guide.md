# HRFunc Desktop GUI — User Guide

A point-and-click workflow for fNIRS HRF estimation and neural-activity
recovery without writing Python.

The GUI ships in v1.3.0 as an opt-in extra on top of the existing
`hrfunc` library. Library users keep their Python workflow unchanged;
researchers who prefer a desktop app get the same pipeline through a
native window.

---

## Installing

The GUI requires three extra dependencies — `nicegui`, `plotly`, and
`pywebview`. Install via the `[gui]` extra:

```bash
pip install hrfunc[gui]
```

Python ≥ 3.8 is required (same as the core library). The GUI runs as a
native desktop window on macOS, Linux, and Windows.

---

## Launching

The `hrfunc` console script always launches the GUI:

```bash
hrfunc                       # opens to the welcome screen
hrfunc /path/to/study        # opens with that folder pre-loaded
hrfunc subject_01.snirf      # opens with a single file pre-loaded
hrfunc --version             # prints version and exits
hrfunc --help                # shows CLI help (also: hrfunc help)
```

There is no `--gui` flag. The bare `hrfunc` command is reserved as the
GUI entry; future library subcommands (e.g. `hrfunc estimate`) will be
added as positional commands.

### System menu integration

After your first `hrfunc` launch, the welcome screen offers to add
HRFunc to your system menu so you can open it without the terminal.
"Yes, add it" puts a clickable launcher in Spotlight (macOS), the
Start menu (Windows), or Activities (Linux); "Not now" defers the
question to the next launch; "Don't ask again" silences the prompt
without installing.

You can also manage the shortcut from the command line at any time:

```bash
hrfunc install-shortcut      # add HRFunc to your system menu
hrfunc uninstall-shortcut    # remove the system menu entry
```

The shortcut points at the same `hrfunc` console script the welcome
prompt would have installed, so the user experience after install is
identical no matter which path you took.

---

## The welcome screen

When the GUI opens with no folder pre-loaded, you see three paths:

1. **Open my data** — pick a folder on disk. HRFunc scans for SNIRF,
   NIRx, and FIF files (with BIDS subject/session metadata if your
   folders follow the convention). Opens the workspace once the scan
   completes.
2. **Browse HRF library** — explore the bundled literature-derived HRF
   databases (`hbo_hrfs.json` / `hbr_hrfs.json`) without any of your own
   data. Opens the `/library` page directly. See [HRF
   Library](#hrf-library) below for the full walkthrough.
3. **Recent projects** — re-open a folder you scanned previously. The
   scan manifest is cached on disk per the XDG cache convention
   (`~/Library/Caches/hrfunc/` on macOS, `~/.cache/hrfunc/` on Linux,
   `%LOCALAPPDATA%\hrfunc\` on Windows) so the welcome dialog remembers
   recently-opened folders.

Folder scans use AND-of-markers detection for NIRx (a directory needs
both `*_probeInfo.mat` and `*.wl1` to be recognized) and a header sniff
via `mne.io.read_info` for FIF so MEG/EEG-only `.fif` files are skipped.

---

## The workspace

After a successful folder scan, the workspace opens. Three resizable
panes:

```
┌─────────────┬──────────────────────────────────┬──────────────┐
│  Dataset    │  [Inspect][Quality][Preprocess]                  │
│  tree       │  [HRFs][Activity][HRtree][Export]                │
│  (subjects, │  ─────────────────────────────                   │
│   sessions, │   active tab content                             │
│   scans)    │                                  │  Manifest     │
│             │                                  │  summary      │
└─────────────┴──────────────────────────────────┴──────────────┘
```

- **Left pane** — the dataset tree, grouped by BIDS subject → session →
  scan (or by parent directory for non-BIDS folders). A filter input
  above the tree narrows the visible scans by case-insensitive
  substring match against display name or path. Click a scan to load it.
- **Center pane** — seven tabs that drive the workflow. Inspect,
  Preprocess, HRFs, and Activity are real in v1.3.0; Quality, HRtree,
  and Export render placeholders pointing at the sprint that fills them.
- **Right pane** — manifest summary (root path, scan count, scan-format
  histogram, scan time).

When you click a scan in the left tree:

1. The Inspect tab updates immediately with metadata from the scan
   index (format, path, channel count and sampling rate when available,
   BIDS components).
2. A background task loads the MNE Raw file. While it loads, the
   "Recording" section in Inspect shows a "Loading recording…" spinner.
3. Once loaded, the Recording section populates with three collapsible
   expansions: **Channels** (the full channel-name list), **Probe
   layout** (a 2D sensor plot via MNE), **Events** (a table of
   annotation descriptions / onsets / durations from the scan).

The loaded MNE Raw lives in an in-memory LRU cache (size 3) so
switching scans is fast — three recent scans stay in memory.

---

## Recommended workflow

The four real tabs form a left-to-right pipeline. Each tab consumes
state produced by the previous one:

1. **Inspect** — sanity-check the scan loaded correctly. Confirm the
   probe layout and event table match what you expect from the
   acquisition protocol.
2. **Preprocess** — convert the raw scan to preprocessed haemoglobin
   concentration. The full pipeline (optical density → scalp coupling →
   bad-channel interpolation → motion correction → Beer-Lambert →
   baseline correction → bandpass filter) runs in a single click.
3. **HRFs** — estimate per-channel hemodynamic response functions from
   the preprocessed scan and its event annotations. Toeplitz mode
   deconvolves the user's data; canonical mode renders a reference
   SPM-style double-gamma HRF for comparison.
4. **Activity** — deconvolve the preprocessed scan using either the
   user's estimated HRFs (toeplitz mode, requires running HRFs first)
   or the bundled canonical HRFs (canonical mode, no HRFs estimation
   required). Output is a Raw with neural-activity values in place of
   haemoglobin values.

---

## Preprocess tab

| Control | Effect |
|---|---|
| **Run full pipeline** | Executes `preprocess_fnirs` and stores the result in the processed-Raw cache. Reuses any previously-loaded Raw. |
| **Deconvolution mode** switch | Adds polynomial detrend after TDDR; skips the GLM-friendly bandpass at the end. Use this mode if you will run the HRFs tab in toeplitz mode. |
| **Motion correction (TDDR)** switch | Default on. Diagnostic only — skipping motion correction on real fNIRS data is not recommended for publishable analyses. |
| **Beer-Lambert conversion** switch | Default on. Skipping leaves the output in optical-density units. Diagnostic only. |
| **Baseline correct** switch | Hidden in GLM mode (always applied per library default). Visible in deconvolution mode where you can opt out. |

A diagnostic disclaimer under the options card reminds users that the
default settings reproduce the library's canonical pipeline and are
publication-ready; non-default toggles are for diagnostic exploration
only.

After a successful run, a before/after preview renders the first four
channels: raw signal on top, preprocessed signal on the bottom, for
quick visual sanity-checking.

---

## HRFs tab

Two modes:

### Toeplitz mode (default)

Deconvolves per-channel HRFs from the preprocessed scan via
`montage.estimate_hrf`. Requires the Preprocess tab has been run.

| Control | Effect |
|---|---|
| **Events** checkboxes | Select which annotation descriptions count as event onsets. Defaults to every distinct description in the scan. |
| **Regularization (lambda)** slider | Log scale from 1e-5 to 1e-1, step 1 decade. Library default 1e-3. Higher lambda = smoother HRF; lower = noisier but more detail. |
| **Duration (seconds)** | The length of the HRF window. Default 30 s. |
| **Estimate HRFs** button | Dispatches `estimate_hrf` in the background. Progress bar updates per channel via `progress_callback`. |

A note below the duration field reminds you that events in the first
`0.15 × duration` seconds (~4.5 s by default) are silently dropped by
the toeplitz edge-expansion window — important for trial-onset timing.

### Canonical mode

Skips estimation entirely. Renders the SPM-style double-gamma HRF
(gamma with `a=6` minus a scaled gamma with `a=16`, peak normalized to
1.0). Useful as a reference shape or when no events are available.

Note that the GUI's canonical HRF is time-indexed (peak at ~5 s
regardless of scan sampling rate), which deliberately differs from
the library's internal `correlate_canonical` (sample-indexed, scan-rate
dependent).

After a successful run, the gallery renders a **clickable channel
grid** — one mini-plot per channel — and a detail panel below showing
the currently-focused channel's full trace with ±1 standard-deviation
shading. Click any mini-plot to focus a different channel; the grid
highlights the active selection.

---

## Activity tab

Recovers neural-activity time series from the preprocessed scan via
`montage.estimate_activity`.

| Control | Effect |
|---|---|
| **Toeplitz / canonical** radio | Toeplitz uses the per-channel HRFs you estimated in the HRFs tab. Canonical pulls from the bundled HRF library for each channel. |
| **Regularization (lambda)** slider | Log scale from 1e-6 to 1e-1. Library default 1e-4 (one decade smaller than HRF estimation). |
| **Estimate activity** button | Dispatches `estimate_activity` in the background. Progress bar updates per channel. |
| **Preview channel** dropdown | Pick which channel's overlay to display after the run. |

Toeplitz mode is disabled when:
- No HRFs have been estimated yet (run the HRFs tab first), OR
- The HRFs tab last produced a canonical reference shape (re-run in
  toeplitz mode), OR
- The HRFs in memory came from a *different* scan than the one
  currently selected (a guard rail — applying scan A's HRFs to scan
  B's Raw silently produces wrong results because the library matches
  channels by name).

The Activity preview renders a `lens.plot_nirx`-style overlay:
preprocessed signal in red-dashed (rescaled to match the y-range) and
the deconvolved neural-activity signal in blue, with event markers as
orange vertical lines.

---

## HRF Library

The `/library` page is the **Browser** persona's entry point — a
researcher who wants to explore HRFunc's bundled literature HRFs without
opening their own data. Reach it via the welcome screen's "Browse HRF
library" card, or by navigating directly.

Three resizable panes:

```
┌──────────────┬──────────────────────────────┬──────────────┐
│  Filter      │  HRtree (plotly 3D)          │  Detail      │
│  (context    │   - Points = HRF location    │  (selected   │
│   form)      │   - Color = HbO / HbR        │   HRF info,  │
│              │   - Hover for context        │   trace plot)│
│              │   - Click to select          │              │
└──────────────┴──────────────────────────────┴──────────────┘
```

The library loads two bundled databases on first visit — `hbo_hrfs.json`
and `hbr_hrfs.json`, both shipped inside the `hrfunc` package — into
their k-d tree containers. The trees stay in memory for the rest of the
session.

### Filter pane

The left sidebar exposes six common context fields as text inputs:

- **task** — e.g. `flanker`, `nback`, `rest`
- **doi** — paper identifier or `temp` for not-yet-published entries
- **study** — internal study name
- **demographics** — e.g. `children`, `adults`, `women`
- **stimulus** — visual / auditory / etc.
- **conditions** — experiment condition names

Matching is **case-insensitive substring**. Leave a field blank to
ignore it. Click **Apply** to refresh the viz; **Reset** clears all
fields. A live count below the buttons shows `N / M HRFs match` where M
is the total bundled count.

For context fields whose values in the database are lists (e.g.
`conditions: ["congruent", "incongruent"]`), the filter matches if any
list entry contains the needle.

### HRtree viz pane (center)

Plotly 3D scatter of HRFs positioned by their `(x, y, z)` optode
coordinates. Two traces: HbO (red) and HbR (blue), so you can toggle
oxygenation visibility from the plotly legend.

- **Hover** — shows the HRF key plus its task / doi / study /
  demographics for quick scanning.
- **Click** — sets the selected HRF; the right pane updates with the
  full context dict and a trace preview.
- **Rotate / zoom / pan** — standard plotly 3D controls (drag to
  rotate, scroll to zoom, shift-drag to pan).

HRFs without a recorded 3D location are excluded from the viz rather
than clustered at the origin (the spatial display only makes sense for
HRFs with measured coordinates).

### Detail pane (right)

Empty until you click an HRF in the viz. Once selected, shows:

- The HRF key (e.g. `s1_d1_hbo-doi/10.1234/abcd`)
- Oxygenation (HbO / HbR), sampling rate, location, trace length
- Full context dict — every non-None field from the database entry
- A trace preview plot in seconds (with ±1 standard-deviation shading
  when `hrf_std` is available)

### Library limitations to know about

- **Read-only** — v1.3.0 does not let you delete, insert, or merge
  HRFs from the GUI. Use the `hrfunc.tree` Python API for those
  operations.
- **Filter is view-only** — applying a filter does not modify the
  underlying tree on disk. Re-opening the page resets to "all HRFs
  visible".
- **No cross-reference with the workspace** — you cannot drag an HRF
  from the library into your workspace's montage yet. Use
  `hrfunc.localize_hrfs` in Python for that flow.

---

## Caching behavior

The GUI uses two LRU caches of size 3 (so up to 3 scans stay in memory
across "previous / current / next" navigation):

- **`raw_cache`** — source MNE Raw objects loaded from disk.
- **`processed_cache`** — outputs of `preprocess_fnirs`.

Both caches survive scan-tab switching. They are cleared by closing the
project (returning to the welcome page).

> **Known limitation:** the processed-cache key is the scan path only,
> not (path, options). If you preprocess a scan with one set of toggles,
> change the toggles, and re-run, the new result overwrites the cache
> silently. The HRFs and Activity tabs read whichever preprocess output
> was produced most recently. This is mitigated in v1.3.0 by the
> "diagnostic-only" disclaimer; a future release may key the cache on
> the full options hash.

---

## Background tasks

Long-running operations (Preprocess, HRFs, Activity) run on a single
background worker. While one is running, the workspace sets a
`busy=True` flag that disables the Run buttons in all three panels.
This prevents queuing overlapping pipeline work that would conflict on
the shared `processed_cache` / `state.montage` resources.

Scan loads (triggered by clicking the dataset tree) bypass the busy
gate so you can navigate freely during a long estimation. The loaded
Raw goes into `raw_cache` whenever the load completes — even mid-
estimation.

---

## Troubleshooting

### "No dataset loaded" on the workspace

You navigated to `/workspace` without scanning a folder first. Click
"Back to welcome" and use "Open my data" to scan a folder.

### Welcome screen says "Recent projects" is empty after I've used the GUI

Recent projects are read from the XDG cache directory. If the cache
directory is unwritable (read-only home directory, restrictive
permissions), manifests aren't persisted between sessions. Check that
the cache path resolves and is writable:

```bash
python -c "import platformdirs; print(platformdirs.user_cache_dir('hrfunc'))"
```

### Inspect tab probe layout says "Probe layout unavailable for this scan"

The MNE `plot_sensors()` call failed — typically because the scan has
no montage / no sensor positions. The channel and event sections still
work. The probe layout is a convenience; downstream tabs do not depend
on it.

### Preprocess tab says "Cannot preprocess — all channels marked bad"

Every channel scored a scalp-coupling index below 0.95, which the
library's `preprocess_fnirs` treats as an unusable scan. This usually
indicates a probe-placement or hardware issue with the recording rather
than a software problem.

### HRFs tab shows "No events found in the preprocessed scan"

The preprocessed scan has no `mne.Annotations`. This can happen if your
SNIRF or NIRx file didn't include event triggers, or if the format
exporter dropped them. Either re-export with events included, or switch
to canonical mode (which doesn't need events).

### HRFs tab "Estimate HRFs" button stays disabled

Three preconditions must be met:

1. A scan is selected,
2. The Preprocess tab has been run for that scan (so the processed Raw
   is cached),
3. At least one event description is checked.

The button label clarifies which precondition is missing in the row
beneath it.

### Activity tab "Estimate activity" button stays disabled in toeplitz mode

Toeplitz Activity needs HRFs estimated from the *currently-selected
scan*. Three guard conditions can disable it:

1. No Montage in memory — run the HRFs tab in toeplitz mode first.
2. Montage is in memory but came from canonical mode — re-run HRFs in
   toeplitz mode.
3. Montage came from a *different scan* — switch back to that scan, or
   re-run HRFs on the current scan.

Canonical Activity has no such constraints; it only needs the
Preprocess tab to have been run.

### Activity preview says "Preview unavailable"

The matplotlib render failed (most often: the channel you picked was
dropped during estimation and the deconvolved Raw has fewer channels
than the processed Raw). Pick a different channel from the dropdown.

### Export tab is empty after I clicked Save

Each exporter only enables when its source data is in state. If the
button is greyed out, the row's "italic hint" line tells you what to
do next (e.g. "Run the HRFs tab in toeplitz mode first" or "Compute
metrics in the Quality tab first"). The Save button is disabled, not
hidden — so you can see all five exporters at a glance even when only
one is ready.

### Save dialog doesn't appear

The Export panel uses `pywebview`'s native save/folder dialogs. Same
fallback as the welcome-screen folder picker — if no native window is
attached (e.g. launching via `python -m hrfunc.gui.app` instead of the
`hrfunc` console script), the panel shows a NiceGUI toast telling you
to launch in native mode.

### SNIRF export of haemoglobin / activity data

The Export panel writes whatever's in the source Raw to the chosen
SNIRF file. For round-trip workflows (export → re-load in MNE), prefer
`.fif` over `.snirf` when exporting **preprocessed** or **activity**
data — FIF preserves the full MNE channel-type metadata (`hbo`, `hbr`,
plus user-added types), whereas SNIRF's data-type indices may not
round-trip the haemoglobin / neural-activity unit information cleanly.
If you only need the data values (e.g. for analysis in software that
re-derives types from channel names), SNIRF is fine.

### "N channels dropped during deconvolution" in the activity-save toast

`estimate_activity` silently drops channels whose Tikhonov-regularized
lstsq solve fails or times out (default 30 s ceiling per channel). The
exported activity Raw has these channels removed. The save-toast now
surfaces the count so you know if the output has fewer channels than
the preprocessed source. If the count is non-zero, check
`state.last_error` and the underlying console output for which
channels failed — typically caused by pathological lstsq matrices on
all-bad-channel inputs.

### GUI crashes on launch with a port-binding error

NiceGUI defaults to port 8080, but HRFunc asks the OS to pick a free
loopback port at launch time, so the bound port should never collide
with another service. If you still see a "port in use" error, the most
likely cause is a host firewall blocking loopback ports — check that
`127.0.0.1` is reachable on arbitrary high ports.

### Tests pass but the GUI window doesn't open

The GUI uses `pywebview` to host the NiceGUI page in a native window.
On Linux, `pywebview` requires a system GTK or Qt backend installed.
On macOS and Windows, it should work out of the box. If you see a
console warning about missing backends, install one:

```bash
# Linux (Ubuntu/Debian)
sudo apt-get install python3-gi gir1.2-webkit2-4.0

# Or use the Qt backend
pip install pyqt5 QtWebEngine
```

---

## Limitations to know about

These are documented design constraints in v1.3.0 — not bugs:

- **One scan worked at a time.** The dataset tree lets you navigate
  freely, but `state.montage` and `state.activity_raw` are single-slot.
  Switching scans then running Activity in toeplitz mode is blocked
  until you re-run HRFs on the new scan.
- **Edge-expansion is fixed.** The HRFs tab uses the library's default
  `edge_expansion=0.15` (drops events in the first ~4.5 s of a 30 s-
  duration HRF). No user control yet.
- **Lambda is discrete.** The sliders step in single decades on a log
  scale (1e-5, 1e-4, …). Intermediate values like 5e-4 aren't
  accessible from the GUI; use the Python API if you need them.
- **Cancel during estimation is not supported.** Once you click Run,
  the task runs to completion. Scans typically estimate in a few
  seconds; if you need to abort a longer run, close the window.

---

## See also

- [`docs/external/api_reference.md`](api_reference.md) — programmatic
  Python API (the `hrfunc.montage` class, `estimate_hrf`,
  `estimate_activity`).
- [`docs/external/workflow_examples.md`](workflow_examples.md) — end-to-
  end examples in the Python API.
- [www.hrfunc.org](https://www.hrfunc.org) — guides and demo videos.
