## v1.1.0
- Final build for release with basic functionality
- Released in conjunction with HRfunc paper

## v1.1.1
- Exposed the preprocess_fnirs function in the library

## v1.3.0
- New: NiceGUI-based desktop GUI installed via `pip install hrfunc[gui]`
- New: `hrfunc` console script launches the GUI; `hrfunc <path>` preloads
- New: 3-path welcome screen (Open my data / Browse HRF library /
  Recent projects) with XDG-cache-backed recent-folder list
- New: BIDS-aware dataset tree with case-insensitive substring filter
- New: Inspect tab — channel list, 2D probe layout, event-annotation
  table; loads MNE Raw lazily via an in-memory LRU(3) cache
- New: Preprocess tab — full pipeline button + diagnostic stage toggles
  + before/after preview; results stored in a separate `processed_cache`
- New: HRFs tab — toeplitz deconvolution with multi-event picker,
  log-scale lambda slider, duration field, progress bar; canonical
  mode renders an SPM-style double-gamma reference HRF
- New: Activity tab — toeplitz (reuses estimated HRFs) and canonical
  (uses bundled HRF library) modes; lens.plot_nirx-style overlay
  preview with event markers
- New: Quality tab — per-scan SNR / skewness / kurtosis / SCI metrics
  across raw / preprocessed / deconvolved stages plus a dataset-wide
  aggregate that walks every scan in the manifest
- New: HRF Library page at `/library` — three-pane Browser-persona
  flow with Context Filter sidebar, plotly 3D HRtree viz, and per-HRF
  detail pane with trace preview
- New: HRF gallery — HRFs-tab preview replaced by a clickable channel
  grid (one mini-plot per channel) with a per-channel detail panel
  showing full trace + ±1 std shading
- New: Export tab — saves processed Raw (SNIRF/FIF), activity Raw
  (SNIRF/FIF), montage HRFs (JSON via montage.save), per-channel HRF
  plot PNGs to a folder, and quality metrics as a flat CSV (one row
  per scan × stage)
- New: Cross-component event bus (`scan_selected`, `scan_loaded`,
  `preprocess_done`, `hrf_estimated`, `activity_estimated`,
  `quality_computed`, `library_filter_changed`,
  `library_selection_changed`) for panel reactivity
- New: Folder-scan I/O subsystem (`hrfunc.io.scan_folder`,
  `classify_path`, `RawCache`) reusable from the Python API
- New: `progress_callback` kwarg added to `montage.estimate_hrf` and
  `montage.estimate_activity` for non-GUI progress tracking
- New: `hrfunc install-shortcut` / `hrfunc uninstall-shortcut`
  subcommands add or remove a system-level launcher (Spotlight on
  macOS, Start menu on Windows, Activities on Linux) so non-coder
  researchers can open HRfunc like any other desktop app. First GUI
  launch prompts the user to install the shortcut automatically.
- See [docs/external/gui_guide.md](docs/external/gui_guide.md) for the
  full GUI walkthrough and troubleshooting guide