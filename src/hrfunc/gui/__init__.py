"""HRfunc desktop GUI — NiceGUI-based interface for non-tech researchers.

The GUI is an optional install (`pip install hrfunc[gui]`) that adds NiceGUI,
plotly, and pywebview. The `hrfunc` CLI entry point launches `hrfunc.gui.app.main`,
which opens a native desktop window backed by NiceGUI's pywebview integration.

Public entry points (rarely imported directly; the CLI wires them up):
    main(argv=None) -> int   - CLI entry (see hrfunc.gui.app)
    state                     - module-level AppState singleton (see hrfunc.gui.state)

This subpackage is **not** imported by the core library — `import hrfunc` does
not load NiceGUI. Users without the [gui] extras can still use every public
API in hrfunc and hrfunc.io.
"""
