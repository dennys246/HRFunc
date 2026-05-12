"""HRfunc CLI helpers shared between subcommands.

The bare ``hrfunc`` command launches the GUI (see ``hrfunc.gui.app:main``);
the subcommands live in this package so they can be imported lazily by
the launcher without dragging GUI dependencies into the import graph.
"""
