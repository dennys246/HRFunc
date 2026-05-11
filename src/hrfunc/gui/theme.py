"""Quasar / Tailwind theming for the HRFunc GUI.

NiceGUI ships Quasar (Vue) and Tailwind under the hood. This module sets the
brand color tokens, default dark mode, and shared layout primitives so every
page has a consistent look without duplicating styling code.

`apply_theme()` must be called once per page render — typically at the top of
each `@ui.page` handler — to set the dark-mode default and inject the brand
palette via Quasar's `ui.colors`.

Color choices reflect the v1.3.0 plan ("modern, scientific dashboard"):
- Primary:   indigo-500   - main accent for buttons, active tabs, sliders
- Secondary: violet-400   - secondary accent for HRF traces and HRtree edges
- Accent:    cyan-400     - highlights, links, focus rings
- Positive:  emerald-400  - good-quality channel indicators
- Negative:  rose-400     - bad channels, errors, flagged data
- Warning:   amber-400    - in-progress states, cautionary metrics
- Info:      sky-400      - informational banners
- Dark:      slate-900    - background base for dark mode

Light mode is supported via a toggle but is not the default. Researchers
typically work in dim environments; dark mode also makes plotly's default
plot palette read better.
"""

from __future__ import annotations

from nicegui import ui


# Brand palette — hex strings matching Tailwind 500-level tones for legibility.
COLORS = {
    "primary": "#6366f1",     # indigo-500
    "secondary": "#a78bfa",   # violet-400
    "accent": "#22d3ee",      # cyan-400
    "positive": "#34d399",    # emerald-400
    "negative": "#fb7185",    # rose-400
    "warning": "#fbbf24",     # amber-400
    "info": "#38bdf8",        # sky-400
    "dark": "#0f172a",        # slate-900
}


def apply_theme(dark: bool = True) -> None:
    """Apply the HRFunc color palette and (optionally) enable dark mode.

    Call once per page handler before constructing UI elements. Subsequent
    calls are idempotent — Quasar caches the colors and `ui.dark_mode` is a
    no-op when already in the requested state.

    Args:
        dark: If True (default), enable dark mode. The user-facing toggle in
            the workspace toolbar can flip this at runtime.
    """
    ui.colors(**COLORS)
    if dark:
        ui.dark_mode().enable()
    else:
        ui.dark_mode().disable()


def page_container():
    """Return a top-level page container with consistent padding and width.

    Standardizes the outer layout so welcome / workspace / library pages all
    share the same gutter and max-width. Use as a context manager:

        with page_container():
            ui.label("contents")
    """
    return ui.column().classes(
        "w-full max-w-screen-2xl mx-auto p-6 gap-4"
    )
