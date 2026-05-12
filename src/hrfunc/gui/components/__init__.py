"""Reusable UI components for the HRfunc GUI.

Each module here exports one component (or one closely related family) that
pages compose. Components encapsulate rendering + event wiring; they do not
own state — they read from and write to the shared ``AppState``.

Adding a component = create a new module, export a top-level ``render`` or
factory function, and import it from the page that uses it.
"""
