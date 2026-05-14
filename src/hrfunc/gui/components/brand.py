"""Brand wordmark component — Times New Roman with an italicized suffix.

Used wherever the GUI displays the ``HRfunc`` / ``HRtree`` wordmarks so the
typography stays consistent: small inline in the toolbar, larger in the
Library tab header, etc. The styling mirrors the paper's typesetting (the
italicized half is the "function" / "tree" suffix of the brand name).

Use directly:

    from hrfunc.gui.components.brand import brand
    brand("HRfunc", italic_suffix="func")              # default ~1.5rem
    brand("HRtree", italic_suffix="tree", size_rem=2)  # larger header

The function returns the underlying ``ui.html`` element so callers can chain
further ``.style()`` / ``.classes()`` calls if a specific surface needs more
tuning than the kwargs cover.
"""

from __future__ import annotations

from nicegui import ui


def brand(text: str, *, italic_suffix: str, size_rem: float = 1.5):
    """Render an HRfunc-style wordmark.

    Splits ``text`` so the trailing ``italic_suffix`` is wrapped in ``<em>``
    and the rest renders upright. Both halves share the same Times New
    Roman bold styling so the wordmark reads as one unit. ``ui.html`` (vs
    ``ui.label``) is required because Quasar's label component escapes
    inline tags.

    :param text: The full wordmark, e.g. ``"HRfunc"`` or ``"HRtree"``.
    :param italic_suffix: The trailing substring of ``text`` to italicize,
        e.g. ``"func"`` for ``"HRfunc"``. Must be a suffix of ``text``;
        passing a non-suffix raises ``ValueError`` rather than silently
        producing a malformed wordmark.
    :param size_rem: Font size in rem units. Defaults to ``1.5`` — a
        comfortable toolbar size. Use larger (e.g. 2.5–4.5) for page
        headers or the historical welcome-style hero typography.
    """
    if not text.endswith(italic_suffix):
        raise ValueError(
            f"italic_suffix {italic_suffix!r} is not a suffix of text {text!r}"
        )
    prefix = text[: len(text) - len(italic_suffix)]
    return ui.html(f"{prefix}<em>{italic_suffix}</em>").style(
        'font-family: "Times New Roman", Times, serif; '
        f"font-size: {size_rem}rem; "
        "font-weight: bold; "
        "letter-spacing: -0.025em; "
        "line-height: 1;"
    )
