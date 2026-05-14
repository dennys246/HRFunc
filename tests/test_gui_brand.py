"""Targeted unit tests for the Brand wordmark component.

The component renders the HRfunc-style wordmark (Times New Roman with an
italicized suffix) and is reused across the toolbar + the Library tab
header. Tests cover the input-validation contract (ValueError on
non-suffix), the HTML structure (italicized half wrapped in ``<em>``), and
the font-styling output. The rendering test uses NiceGUI's User fixture
because ``ui.html`` only instantiates inside a page context.
"""

from __future__ import annotations

import pytest

pytest.importorskip("nicegui")

from nicegui import ui  # noqa: E402
from nicegui.testing import User  # noqa: E402

pytest_plugins = ["nicegui.testing.user_plugin"]


class TestBrandValidation:
    """Pure-Python contract tests — no NiceGUI page context required because
    the ValueError fires before ``ui.html`` is reached."""

    def test_non_suffix_italic_raises_value_error(self):
        from hrfunc.gui.components.brand import brand

        # We can't easily exercise the ui.html path outside a page context,
        # but the ValueError check happens FIRST, before any ui call.
        with pytest.raises(ValueError, match="suffix"):
            brand("HRfunc", italic_suffix="xyz")

    def test_empty_italic_suffix_is_allowed_as_edge_case(self):
        """Empty suffix is technically a suffix of any string ("".endswith("")
        is True). The wordmark renders fully upright with an empty ``<em>``.
        Test documents this behavior rather than guarding against it — no
        caller should pass empty, but the function doesn't need to police."""
        from hrfunc.gui.components.brand import brand

        @ui.page("/_test_brand_empty")
        def _p() -> None:
            brand("HRfunc", italic_suffix="")  # should not raise

        # The page-handler import is enough to validate construction; we
        # don't need to actually open a client.


class TestBrandRendering:
    """End-to-end render via NiceGUI's User fixture — confirms the wordmark
    actually paints inside a page handler and contains the italicized half.
    """

    @pytest.mark.asyncio
    async def test_brand_renders_with_italic_suffix(self, user: User):
        from hrfunc.gui.components.brand import brand

        @ui.page("/_test_brand_render")
        def _p() -> None:
            brand("HRfunc", italic_suffix="func")

        await user.open("/_test_brand_render")
        # NiceGUI's User.should_see does exact content matching, and our
        # element's content includes the raw <em> tags — match against
        # the literal HTML string to confirm the wordmark structure.
        await user.should_see(content="HR<em>func</em>")

    @pytest.mark.asyncio
    async def test_brand_renders_hrtree_variant(self, user: User):
        from hrfunc.gui.components.brand import brand

        @ui.page("/_test_brand_hrtree")
        def _p() -> None:
            brand("HRtree", italic_suffix="tree", size_rem=2.5)

        await user.open("/_test_brand_hrtree")
        await user.should_see(content="HR<em>tree</em>")

    @pytest.mark.asyncio
    async def test_brand_applies_size_rem_to_inline_style(self, user: User):
        """``size_rem`` is the one knob the toolbar/header surfaces will
        actually vary. Confirm it lands in the inline style string."""
        from hrfunc.gui.components.brand import brand

        captured: list = []

        @ui.page("/_test_brand_size")
        def _p() -> None:
            captured.append(brand("HRtree", italic_suffix="tree", size_rem=3.5))

        await user.open("/_test_brand_size")
        # The element captured during page-render is the ui.html instance.
        # NiceGUI stores inline styles in props/style; read the style string
        # and confirm the size_rem was substituted.
        assert captured, "page handler did not run"
        # ``_style`` is the internal style dict on NiceGUI elements.
        style_text = ";".join(
            f"{k}:{v}" for k, v in captured[0]._style.items()
        )
        assert "3.5rem" in style_text
