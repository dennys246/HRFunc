"""Page handlers for the HRfunc GUI.

Each module in this subpackage registers one or more ``@ui.page`` handlers
when imported. ``app._register_pages()`` imports all of them at startup so
NiceGUI knows the full route table before the server starts.

Adding a page = create a new module here, import it in ``app._register_pages``.
"""
