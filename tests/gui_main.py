"""NiceGUI ``main_file`` for the User-fixture rendering tests.

NiceGUI's ``nicegui.testing.User`` fixture needs a Python file it can import
to bootstrap the app — the equivalent of a vanilla NiceGUI app's
``main.py``. This module exists solely to satisfy that contract: it imports
the HRfunc gui package, registers all routes, and calls ``ui.run()``
(intercepted by the User fixture, so no real server starts).

Not a pytest test file (name does not start with ``test_``) so pytest will
not try to collect it. Configured as ``main_file`` in pyproject.toml's
``[tool.pytest.ini_options]``.
"""

from nicegui import ui

from hrfunc.gui import app as gui_app

# Register the welcome page + workspace/library stubs so the User fixture
# can hit each route during rendering tests.
gui_app._register_pages()

# NiceGUI requires a ui.run() in the main_file to consider the app
# configured. The User fixture intercepts the call — no real server starts.
ui.run()
