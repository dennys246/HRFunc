"""
Targeted unit tests for fix/estimate-activity-threading (ND-003 + ND-004).

ND-003: `success` variable set inside the nested `deconvolution` closure was
        not visible to the outer channel loop — the `if success == False`
        check always read None, so channels were never actually dropped on
        timeout. Fixed by adding `nonlocal success` and declaring `success`
        at estimate_activity scope before the closure is defined.

ND-004: `timeout=1500` default was 25 minutes, not 1.5 seconds.
        Fixed to `timeout=30` (see Q1 decision in open questions memory).

These tests exercise the scoping and signature without needing real fNIRS
data. The full estimate_activity path still requires an MNE Raw object,
which is covered by integration tests (currently disabled).
"""

import inspect
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# ND-004: timeout default is 30s
# ---------------------------------------------------------------------------

class TestTimeoutDefault:
    def test_estimate_activity_timeout_default_is_30(self):
        """The timeout parameter default must be 30 (seconds), not 1500."""
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        assert sig.parameters['timeout'].default == 30

    def test_timeout_is_still_a_parameter(self):
        """timeout must remain user-configurable, not hardcoded."""
        from hrfunc.hrfunc import montage
        sig = inspect.signature(montage.estimate_activity)
        assert 'timeout' in sig.parameters


# ---------------------------------------------------------------------------
# ND-003: success variable propagates from closure to outer scope
# ---------------------------------------------------------------------------

class TestSuccessNonlocal:
    def test_deconvolution_closure_declares_nonlocal_success(self):
        """The fix for ND-003 is a `nonlocal success` in the closure.
        Without it, success=True/False inside the closure is a no-op for
        the outer drop-channel check."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'nonlocal success' in source, (
            "deconvolution closure must declare `nonlocal success` — "
            "without it, ND-003 re-regresses and channels are never dropped"
        )

    def test_success_initialized_before_closure_definition(self):
        """For `nonlocal` to bind cleanly, `success = None` must appear in
        estimate_activity's body before `def deconvolution`."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        # Find the first occurrence of each
        success_init_idx = source.find('success = None')
        def_deconv_idx = source.find('def deconvolution')
        assert success_init_idx != -1, "success must be initialized to None somewhere"
        assert def_deconv_idx != -1, "deconvolution closure must exist"
        assert success_init_idx < def_deconv_idx, (
            "`success = None` must appear before `def deconvolution` so that "
            "`nonlocal success` resolves to estimate_activity's scope"
        )

    def test_drop_channel_check_uses_is_false(self):
        """After the fix, the drop check reads `success is False`, not `== False`.
        This is a PEP 8 style fix for clarity, not a correctness fix."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'success is False' in source
        assert 'success == False' not in source


# ---------------------------------------------------------------------------
# Functional scope smoke test — closure + nonlocal mechanics
# ---------------------------------------------------------------------------

class TestNonlocalMechanics:
    """Verifies the `nonlocal` pattern actually propagates state correctly,
    independent of the full estimate_activity call path. This is a sanity
    check on the Python semantics we're relying on."""

    def test_nonlocal_write_visible_to_outer(self):
        """Demonstrates the exact pattern used in estimate_activity:
        outer function defines success, nested closure writes to it
        via `nonlocal`, outer function reads the updated value."""
        def outer():
            success = None
            def inner():
                nonlocal success
                success = True
            inner()
            return success

        assert outer() is True

    def test_nonlocal_without_declaration_does_not_propagate(self):
        """Without `nonlocal`, the inner assignment creates a closure-local.
        This is exactly the bug ND-003 described."""
        def outer():
            success = None
            def inner():
                success = True  # no nonlocal — local to inner
            inner()
            return success

        assert outer() is None  # bug reproduction


# ---------------------------------------------------------------------------
# Regression: estimate_activity signature and return contract
# ---------------------------------------------------------------------------

class TestEstimateActivityContract:
    def test_returns_nirx_obj_on_success_path(self):
        """phase 1c fix 1.16: estimate_activity must return nirx_obj (not None)
        when it completes normally. Checked via source inspection since a
        real call needs an MNE Raw object."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'return nirx_obj' in source

    def test_returns_none_when_all_channels_bad(self):
        """When preprocess_fnirs returns None (all channels bad), estimate_activity
        must return None early and not attempt deconvolution."""
        from hrfunc.hrfunc import montage
        source = inspect.getsource(montage.estimate_activity)
        assert 'return None' in source