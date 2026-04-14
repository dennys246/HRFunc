"""
Targeted unit tests for refactor/circular-imports-phase2.

Verifies that the circular import hrfunc.py ↔ hrtree.py has been broken
by moving shared helpers into hrfunc._utils, and that all existing
functionality using standardize_name / _is_oxygenated / _LIB_DIR still
resolves correctly through the new import path.
"""

import os
import pytest


# ---------------------------------------------------------------------------
# No circular imports
# ---------------------------------------------------------------------------

class TestNoCircularImports:
    def test_hrfunc_does_not_self_import(self):
        """hrfunc.py must not contain `import hrfunc` anymore."""
        import hrfunc.hrfunc as hf
        source_path = hf.__file__
        with open(source_path) as f:
            source = f.read()
        # Check for the specific self-import pattern, not references in strings/comments
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            assert stripped != 'import hrfunc', (
                "hrfunc.py still contains `import hrfunc` self-import"
            )

    def test_hrtree_does_not_import_hrfunc(self):
        """hrtree.py must not import from hrfunc anymore."""
        import hrfunc.hrtree as ht
        source_path = ht.__file__
        with open(source_path) as f:
            source = f.read()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
            assert 'from . import hrhash, hrfunc' not in stripped, (
                "hrtree.py still contains circular `from . import hrhash, hrfunc`"
            )
            assert 'from .hrfunc import' not in stripped, (
                "hrtree.py still has a direct hrfunc import"
            )

    def test_utils_module_is_importable_directly(self):
        """_utils.py exists and is directly importable."""
        from hrfunc import _utils
        assert hasattr(_utils, 'standardize_name')
        assert hasattr(_utils, '_is_oxygenated')
        assert hasattr(_utils, '_LIB_DIR')

    def test_utils_does_not_import_hrfunc_or_hrtree(self):
        """_utils.py must stay dependency-free of hrfunc.py and hrtree.py."""
        from hrfunc import _utils
        with open(_utils.__file__) as f:
            source = f.read()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('"""'):
                continue
            assert 'from .hrfunc' not in stripped
            assert 'from .hrtree' not in stripped
            assert 'import hrfunc.hrfunc' not in stripped
            assert 'import hrfunc.hrtree' not in stripped


# ---------------------------------------------------------------------------
# _LIB_DIR resolves to the package directory
# ---------------------------------------------------------------------------

class TestLibDirConstant:
    def test_lib_dir_points_to_package(self):
        """_LIB_DIR must resolve to the src/hrfunc directory containing hrfs/."""
        from hrfunc._utils import _LIB_DIR
        assert os.path.isdir(_LIB_DIR)
        # The bundled library JSONs must be reachable from _LIB_DIR
        assert os.path.isfile(os.path.join(_LIB_DIR, 'hrfs', 'hbo_hrfs.json'))
        assert os.path.isfile(os.path.join(_LIB_DIR, 'hrfs', 'hbr_hrfs.json'))

    def test_montage_lib_dir_matches_utils(self):
        """montage.__init__ sets self.lib_dir to _LIB_DIR."""
        from hrfunc.hrfunc import montage
        from hrfunc._utils import _LIB_DIR
        m = montage()
        assert m.lib_dir == _LIB_DIR

    def test_tree_lib_dir_matches_utils(self):
        """tree.__init__ sets self.lib_dir to _LIB_DIR."""
        from hrfunc.hrtree import tree
        from hrfunc._utils import _LIB_DIR
        t = tree()
        assert t.lib_dir == _LIB_DIR


# ---------------------------------------------------------------------------
# standardize_name + _is_oxygenated still resolve through both files
# ---------------------------------------------------------------------------

class TestHelperResolution:
    def test_hrfunc_uses_utils_helpers(self):
        """hrfunc.py imports standardize_name and _is_oxygenated from _utils."""
        from hrfunc import _utils
        import hrfunc.hrfunc as hf
        # After the refactor, hrfunc.hrfunc.standardize_name should be the
        # same function object as hrfunc._utils.standardize_name
        assert hf.standardize_name is _utils.standardize_name
        assert hf._is_oxygenated is _utils._is_oxygenated

    def test_hrtree_uses_utils_helpers(self):
        """hrtree.py imports standardize_name and _is_oxygenated from _utils."""
        from hrfunc import _utils
        import hrfunc.hrtree as ht
        assert ht.standardize_name is _utils.standardize_name
        assert ht._is_oxygenated is _utils._is_oxygenated

    def test_standardize_name_still_works(self):
        """Regression: standardize_name behavior unchanged after move."""
        from hrfunc._utils import standardize_name
        # A simple hbo channel should round-trip through normalization
        result = standardize_name('S1-D1 hbo')
        assert result.endswith('hbo')

    def test_is_oxygenated_still_works(self):
        """Regression: _is_oxygenated behavior unchanged after move."""
        from hrfunc._utils import _is_oxygenated
        assert _is_oxygenated('s1_d1_hbo') is True
        assert _is_oxygenated('s1_d1_hbr') is False
        with pytest.raises(ValueError):
            _is_oxygenated('s1_d1_xyz')


# ---------------------------------------------------------------------------
# load_montage still works with the renamed local variable
# ---------------------------------------------------------------------------

class TestLoadMontageLocalRename:
    def test_load_montage_uses_bundled_library_file(self, tmp_path):
        """load_montage must still successfully construct a montage via
        the module-level `montage` class even though its local variable
        was renamed to `_montage` to avoid UnboundLocalError."""
        from hrfunc.hrfunc import load_montage
        from hrfunc._utils import _LIB_DIR
        # Use the bundled hbo_hrfs.json as a known-good input
        bundled = os.path.join(_LIB_DIR, 'hrfs', 'hbo_hrfs.json')
        m = load_montage(bundled)
        assert m is not None
        assert m.configured is True