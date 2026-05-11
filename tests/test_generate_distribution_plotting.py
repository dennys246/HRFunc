"""
Targeted unit tests for montage.generate_distribution(plot_dir=...) plotting.

Bug history: HRF.plot() was refactored from `plot(self, plot_dir)`
(which built the per-channel filename internally as
`{plot_dir}/{ch_name}_hrf_estimate.png`) to `plot(self, plot_path=None)`
(treats the arg as the full file path). montage.generate_distribution
was never updated and continued to pass the directory to plot(), so
`plt.savefig("plots/")` silently failed or raised IsADirectoryError
and no per-channel plots were produced.

The fix: generate_distribution now builds the per-channel filename
inside plot_dir, ensures plot_dir exists, and passes show=False so
headless cluster runs don't block on plt.show().

Tests use the Agg backend so they run on any environment.
"""

import os
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest


def _populated_montage():
    """Build a montage with a couple of channels that have estimates ready
    for generate_distribution to consume."""
    from hrfunc.hrfunc import montage
    from hrfunc.hrtree import HRF

    m = montage()
    m.sfreq = 7.81

    # Insert two channels with synthetic estimates
    hbo = HRF('doi', 's1_d1_hbo', 30.0, 7.81, np.zeros(234),
              location=[0.0, 0.0, 0.0], estimates=[], locations=[])
    hbo.estimates = [list(np.linspace(0, 1, 234))]
    hbo.locations = [[0.0, 0.0, 0.0]]
    hbr = HRF('doi', 's1_d1_hbr', 30.0, 7.81, np.zeros(234),
              location=[0.01, 0.0, 0.0], estimates=[], locations=[])
    hbr.estimates = [list(-np.linspace(0, 1, 234))]
    hbr.locations = [[0.01, 0.0, 0.0]]

    m.channels['s1_d1_hbo'] = m.hbo_tree.insert(hbo)
    m.channels['s1_d1_hbr'] = m.hbr_tree.insert(hbr)
    m.hbo_channels = ['s1_d1_hbo']
    m.hbr_channels = ['s1_d1_hbr']

    return m


class TestGenerateDistributionPlotting:
    def test_plot_dir_creates_per_channel_files(self, tmp_path):
        m = _populated_montage()
        plot_dir = str(tmp_path / "plots")

        m.generate_distribution(plot_dir=plot_dir)

        # Both channels should have produced a PNG
        assert os.path.isdir(plot_dir)
        files = os.listdir(plot_dir)
        assert any('s1_d1_hbo' in f and f.endswith('.png') for f in files), files
        assert any('s1_d1_hbr' in f and f.endswith('.png') for f in files), files

    def test_plot_dir_created_if_missing(self, tmp_path):
        m = _populated_montage()
        # Use a nested dir that doesn't exist yet
        plot_dir = str(tmp_path / "deep" / "nested" / "plots")
        assert not os.path.exists(plot_dir)

        m.generate_distribution(plot_dir=plot_dir)

        assert os.path.isdir(plot_dir)
        assert any(f.endswith('.png') for f in os.listdir(plot_dir))

    def test_no_plot_dir_skips_plotting(self, tmp_path, monkeypatch):
        """When plot_dir is None / falsy, no plots should be produced and
        no directory should be created."""
        m = _populated_montage()
        # Run inside tmp_path so any accidental relative writes are caught
        monkeypatch.chdir(tmp_path)

        m.generate_distribution()  # plot_dir defaults to None

        # No stray files written in cwd
        assert os.listdir(tmp_path) == []

    def test_plot_call_does_not_block_on_show(self, tmp_path):
        """Regression guard for the show=True default: generate_distribution
        must pass show=False so headless / cluster runs don't try to
        open an interactive window."""
        import inspect
        from hrfunc.hrfunc import montage
        src = inspect.getsource(montage.generate_distribution)
        assert 'show=False' in src
