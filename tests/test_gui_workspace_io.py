"""Unit tests for ``hrfunc.gui.workspace_io``.

Covers:
- ``workspace_dir`` default location, env-var override, lazy mkdir.
- ``save_roi_average`` schema, ROI provenance, filename sanitization.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest


from hrfunc.gui.workspace_io import (
    _safe_filename_fragment,
    save_roi_average,
    workspace_dir,
)


# ---------------------------------------------------------------------------
# workspace_dir
# ---------------------------------------------------------------------------


class TestWorkspaceDir:
    def test_env_override_used_when_set(self, tmp_path, monkeypatch):
        target = tmp_path / "custom_workspace"
        monkeypatch.setenv("HRFUNC_WORKSPACE", str(target))
        result = workspace_dir()
        assert result == target
        assert target.exists()

    def test_env_override_expands_user(self, tmp_path, monkeypatch):
        """Tilde in HRFUNC_WORKSPACE should expand to the user's home."""
        monkeypatch.setenv("HRFUNC_WORKSPACE", "~/test_hrfunc_ws_pytest")
        result = workspace_dir()
        assert str(result).startswith(str(Path.home()))
        # Clean up
        if result.exists():
            result.rmdir()

    def test_default_uses_home_when_no_env(self, monkeypatch, tmp_path):
        # Stub Path.home() so the test doesn't pollute the real home dir.
        monkeypatch.delenv("HRFUNC_WORKSPACE", raising=False)
        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
        result = workspace_dir()
        assert result == fake_home / "hrfunc_workspace"
        assert result.exists()

    def test_idempotent_create(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HRFUNC_WORKSPACE", str(tmp_path / "idempotent"))
        first = workspace_dir()
        second = workspace_dir()
        assert first == second
        assert first.exists()


# ---------------------------------------------------------------------------
# filename sanitization
# ---------------------------------------------------------------------------


class TestSafeFilenameFragment:
    def test_alphanumeric_passes_through(self):
        assert _safe_filename_fragment("s1_d1_hbo") == "s1_d1_hbo"

    def test_replaces_slashes_colons_spaces(self):
        out = _safe_filename_fragment("hbo:s1_d1_hbo-temp")
        assert ":" not in out
        # Underscore between the prefix and rest after sanitizing the colon
        assert "hbo" in out and "s1_d1_hbo-temp" in out

    def test_collapses_runs_of_underscores(self):
        out = _safe_filename_fragment("a???b")
        assert "__" not in out  # collapsed

    def test_strips_leading_trailing_dots_underscores(self):
        assert _safe_filename_fragment(".._hidden_._") == "hidden"

    def test_empty_falls_back_to_untitled(self):
        assert _safe_filename_fragment("") == "untitled"
        assert _safe_filename_fragment("___") == "untitled"


# ---------------------------------------------------------------------------
# save_roi_average
# ---------------------------------------------------------------------------


class TestSaveRoiAverage:
    def _anchor(self):
        return {
            "_key": "hbo:s1_d1_hbo-temp",
            "oxygenation": True,
            "location": [0.01, 0.02, 0.03],
            "context": {"task": "flanker", "doi": "doi/A"},
        }

    def test_writes_json_in_workspace(self, tmp_path):
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"hbo:s1_d1_hbo-temp", "hbo:s2_d1_hbo-temp"},
            hrf_mean=mean,
            hrf_std=std,
            sfreq=7.8125,
            radius_m=0.02,
            workspace=tmp_path,
        )
        assert out.parent == tmp_path
        assert out.suffix == ".json"
        assert out.exists()

    def test_filename_includes_anchor_key_and_timestamp(self, tmp_path):
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys=set(),
            hrf_mean=[0.0],
            hrf_std=[0.0],
            sfreq=1.0,
            radius_m=0.0,
            workspace=tmp_path,
        )
        # Sanitized anchor key + timestamp pattern
        assert out.name.startswith("roi_")
        assert "hbo_s1_d1_hbo-temp" in out.name
        # ISO basic format includes a 'T' separator
        assert "T" in out.name

    def test_payload_schema(self, tmp_path):
        mean = np.array([1.5, 2.5])
        std = np.array([0.5, 0.5])
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"hbo:s1_d1_hbo-temp", "hbo:s2_d1_hbo-temp"},
            hrf_mean=mean,
            hrf_std=std,
            sfreq=7.8125,
            radius_m=0.02,
            library_filter={"task": "flanker"},
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["hrf_mean"] == [1.5, 2.5]
        assert payload["hrf_std"] == [0.5, 0.5]
        assert payload["sfreq"] == 7.8125
        assert payload["oxygenation"] is True
        assert payload["location"] == [0.01, 0.02, 0.03]
        ctx = payload["context"]
        # Anchor's original context preserved
        assert ctx["task"] == "flanker"
        assert ctx["doi"] == "doi/A"
        # ROI provenance attached
        assert ctx["roi_average"] is True
        assert ctx["roi_anchor_key"] == "hbo:s1_d1_hbo-temp"
        assert ctx["roi_radius_m"] == 0.02
        assert ctx["roi_member_keys"] == [
            "hbo:s1_d1_hbo-temp", "hbo:s2_d1_hbo-temp",
        ]
        assert ctx["roi_library_filter"] == {"task": "flanker"}
        assert "saved_at" in ctx

    def test_numpy_arrays_serialize_cleanly(self, tmp_path):
        """numpy arrays must serialize through json.dump without TypeError."""
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"a"},
            hrf_mean=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            hrf_std=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sfreq=10.0,
            radius_m=0.02,
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        # Floats decode to plain Python floats
        assert all(isinstance(x, float) for x in payload["hrf_mean"])

    def test_member_keys_sorted_for_determinism(self, tmp_path):
        """Set iteration order is unpredictable; ROI member keys in the
        output should always sort for reproducibility."""
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"hbo:z", "hbo:a", "hbo:m"},
            hrf_mean=[0.0],
            hrf_std=[0.0],
            sfreq=1.0,
            radius_m=0.0,
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["context"]["roi_member_keys"] == ["hbo:a", "hbo:m", "hbo:z"]
