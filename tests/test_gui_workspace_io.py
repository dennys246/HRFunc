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
    MONTAGE_SCHEMA_VERSION,
    _safe_filename_fragment,
    build_roi_entry,
    save_montage,
    save_roi_average,
    workspace_dir,
)


def _read_single_roi(out_path):
    """Helper: open a montage.json and return ``payload["rois"][0]``.

    PR #55 unified the on-disk format -- every save_roi_average call
    now writes a 1-entry montage. Tests that pre-date the schema
    change asserted the trace shape at the top level; this helper
    centralises the ``payload["rois"][0]`` indirection so each call
    site stays terse.
    """
    payload = json.loads(out_path.read_text())
    assert payload["version"] == MONTAGE_SCHEMA_VERSION
    assert "alignment" in payload
    assert len(payload["rois"]) == 1
    return payload, payload["rois"][0]


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
        # PR #55: filename prefix is "montage_" (every save writes the
        # wrapper schema, even for a single-ROI montage). The anchor
        # key still rides in the filename for discoverability.
        assert out.name.startswith("montage_")
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
        _, roi = _read_single_roi(out)
        assert roi["hrf_mean"] == [1.5, 2.5]
        assert roi["hrf_std"] == [0.5, 0.5]
        assert roi["sfreq"] == 7.8125
        assert roi["oxygenation"] is True
        assert roi["location"] == [0.01, 0.02, 0.03]
        ctx = roi["context"]
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
        _, roi = _read_single_roi(out)
        # Floats decode to plain Python floats
        assert all(isinstance(x, float) for x in roi["hrf_mean"])

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
        _, roi = _read_single_roi(out)
        assert roi["context"]["roi_member_keys"] == ["hbo:a", "hbo:m", "hbo:z"]


# ---------------------------------------------------------------------------
# save_roi_average (PR #49): shape descriptor + free-floating ROIs
# ---------------------------------------------------------------------------


class TestSaveRoiAverageWithShape:
    """PR #49 expanded ``save_roi_average`` to accept an optional shape
    descriptor and made the anchor optional. These tests pin down the
    extended schema."""

    def _anchor(self):
        return {
            "_key": "hbo:s1_d1_hbo-temp",
            "oxygenation": True,
            "location": [0.01, 0.02, 0.03],
            "context": {"task": "flanker"},
        }

    def test_box_shape_recorded_in_descriptor(self, tmp_path):
        from hrfunc.spatial.shapes import Box
        box = Box(center_mm=(10.0, 20.0, 30.0), half_extents_mm=(5.0, 5.0, 5.0))
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"hbo:a", "hbo:b"},
            hrf_mean=[1.0, 2.0],
            hrf_std=[0.1, 0.2],
            sfreq=10.0,
            shape=box,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "box"
        assert shape_desc["center_mm"] == [10.0, 20.0, 30.0]
        assert shape_desc["half_extents_mm"] == [5.0, 5.0, 5.0]

    def test_sphere_shape_recorded_in_descriptor(self, tmp_path):
        from hrfunc.spatial.shapes import Sphere
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=15.0)
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"hbo:a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=sphere,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "sphere"
        assert shape_desc["center_mm"] == [0.0, 0.0, 0.0]
        assert shape_desc["radius_mm"] == 15.0
        # The legacy roi_radius_m key still appears for back-compat audit scripts.
        assert roi["context"]["roi_radius_m"] == 0.015

    def test_free_floating_no_anchor_uses_shape_centre_as_location(self, tmp_path):
        """When the user has not clicked an anchor, the shape centre
        stands in as the saved ``location`` -- converted from mm back
        to meters so the saved JSON round-trips into ``hrfunc.tree``
        without unit fixups."""
        from hrfunc.spatial.shapes import Box
        box = Box(center_mm=(15.0, 25.0, 35.0), half_extents_mm=(5.0, 5.0, 5.0))
        out = save_roi_average(
            roi_keys={"hbo:a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=box,
            oxygenation_filter=True,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        # Centre 15 mm -> 0.015 m (matching v1.2 meter convention).
        assert roi["location"] == [0.015, 0.025, 0.035]
        # Oxygenation falls back to the explicit filter when no anchor.
        assert roi["oxygenation"] is True

    def test_free_floating_filename_prefix(self, tmp_path):
        """Free-floating saves should still have a recognisable filename
        prefix even without an anchor key to derive one from."""
        from hrfunc.spatial.shapes import Box
        out = save_roi_average(
            roi_keys=set(),
            hrf_mean=[0.0],
            hrf_std=[0.0],
            sfreq=1.0,
            shape=Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(1.0, 1.0, 1.0)),
            workspace=tmp_path,
        )
        assert "freefloat_box" in out.name

    def test_anchor_key_is_none_in_context_when_no_anchor(self, tmp_path):
        from hrfunc.spatial.shapes import Sphere
        sphere = Sphere(center_mm=(0.0, 0.0, 0.0), radius_mm=10.0)
        out = save_roi_average(
            roi_keys={"hbo:a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=sphere,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        assert roi["context"]["roi_anchor_key"] is None

    def test_oxygenation_filter_none_means_mixed(self, tmp_path):
        """When neither anchor nor oxygenation_filter is set, the saved
        ``oxygenation`` is None -- signalling a mixed-haemoglobin ROI."""
        from hrfunc.spatial.shapes import Box
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(5.0, 5.0, 5.0))
        out = save_roi_average(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=box,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        assert roi["oxygenation"] is None

    def test_legacy_radius_m_only_call_still_works(self, tmp_path):
        """v1.2 call sites that pass ``radius_m`` without ``shape`` keep
        working unchanged; the saved JSON gets a synthetic sphere
        descriptor + the ``roi_radius_m`` legacy key."""
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            radius_m=0.02,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        assert roi["context"]["roi_radius_m"] == 0.02
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "sphere"
        assert shape_desc["radius_m"] == 0.02


class TestSaveRoiAverageBoxOrientation:
    """PR #52 added an optional ``orientation_mm`` field to the box
    shape descriptor. For axis-aligned boxes the field is omitted so
    pre-PR-#52 readers don't see schema noise; for rotated boxes the
    rotation matrix is serialised as a nested list."""

    def _anchor(self):
        return {
            "_key": "hbo:k",
            "oxygenation": True,
            "location": [0.01, 0.02, 0.03],
            "context": {},
        }

    def test_axis_aligned_box_omits_orientation_field(self, tmp_path):
        from hrfunc.spatial.shapes import Box
        box = Box(center_mm=(0.0, 0.0, 0.0), half_extents_mm=(5.0, 5.0, 5.0))
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=box,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "box"
        # Pre-PR-#52 readers parse this descriptor unchanged.
        assert "orientation_mm" not in shape_desc

    def test_rotated_box_includes_orientation_matrix(self, tmp_path):
        import numpy as np
        from hrfunc.spatial.shapes import Box
        # 90 degrees about z.
        R = np.array(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        box = Box(
            center_mm=(0.0, 0.0, 0.0),
            half_extents_mm=(5.0, 5.0, 5.0),
            orientation_mm=R,
        )
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=box,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "box"
        assert "orientation_mm" in shape_desc
        # The matrix round-trips through json as a list-of-lists.
        recovered = np.asarray(shape_desc["orientation_mm"])
        np.testing.assert_allclose(recovered, R)

    def test_atlas_region_descriptor_records_atlas_and_region(self, tmp_path):
        """PR #53: AtlasRegion shapes serialise their atlas name +
        region label so readers can reconstruct the same ROI."""
        import numpy as np
        from hrfunc.spatial.atlas import Atlas
        from hrfunc.spatial.shapes import AtlasRegion

        volume = np.zeros((3, 3, 3), dtype=np.int64)
        volume[:, 1, :] = 1
        affine = np.diag([10.0, 10.0, 10.0, 1.0])
        atlas = Atlas(
            name="synthetic-test",
            volume=volume,
            affine=affine,
            labels=["Background", "Region_A"],
            background_label=0,
        )
        region = AtlasRegion(atlas, "Region_A")
        out = save_roi_average(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=region,
            oxygenation_filter=True,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        assert shape_desc["type"] == "atlas_region"
        assert shape_desc["atlas"] == "synthetic-test"
        assert shape_desc["region_name"] == "Region_A"

    def test_atlas_region_location_uses_region_centroid(self, tmp_path):
        """Free-floating atlas-region save (no anchor) records the
        region's voxel-centroid in MNE meters as ``location``."""
        import numpy as np
        from hrfunc.spatial.atlas import Atlas
        from hrfunc.spatial.shapes import AtlasRegion

        # Build a tiny atlas where Region_A occupies the y=1 slab so
        # its voxel centroid is at (1, 1, 1) -> MNI (10, 10, 10) mm
        # -> 0.01 m on each axis.
        volume = np.zeros((3, 3, 3), dtype=np.int64)
        volume[:, 1, :] = 1
        affine = np.diag([10.0, 10.0, 10.0, 1.0])
        atlas = Atlas(
            name="synth-centroid",
            volume=volume,
            affine=affine,
            labels=["Background", "Region_A"],
            background_label=0,
        )
        region = AtlasRegion(atlas, "Region_A")
        out = save_roi_average(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=region,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        # Region_A occupies voxels (i, 1, k) for i, k in [0, 3) -- the
        # centroid is voxel (1, 1, 1) -> MNI (10, 10, 10) mm = 0.01 m.
        loc = roi["location"]
        assert loc == pytest.approx([0.01, 0.01, 0.01], abs=1e-9)

    def test_atlas_region_filename_prefix(self, tmp_path):
        """Free-floating atlas-region saves get a recognisable prefix."""
        import numpy as np
        from hrfunc.spatial.atlas import Atlas
        from hrfunc.spatial.shapes import AtlasRegion

        volume = np.zeros((3, 3, 3), dtype=np.int64)
        volume[:, 1, :] = 1
        atlas = Atlas(
            name="synth",
            volume=volume,
            affine=np.diag([10.0, 10.0, 10.0, 1.0]),
            labels=["Background", "Region_A"],
        )
        out = save_roi_average(
            roi_keys=set(),
            hrf_mean=[0.0],
            hrf_std=[0.0],
            sfreq=1.0,
            shape=AtlasRegion(atlas, "Region_A"),
            workspace=tmp_path,
        )
        assert "freefloat_atlas_region" in out.name

    def test_orientation_matrix_round_trips_to_box(self, tmp_path):
        """End-to-end: save a rotated box, reload the descriptor, build
        a Box from it, and confirm membership decisions match the
        original."""
        import numpy as np
        from hrfunc.spatial.shapes import Box

        R = np.array(
            [[np.cos(0.4), -np.sin(0.4), 0.0],
             [np.sin(0.4), np.cos(0.4), 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        original = Box(
            center_mm=(2.0, 3.0, 4.0),
            half_extents_mm=(5.0, 7.0, 9.0),
            orientation_mm=R,
        )
        out = save_roi_average(
            anchor=self._anchor(),
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            shape=original,
            workspace=tmp_path,
        )
        _, roi = _read_single_roi(out)
        shape_desc = roi["context"]["roi_shape"]
        rebuilt = Box(
            center_mm=tuple(shape_desc["center_mm"]),
            half_extents_mm=tuple(shape_desc["half_extents_mm"]),
            orientation_mm=np.asarray(shape_desc["orientation_mm"]),
        )
        rng = np.random.default_rng(seed=13)
        points = rng.uniform(-20, 20, size=(100, 3))
        np.testing.assert_array_equal(
            original.contains_batch(points),
            rebuilt.contains_batch(points),
        )


# ---------------------------------------------------------------------------
# PR #55: save_montage wrapper schema + multi-ROI lists
# ---------------------------------------------------------------------------


class TestSaveMontage:
    """The new save_montage writer is the canonical on-disk path; the
    legacy save_roi_average delegates to it. These tests pin down the
    wrapper schema, alignment block, filename convention, and
    multi-ROI behaviour."""

    def _entry(self, name="ROI 1"):
        return build_roi_entry(
            roi_keys={"hbo:a"},
            hrf_mean=[1.0, 2.0],
            hrf_std=[0.1, 0.2],
            sfreq=10.0,
            anchor={
                "_key": "hbo:s1_d1_hbo-temp",
                "oxygenation": True,
                "location": [0.01, 0.02, 0.03],
                "context": {"task": "flanker"},
            },
            radius_m=0.02,
            name=name,
        )

    def test_wrapper_keys_and_version(self, tmp_path):
        out = save_montage(
            rois=[self._entry()],
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["version"] == MONTAGE_SCHEMA_VERSION
        assert set(payload) == {"version", "alignment", "rois"}

    def test_alignment_block_defaults(self, tmp_path):
        """Without explicit alignment, the wrapper still records the
        block -- offset_mm zeros, affine null. Readers don't have to
        special-case the absence."""
        out = save_montage(
            rois=[self._entry()],
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["alignment"]["offset_mm"] == [0.0, 0.0, 0.0]
        assert payload["alignment"]["affine"] is None

    def test_alignment_offset_round_trips(self, tmp_path):
        out = save_montage(
            rois=[self._entry()],
            alignment_offset_mm=(1.5, -2.5, 3.0),
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["alignment"]["offset_mm"] == [1.5, -2.5, 3.0]

    def test_alignment_affine_round_trips(self, tmp_path):
        import numpy as np
        affine = np.array(
            [[1.0, 0.0, 0.0, 5.0],
             [0.0, 1.0, 0.0, -3.0],
             [0.0, 0.0, 1.0, 1.5],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        out = save_montage(
            rois=[self._entry()],
            alignment_affine=affine,
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        recovered = np.asarray(payload["alignment"]["affine"])
        np.testing.assert_allclose(recovered, affine)

    def test_rois_list_preserves_order_and_names(self, tmp_path):
        out = save_montage(
            rois=[self._entry("Pole"), self._entry("OFC"), self._entry("PFC")],
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert [r["name"] for r in payload["rois"]] == ["Pole", "OFC", "PFC"]

    def test_filename_single_roi_has_no_plus_suffix(self, tmp_path):
        """Single-ROI montages keep the descriptive fragment-only
        filename so they read like the pre-PR-#55 single-file saves."""
        out = save_montage(
            rois=[self._entry()],
            workspace=tmp_path,
        )
        assert out.name.startswith("montage_hbo_s1_d1_hbo-temp_")
        assert "_plus" not in out.name

    def test_filename_multi_roi_has_plus_suffix(self, tmp_path):
        """Multi-ROI montages tag the filename with how many extra
        ROIs ride alongside the first, so a directory listing surfaces
        the size without having to open the file."""
        out = save_montage(
            rois=[self._entry(), self._entry("ROI 2"), self._entry("ROI 3")],
            workspace=tmp_path,
        )
        assert "_plus2_" in out.name

    def test_empty_rois_list_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            save_montage(rois=[], workspace=tmp_path)

    def test_save_roi_average_writes_single_entry_montage(self, tmp_path):
        """The legacy single-save shim writes the same wrapper schema
        with a 1-entry rois list -- one writer, one reader."""
        out = save_roi_average(
            anchor={
                "_key": "hbo:k",
                "oxygenation": True,
                "location": [0.0, 0.0, 0.0],
                "context": {},
            },
            roi_keys={"hbo:a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            radius_m=0.02,
            workspace=tmp_path,
        )
        payload = json.loads(out.read_text())
        assert payload["version"] == MONTAGE_SCHEMA_VERSION
        assert len(payload["rois"]) == 1
        assert payload["rois"][0]["hrf_mean"] == [1.0]


class TestBuildRoiEntry:
    """build_roi_entry is the pure helper that turns ROI inputs into
    one block of the rois list. Same content as the legacy single-save
    payload, with an optional ``name`` field for the multi-ROI list
    display label."""

    def test_name_omitted_when_not_provided(self):
        entry = build_roi_entry(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
        )
        assert "name" not in entry

    def test_name_included_when_provided(self):
        entry = build_roi_entry(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
            name="Visual Cortex",
        )
        assert entry["name"] == "Visual Cortex"

    def test_pure_no_io(self, tmp_path):
        """Build the entry without touching the filesystem so callers
        can build a montage in memory before deciding where to save."""
        # No workspace argument -- this must not create files anywhere.
        entry = build_roi_entry(
            roi_keys={"a"},
            hrf_mean=[1.0],
            hrf_std=[0.1],
            sfreq=10.0,
        )
        assert isinstance(entry, dict)
        # tmp_path should remain empty since build_roi_entry doesn't
        # write -- we only pass it to confirm no surprise mkdir/write.
        assert list(tmp_path.iterdir()) == []
