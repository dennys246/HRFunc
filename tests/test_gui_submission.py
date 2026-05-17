"""Unit tests for ``hrfunc.gui.submission``.

Covers the pure helpers (metadata dataclass, wire serialisation,
required-field check, URL resolution) and the HTTP client (with
``requests.post`` mocked so we never hit the network). The NiceGUI
render path isn't exercised here -- it's covered indirectly by the
Export / HRFs panel tests that mount the surrounding panels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("nicegui")

from hrfunc.gui.submission import (  # noqa: E402
    DEFAULT_UPLOAD_URL,
    SubmissionMetadata,
    SubmissionResult,
    submit_payload,
    upload_url,
)


# ---------------------------------------------------------------------------
# SubmissionMetadata
# ---------------------------------------------------------------------------


def _filled_metadata(**overrides) -> SubmissionMetadata:
    """Build a fully-filled metadata object so individual tests can
    override single fields without re-typing 18 attributes."""
    m = SubmissionMetadata(
        name="Pat", email="pat@example.com", phone="555-0100",
        study="Flanker Study", area_codes="A1", doi="doi/x",
        dataset_ownership="yes",
        hrfunc_standard="yes",
        dataset_subset="no",
        task="flanker", conditions="cong,incong", stimuli="arrows",
        medium="monitor", intensity="1.0", protocol="default",
        age="(18,65)", demographics="all", health_status="untested",
    )
    for k, v in overrides.items():
        setattr(m, k, v)
    return m


class TestSubmissionMetadataDefaults:
    def test_all_fields_default_to_empty_string(self):
        m = SubmissionMetadata()
        # Spot-check: every Python attribute used by the form starts
        # blank so the panel renders a clean form on first paint.
        assert m.name == ""
        assert m.email == ""
        assert m.dataset_ownership == ""
        assert m.health_status == ""
        assert m.comment == ""


class TestToFormDict:
    """Empty fields are excluded; the dashes-in-wire-names ones use
    the dashed form so the backend's ``request.form.to_dict()``
    surface matches the web form byte-for-byte."""

    def test_empty_metadata_yields_empty_dict(self):
        m = SubmissionMetadata()
        assert m.to_form_dict() == {}

    def test_filled_fields_appear_with_underscore_names(self):
        m = SubmissionMetadata(name="Pat", email="pat@x.com")
        form = m.to_form_dict()
        assert form["name"] == "Pat"
        assert form["email"] == "pat@x.com"

    def test_dashed_wire_names_for_area_codes_and_health_status(self):
        """``area_codes`` and ``health_status`` use Python underscores
        in the dataclass but dashes on the wire (because the web form's
        HTML uses ``name="area-codes"`` / ``name="health-status"``).
        Backend reads both; we mirror the web exactly."""
        m = SubmissionMetadata(area_codes="A1", health_status="healthy")
        form = m.to_form_dict()
        assert "area-codes" in form
        assert "health-status" in form
        assert "area_codes" not in form
        assert "health_status" not in form

    def test_internal_attributes_excluded(self):
        """Private/internal attributes (``_WIRE_OVERRIDES``) must not
        show up in the wire dict."""
        m = SubmissionMetadata(name="Pat")
        form = m.to_form_dict()
        assert "_WIRE_OVERRIDES" not in form
        assert all(not k.startswith("_") for k in form)


class TestMissingRequired:
    """The check mirrors the web form's ``required`` attribute, with
    the same conditional rules (permission only when not the owner,
    extension only when not standard library)."""

    def test_empty_metadata_lists_every_required_field(self):
        m = SubmissionMetadata()
        missing = m.missing_required()
        # 18 unconditionally-required fields on a fresh metadata.
        assert len(missing) == 18

    def test_filled_metadata_owns_dataset_returns_empty(self):
        m = _filled_metadata()
        assert m.missing_required() == []

    def test_owner_no_requires_permission(self):
        m = _filled_metadata(dataset_ownership="no")
        assert "Dataset Permission" in m.missing_required()

    def test_permission_yes_requires_owner_and_contact(self):
        m = _filled_metadata(
            dataset_ownership="no", dataset_permission="yes",
        )
        missing = m.missing_required()
        assert "Dataset Owner Name" in missing
        assert "Dataset Owner Email" in missing

    def test_permission_no_does_not_require_owner(self):
        """No permission -> can't submit; owner / contact aren't
        required because the user can't proceed anyway."""
        m = _filled_metadata(
            dataset_ownership="no", dataset_permission="no",
        )
        missing = m.missing_required()
        assert "Dataset Owner Name" not in missing
        assert "Dataset Owner Email" not in missing

    def test_hrfunc_standard_no_requires_extension(self):
        m = _filled_metadata(hrfunc_standard="no")
        assert "HRfunc Modifications" in m.missing_required()

    def test_hrfunc_standard_yes_does_not_require_extension(self):
        m = _filled_metadata(hrfunc_standard="yes")
        assert "HRfunc Modifications" not in m.missing_required()


# ---------------------------------------------------------------------------
# upload_url
# ---------------------------------------------------------------------------


class TestUploadUrl:
    def test_default_is_production(self, monkeypatch):
        monkeypatch.delenv("HRFUNC_UPLOAD_URL", raising=False)
        assert upload_url() == DEFAULT_UPLOAD_URL

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv(
            "HRFUNC_UPLOAD_URL",
            "http://staging.example.com/upload_json",
        )
        assert upload_url() == "http://staging.example.com/upload_json"


# ---------------------------------------------------------------------------
# submit_payload
# ---------------------------------------------------------------------------


def _write_valid_json(tmp_path: Path) -> Path:
    """Write a minimal valid JSON payload for happy-path tests."""
    p = tmp_path / "montage.json"
    p.write_text(json.dumps({"rois": [{"hrf_mean": [1.0, 2.0]}]}))
    return p


class TestSubmitPayloadPreflight:
    """Pre-flight checks run BEFORE any HTTP call so transient
    failures (bad path, malformed JSON) don't waste a network
    round-trip and so the GUI gets a clean error to surface."""

    def test_missing_file_returns_failure(self, tmp_path):
        result = submit_payload(
            payload_path=tmp_path / "nope.json",
            metadata=SubmissionMetadata(),
        )
        assert isinstance(result, SubmissionResult)
        assert result.ok is False
        assert result.status_code is None
        assert "not found" in result.message.lower()

    def test_invalid_json_returns_failure(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("not valid json {{{")
        result = submit_payload(
            payload_path=bad, metadata=SubmissionMetadata(),
        )
        assert result.ok is False
        assert result.status_code is None
        assert "JSON" in result.message


class _FakeResponse:
    """Stand-in for ``requests.Response`` -- only the two fields the
    helper reads. Module-level so all three failure-bucket tests share
    one shape."""
    def __init__(self, status_code=200, text="OK"):
        self.status_code = status_code
        self.text = text


class TestSubmitPayloadHappyPath:
    def test_post_includes_metadata_and_file(self, tmp_path, monkeypatch):
        """The POST must include both the metadata form-data fields
        AND the JSON file as ``jsonFile``. Capture the call args via
        a stub ``requests.post`` (patched on the real ``requests``
        module since ``submit_payload`` lazy-imports it) and assert
        the wire shape."""
        import requests

        captured = {}

        def _fake_post(url, *, data=None, files=None, timeout=None):
            captured["url"] = url
            captured["data"] = data
            # Resolve the file tuple before the caller closes the
            # underlying file object so we can read the filename.
            files_snapshot = {}
            for key, (filename, fh, mime) in (files or {}).items():
                files_snapshot[key] = (filename, mime)
            captured["files"] = files_snapshot
            captured["timeout"] = timeout
            return _FakeResponse(status_code=200, text="OK")

        monkeypatch.setattr(requests, "post", _fake_post)

        payload = _write_valid_json(tmp_path)
        metadata = _filled_metadata()
        result = submit_payload(
            payload_path=payload, metadata=metadata,
            target_url="http://test.local/upload_json",
        )
        assert result.ok is True
        assert result.status_code == 200
        assert captured["url"] == "http://test.local/upload_json"
        # Metadata travelled as form-data with the wire-name overrides
        # applied (area_codes -> area-codes, health_status -> health-status).
        assert captured["data"]["name"] == "Pat"
        assert captured["data"]["area-codes"] == "A1"
        assert captured["data"]["health-status"] == "untested"
        # File travelled as jsonFile.
        assert "jsonFile" in captured["files"]
        filename, mime = captured["files"]["jsonFile"]
        assert filename == "montage.json"
        assert mime == "application/json"


class TestSubmitPayloadFailureBuckets:
    """Three failure buckets the panel needs to discriminate:
    bad-status (HTTP 4xx/5xx with body), transport error (no
    status_code), and pre-flight (also no status_code)."""

    def test_non_200_status_returns_failure_with_body(self, tmp_path, monkeypatch):
        import requests

        def _fake_post(url, **kwargs):
            return _FakeResponse(
                status_code=429, text="Rate limit -- please wait.",
            )

        monkeypatch.setattr(requests, "post", _fake_post)

        result = submit_payload(
            payload_path=_write_valid_json(tmp_path),
            metadata=_filled_metadata(),
            target_url="http://test.local/upload_json",
        )
        assert result.ok is False
        assert result.status_code == 429
        assert "Rate limit" in result.message

    def test_transport_exception_returns_failure_with_no_status(
        self, tmp_path, monkeypatch,
    ):
        import requests

        def _fake_post(url, **kwargs):
            raise ConnectionError("no route to host")

        monkeypatch.setattr(requests, "post", _fake_post)

        result = submit_payload(
            payload_path=_write_valid_json(tmp_path),
            metadata=_filled_metadata(),
            target_url="http://test.local/upload_json",
        )
        assert result.ok is False
        assert result.status_code is None
        assert "ConnectionError" in result.message
        assert "no route to host" in result.message
