"""Pilot boundary negative controls; fixtures authorize no live execution."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from engine.bridge import pilot, run_receipt as rr, work_order as wo

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "fixture-source-id-123456"
START = "2026-09-19T22:00:00Z"
END = "2026-09-19T22:00:01Z"
LEASE = "2026-09-19T22:10:00Z"
COMMAND = "python tools/manifest_integrity_check.py --coverage .github/manifest-coverage.json"
OUTPUT = "sandbox/PILOT/check.log"


@pytest.fixture
def inputs():
    example = next((ROOT / "engine/bridge/examples").glob("*work_order*.json"))
    order = json.loads(example.read_text())
    order.update(record_kind="RECORD", status="EXECUTABLE", task_id="PILOT-FIXTURE")
    order["repository"]["base_commit_sha"] = "a" * 40
    order["authorization"].update(approved_principal_ids=["fixture-principal"],
                                  source_ref=SOURCE_ID, authorizing_record_drive_id=SOURCE_ID,
                                  verification_status="VERIFIED_BY_OWNER_BOUNDARY")
    sources = {SOURCE_ID: b"fixture authority, not real authorization"}
    order["governing_sources"] = [{"drive_id": SOURCE_ID, "bytes": len(sources[SOURCE_ID]),
        "sha256": sha256(sources[SOURCE_ID]).hexdigest(), "extraction_rule": "raw bytes",
        "revision_if_native": None}]
    order["claim"] = {"work_events_event_id": "fixture-claim", "lease_until_utc": LEASE}
    order["scope"].update(objective="Exercise a manual bridge fixture only",
                          allowed_paths=["sandbox/PILOT/"], allowed_commands=[COMMAND])
    order = wo.with_digest(order)
    boundary = pilot.BoundarySnapshot(order["repository"]["id"], "a" * 40, "b" * 40,
        False, "fixture-principal", "fixture-claim", LEASE, True, None, START)
    return order, boundary, sources


def receipt(inputs, **changes):
    order, boundary, sources = inputs
    kwargs = dict(run_id="PILOT-RUN-1", actor={"authenticated_principal_id": "fixture-principal",
        "declared_provider": "fixture-provider", "declared_model": None,
        "session_id": "fixture-session", "exposure": "fixture-only observations"},
        environment="fixture environment, no live execution", commands=[COMMAND], exit_codes=[0],
        start_utc=START, end_utc=END, outputs={OUTPUT: b"check passed\n"},
        negative_controls=["stale source bytes are refused before execution"],
        coverage="One manifest checker command, fixture observation only.", dirty_after=False)
    kwargs.update(changes)
    return pilot.executed_receipt(order, boundary, sources, **kwargs)


def archive(executed, **changes):
    kwargs = dict(receipt_file_id="fixture-receipt-id-1234", delivery_id="fixture-delivery-1234",
        receipt_readback=rr.to_json(executed).encode(), output_readbacks={OUTPUT: b"check passed\n"},
        readback_utc="2026-09-19T22:00:02Z", observed_output_head=None)
    kwargs.update(changes)
    return pilot.recorded_receipt(executed, **kwargs)


def test_success_records_exact_counts_and_output(inputs):
    order, boundary, sources = inputs
    pilot.preflight(order, boundary, sources, START)
    result = receipt(inputs)
    assert rr.validate_run_receipt(result) == []
    assert rr.check_receipt_against_order(result, order) == []
    assert result["verification"]["tests_passed"] == 1
    assert result["verification"]["tests_failed"] == 0
    assert result["outputs"][0]["sha256"] == sha256(b"check passed\n").hexdigest()
    assert result["review"]["technical_verdict"] == "NOT_REVIEWED"
    assert result["scientific_status_change"] == "UNCHANGED"


def test_deliberate_failed_command_remains_failure(inputs):
    result = receipt(inputs, exit_codes=[7])
    assert result["status"] == "EXECUTED"
    assert result["execution"]["exit_codes"] == [7]
    assert result["verification"]["tests_failed"] == 1
    assert result["verification"]["tests_passed"] == 0
    assert archive(result)["verification"]["tests_failed"] == 1


@pytest.mark.parametrize("changes", [
    {"repository_id": 12}, {"commit_sha": "c" * 40}, {"tree_sha": "invalid"},
    {"dirty_worktree": True}, {"dirty_worktree": 0}, {"principal_id": "unapproved-principal"},
    {"claim_event_id": "wrong-claim"}, {"claim_active": False}, {"claim_active": 1},
    {"claim_lease_until_utc": START}, {"claim_lease_until_utc": "invalid"},
    {"output_head": "new-output"}, {"observed_utc": "2026-09-19T21:54:59Z"},
    {"observed_utc": END}, {"observed_utc": "invalid"},
])
def test_preflight_refuses_stale_or_inconsistent_boundary(inputs, changes):
    order, boundary, sources = inputs
    with pytest.raises(pilot.PilotRefused):
        pilot.preflight(order, replace(boundary, **changes), sources, START)


@pytest.mark.parametrize("sources", [{}, {SOURCE_ID: b"changed bytes"},
    {SOURCE_ID: b"fixture authority, not real authorization", "unexpected-id": b"extra"},
    {SOURCE_ID: "fixture authority, not real authorization"}])
def test_preflight_refuses_missing_or_changed_sources(inputs, sources):
    order, boundary, _ = inputs
    with pytest.raises(pilot.PilotRefused):
        pilot.preflight(order, boundary, sources, START)


def test_authority_must_be_pinned_not_merely_named(inputs):
    order, boundary, sources = inputs
    order["authorization"]["authorizing_record_drive_id"] = "different-authority-1234"
    with pytest.raises(pilot.PilotRefused, match="byte-pinned"):
        pilot.preflight(wo.with_digest(order), boundary, sources, START)


def test_duplicate_source_is_refused(inputs):
    order, boundary, sources = inputs
    order["governing_sources"] *= 2
    with pytest.raises(pilot.PilotRefused, match="source set"):
        pilot.preflight(wo.with_digest(order), boundary, sources, START)


def test_expired_lease_is_refused_even_with_fresh_observation(inputs):
    order, boundary, sources = inputs
    with pytest.raises(pilot.PilotRefused, match="expired"):
        pilot.preflight(order, replace(boundary, observed_utc=LEASE), sources, LEASE)


@pytest.mark.parametrize("changes", [
    {"exit_codes": []}, {"exit_codes": [False]}, {"commands": []},
    {"commands": ["python unrelated.py"]}, {"end_utc": "invalid"},
    {"end_utc": "2026-09-19T21:59:59Z"}, {"end_utc": "2026-09-19T22:10:01Z"},
    {"outputs": {"claims/forbidden.json": b"output"}}, {"outputs": {OUTPUT: "not bytes"}},
    {"negative_controls": []}, {"dirty_after": None},
])
def test_capture_refuses_incomplete_or_out_of_scope_execution(inputs, changes):
    with pytest.raises(pilot.PilotRefused):
        receipt(inputs, **changes)


def test_execution_duration_is_checked(inputs):
    order, boundary, sources = inputs
    order["limits"]["wall_time_seconds"] = 1
    with pytest.raises(pilot.PilotRefused, match="wall-time"):
        receipt((wo.with_digest(order), boundary, sources), end_utc="2026-09-19T22:00:02Z")


def test_archive_requires_all_exact_readbacks_and_preserves_input(inputs):
    executed = receipt(inputs)
    before = deepcopy(executed)
    recorded = archive(executed)
    assert executed == before
    assert recorded["status"] == "RECORDED"
    assert recorded["idempotency_key"] == executed["idempotency_key"]
    assert recorded["body_sha256"] != executed["body_sha256"]
    assert recorded["drive_return"]["acceptance_status"] == "NOT_ACCEPTED"
    assert rr.validate_run_receipt(recorded) == []


@pytest.mark.parametrize("changes", [
    {"receipt_readback": b"different"}, {"output_readbacks": {}},
    {"output_readbacks": {OUTPUT: b"check failed\n"}},
    {"output_readbacks": {OUTPUT: b"check passed\n", "extra": b"data"}},
    {"observed_output_head": "new-head"}, {"receipt_file_id": "bad"},
    {"delivery_id": "bad"}, {"readback_utc": START}, {"readback_utc": "bad"},
])
def test_archive_refuses_false_recorded_claim(inputs, changes):
    with pytest.raises(pilot.PilotRefused):
        archive(receipt(inputs), **changes)


def test_archive_cannot_be_reapplied_or_accept_work(inputs):
    with pytest.raises(pilot.PilotRefused):
        archive(archive(receipt(inputs)))


def test_duplicate_delivery_is_noop_and_conflict_is_held(inputs, tmp_path):
    executed = receipt(inputs)
    first = rr.store_receipt(executed, str(tmp_path / "execution"))
    before = Path(first.path).read_bytes()
    retry = rr.store_receipt(executed, str(tmp_path / "execution"))
    assert retry.disposition == "IDENTICAL_RETRY" and not retry.wrote
    conflicting = receipt(inputs, exit_codes=[1])
    held = rr.store_receipt(conflicting, str(tmp_path / "execution"))
    assert held.disposition == "NEEDS_RECONCILIATION"
    assert Path(first.path).read_bytes() == before
    recorded = archive(executed)
    assert rr.store_receipt(recorded, str(tmp_path / "recorded")).disposition == "STORED"
    assert rr.store_receipt(recorded, str(tmp_path / "execution")).disposition == "NEEDS_RECONCILIATION"
