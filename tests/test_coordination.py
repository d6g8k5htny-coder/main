"""Negative controls for tools/coordination_check.py.

The board's one job is to stop two agents writing the same files from different
branches. A board that reports a clean run while two disjoint agents both
declare `registers/json/*` is worse than no board, because it is an assurance
nobody earned. So most of what follows breaks the checker on purpose and
demands that it notice.

What these tests do not establish: nothing here verifies that a claim is
truthful, that an agent did the work it declared, or anything mathematical.
They test one checker's refusals.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "coordination_check.py")
SCHEMA = os.path.join(ROOT, "coordination", "board.schema.json")

TASK = {
    "task_id": "TASK-ALPHA",
    "title": "A declared unit of work with a long enough title",
    "declared_utc": "2026-09-23T12:00:00Z",
    "declared_by": "claude",
    "paths": ["research/alpha/*"],
    "importance": "NORMAL",
    "intent": "x" * 60,
    "done_when": "The alpha driver runs and its controls pass.",
    "does_not_establish": "d" * 80,
}

CLAIM = {
    "task_id": "TASK-ALPHA",
    "agent": "claude",
    "provider": "Anthropic",
    "claimed_utc": "2026-09-23T12:00:00Z",
    "heartbeat_utc": "2026-09-23T12:00:00Z",
    "approach": "a" * 60,
}


def write_task(root, task, claims=()):
    d = root / task["task_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "TASK.json").write_text(json.dumps(task), encoding="utf-8")
    for claim in claims:
        (d / f"claim-{claim['agent']}.json").write_text(json.dumps(claim), encoding="utf-8")
    return d


def run(tasks_dir, *extra):
    proc = subprocess.run(
        [sys.executable, TOOL, "--tasks", str(tasks_dir), "--schema", SCHEMA, *extra],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, proc.stdout


def variant(base, **over):
    out = dict(base)
    out.update(over)
    return out


# ------------------------------------------------------------ the happy board

def test_a_well_formed_board_passes(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 0, out
    assert "problems=0" in out


def test_the_real_board_in_this_repository_passes():
    code, out = run(os.path.join(ROOT, "coordination", "tasks"))
    assert code == 0, out


# --------------------------------------------------- the collision it prevents

def test_two_disjoint_agents_over_the_same_paths_is_refused(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    write_task(tmp_path,
               variant(TASK, task_id="TASK-BETA", paths=["research/alpha/driver.py"]),
               [variant(CLAIM, task_id="TASK-BETA", agent="chatgpt", provider="OpenAI")])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "nothing is coordinating these writes" in out


def test_a_shared_claimant_is_the_coordination_and_is_allowed(tmp_path):
    """Overlap is fine when one agent holds both tasks -- that agent knows."""
    write_task(tmp_path, TASK, [CLAIM])
    write_task(tmp_path,
               variant(TASK, task_id="TASK-BETA", paths=["research/alpha/driver.py"]),
               [variant(CLAIM, task_id="TASK-BETA"),
                variant(CLAIM, task_id="TASK-BETA", agent="chatgpt", provider="OpenAI")])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 0, out


def test_an_unclaimed_task_cannot_collide(tmp_path):
    """Nobody is writing yet, so a declared overlap is not yet a collision."""
    write_task(tmp_path, TASK, [CLAIM])
    write_task(tmp_path, variant(TASK, task_id="TASK-BETA", paths=["research/alpha/x.py"]))
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 0, out


@pytest.mark.parametrize("a,b", [
    ("registers/json/*", "registers/json/review_queue.json"),
    ("engine/lanes", "engine/lanes/D.json"),
    ("tools/frozen_check.py", "tools/frozen_check.py"),
    ("drive/*", "drive/inventory.jsonl"),
])
def test_overlapping_shapes_are_detected_in_both_directions(tmp_path, a, b):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_coord", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.globs_overlap(a, b), (a, b)
    assert mod.globs_overlap(b, a), (b, a)


def test_unrelated_paths_do_not_collide(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    write_task(tmp_path, variant(TASK, task_id="TASK-BETA", paths=["research/beta/*"]),
               [variant(CLAIM, task_id="TASK-BETA", agent="chatgpt", provider="OpenAI")])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 0, out


# -------------------------------------------------------------- the handover

def test_a_claim_past_the_window_is_reported_stale(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-24T12:00:00Z")
    assert "STALE" in out
    assert "may join or take it over" in out


def test_staleness_is_reported_and_never_fails_the_run(tmp_path):
    """A silent agent is not a defect; the board only says the task is joinable."""
    write_task(tmp_path, TASK, [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-30T12:00:00Z")
    assert code == 0, out
    assert "stale=1" in out


def test_a_fresh_heartbeat_is_not_stale(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T12:30:00Z")
    assert "STALE" not in out
    assert "stale=0" in out


# ------------------------------------------------------- the status firewall

@pytest.mark.parametrize("field", ["grade", "verdict", "status", "independence_credit",
                                   "gate_status", "confidence"])
def test_a_status_field_on_a_claim_is_refused(tmp_path, field):
    write_task(tmp_path, TASK, [dict(CLAIM, **{field: "PASS"})])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out


def test_promotion_language_in_an_intent_is_refused(tmp_path):
    write_task(tmp_path, variant(TASK, intent="The obligation is discharged. " + "x" * 40), [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "cannot report a result" in out


def test_does_not_establish_may_name_what_it_does_not_do(tmp_path):
    """The one field where the forbidden words belong, so it can deny them."""
    task = variant(TASK, does_not_establish=(
        "Nothing here is discharged and this promotes no claim whatsoever. " * 2))
    write_task(tmp_path, task, [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 0, out


# ---------------------------------------------------- one claim file per agent

def test_a_claim_filename_that_disagrees_with_its_agent_is_refused(tmp_path):
    d = write_task(tmp_path, TASK, [CLAIM])
    (d / "claim-chatgpt.json").write_text(json.dumps(CLAIM), encoding="utf-8")
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "one claim file per agent" in out


def test_a_task_id_that_disagrees_with_its_directory_is_refused(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    os.rename(tmp_path / "TASK-ALPHA", tmp_path / "TASK-GAMMA")
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "does not match its directory name" in out


def test_a_stray_file_in_a_task_directory_is_refused(tmp_path):
    d = write_task(tmp_path, TASK, [CLAIM])
    (d / "notes.json").write_text("{}", encoding="utf-8")
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out


def test_a_directory_without_a_task_declaration_is_refused(tmp_path):
    (tmp_path / "TASK-EMPTY").mkdir()
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "has no TASK.json" in out


# ------------------------------------------------------------ schema refusals

@pytest.mark.parametrize("bad", [
    {"importance": "URGENT"},
    {"paths": []},
    {"does_not_establish": "too short"},
    {"title": "short"},
    {"declared_utc": "2026-09-23 12:00:00"},
    {"declared_by": "Claude Opus"},
])
def test_a_malformed_task_is_refused(tmp_path, bad):
    write_task(tmp_path, variant(TASK, **bad), [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out


def test_an_unknown_task_field_is_refused(tmp_path):
    write_task(tmp_path, variant(TASK, priority="high"), [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out


# ------------------------------------------------------------- the ranking

def test_the_ranking_puts_important_and_lonely_tasks_first(tmp_path):
    write_task(tmp_path, variant(TASK, task_id="TASK-LOW", importance="LOW"), [CLAIM])
    write_task(tmp_path, variant(TASK, task_id="TASK-BLOCK", importance="BLOCKING",
                                 paths=["research/block/*"]))
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z", "--rank")
    rows = [ln for ln in out.splitlines() if "claimants=" in ln]
    assert "TASK-BLOCK" in rows[0], rows
    assert "TASK-LOW" in rows[-1], rows


def test_the_ranking_prefers_the_task_with_fewer_claimants_at_equal_importance(tmp_path):
    write_task(tmp_path, variant(TASK, task_id="TASK-CROWDED"),
               [CLAIM, variant(CLAIM, agent="chatgpt", provider="OpenAI")])
    write_task(tmp_path, variant(TASK, task_id="TASK-LONELY", paths=["research/lonely/*"]),
               [variant(CLAIM, task_id="TASK-LONELY", agent="cursor", provider="Cursor")])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z", "--rank")
    rows = [ln for ln in out.splitlines() if "claimants=" in ln]
    assert "TASK-LONELY" in rows[0], rows


# ------------------------------------------------------------- honesty pins

def test_the_module_states_what_it_does_not_establish():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_coord", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    doc = mod.__doc__ or ""
    assert "DOES NOT ESTABLISH" in doc
    assert "NOT DEPLOYED" in doc
    for phrase in ("independence credit", "moves no gate"):
        assert phrase in doc


def test_the_run_prints_what_a_board_entry_is_not(tmp_path):
    write_task(tmp_path, TASK, [CLAIM])
    _code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert "verifies no mathematics" in out
    assert "earns no independence credit" in out


def test_the_schema_uses_no_ref_because_the_validator_cannot_resolve_one(tmp_path):
    """The defect these controls caught, pinned so it cannot come back.

    The board schema first wrote its timestamp, agent and task-id constraints as
    $ref into $defs. tools/reviews_check.validate_instance supports pattern but
    not $ref, so every one of those constraints was inert: a malformed
    declared_utc and a declared_by with a space both sailed through a schema
    that looked strict. Constraints are inlined now, and a reintroduced $ref
    under task or claim fails here rather than silently checking nothing.
    """
    with open(SCHEMA, encoding="utf-8") as handle:
        schema = json.load(handle)

    def refs(node):
        if isinstance(node, dict):
            return ("$ref" in node) or any(refs(v) for v in node.values())
        if isinstance(node, list):
            return any(refs(v) for v in node)
        return False

    for section in ("task", "claim"):
        assert not refs(schema["$defs"][section]), f"$ref under {section} is not enforced"


def test_an_inlined_pattern_actually_rejects_a_bad_timestamp(tmp_path):
    """Belt to the braces above: prove the constraint fires, not just that it exists."""
    write_task(tmp_path, variant(TASK, declared_utc="yesterday"), [CLAIM])
    code, out = run(tmp_path, "--now", "2026-09-23T13:00:00Z")
    assert code == 1, out
    assert "does not match" in out
