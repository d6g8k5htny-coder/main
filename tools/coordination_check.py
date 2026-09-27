"""Invariants for the coordination board in `coordination/`.

Four agents push to this repository concurrently, and the thing they collide on
is not usually the mathematics -- it is two of them editing the same file from
different branches. The board is how a task is declared before the work starts,
so another agent can see it and either pick a different angle on it or pick
something else entirely.

The board is built so that USING it cannot cause the collision it prevents. A
single shared ledger file would be edited by every agent on every claim, which
is a merge conflict generator wearing the costume of a solution. Instead:

* a task is one directory, `coordination/tasks/<TASK-ID>/`;
* its declaration is `TASK.json`, written once by whoever declares it;
* each agent working it adds `claim-<agent>.json` -- its own file, nobody
  else's. Claiming is a file creation, releasing is a deletion, and neither
  touches a line another agent wrote.

So several agents can join one task without contending, which is the point:
more than one claimant per task is normal, not an exception, and the ranking
this tool prints puts high-importance tasks with few claimants at the top so
they are the obvious ones to join.

A claim is a work-scheduling record and nothing else. It cannot promote, close,
discharge or reclassify anything, and the checks below refuse a board file that
tries: no grade, no verdict, no confidence vote, no independence credit. This is
the same firewall `engine/next_action.py` carries -- a ranking is a heuristic
about what to do next and carries no mathematical authority whatsoever.

STALENESS. A claim carries `heartbeat_utc`. Past `STALE_AFTER_HOURS` the tool
reports the claim STALE, which means the task is open for another agent to
pick up. Staleness is reported, never enforced: this tool does not delete
another agent's claim, and a stale claim is not a statement that the agent
failed -- only that the board has not heard from it. The window is an
agent-chosen working number, revisable by the agents, not an owner rule.

WHAT THIS DOES NOT ESTABLISH. A clean board says who said they were working on
what, and that no two declared tasks claim the same paths. It does not verify
that any work happened, that a claim is truthful, that the paths a task
declares are the paths it will actually touch, or that anything it produced is
correct. It transcribes no status, moves no gate, discharges no premise and
awards zero organizational independence credit. NOT DEPLOYED as an enforcement
mechanism: nothing in CI prevents an agent from writing a file it never claimed.

Run: python3 tools/coordination_check.py [--tasks DIR] [--now UTC] [--rank]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reviews_check import (  # noqa: E402  - resolved at call time, not import time
    CONFIDENCE_PATTERNS,
    PROMOTION_PHRASES,
    load_json,
    normalize,
    validate_instance,
    walk_strings,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TASKS = os.path.join(ROOT, "coordination", "tasks")
DEFAULT_DONE = os.path.join(ROOT, "coordination", "done")
DEFAULT_SCHEMA = os.path.join(ROOT, "coordination", "board.schema.json")

# An agent-chosen working window, not an owner rule. Past this with no
# heartbeat, the board reports the claim stale and the task joinable.
STALE_AFTER_HOURS = 3

IMPORTANCE_ORDER = {"BLOCKING": 0, "HIGH": 1, "NORMAL": 2, "LOW": 3}

# A board file states intent. A word from this family states a result, and the
# board is the wrong place for one: it would let a scheduling note read as a
# verdict. Deliberately overlaps reviews_check's vocabulary rather than
# inventing a second one.
FORBIDDEN_FIELDS = ("grade", "verdict", "status", "technical_status",
                    "independence_credit", "gate_status", "claim_grade",
                    "confidence", "promoted", "closed", "discharged")

CLAIM_RE = re.compile(r"^claim-([a-z0-9][a-z0-9._-]{1,40})\.json$")


def parse_utc(value: str) -> _dt.datetime:
    return _dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)


def globs_overlap(a: str, b: str) -> bool:
    """True when two declared path patterns can name the same file.

    Neither glob library matches globs against globs, so each side is tested
    against the other's literal prefix as well. It over-reports rather than
    under-reports: a false overlap costs a conversation, a missed one costs the
    collision this board exists to prevent.
    """
    a, b = a.strip().rstrip("/"), b.strip().rstrip("/")
    if not a or not b:
        return False
    if a == b:
        return True
    if fnmatch.fnmatch(a, b) or fnmatch.fnmatch(b, a):
        return True
    # a directory declaration covers everything beneath it
    return a.startswith(b + "/") or b.startswith(a + "/")


def read_board(tasks_dir: str, schema: dict) -> tuple[list[dict], list[str]]:
    """[{task, claims:[...]}] plus structural problems."""
    problems: list[str] = []
    board: list[dict] = []
    if not os.path.isdir(tasks_dir):
        return board, problems
    for name in sorted(os.listdir(tasks_dir)):
        d = os.path.join(tasks_dir, name)
        if not os.path.isdir(d):
            continue
        tpath = os.path.join(d, "TASK.json")
        if not os.path.isfile(tpath):
            problems.append(f"{name}: directory has no TASK.json")
            continue
        try:
            task = load_json(tpath)
        except ValueError as exc:
            problems.append(f"{name}/TASK.json: {exc}")
            continue
        problems += [f"{name}/TASK.json: {p}"
                     for p in validate_instance(task, schema["$defs"]["task"])]
        if task.get("task_id") != name:
            problems.append(f"{name}/TASK.json: task_id {task.get('task_id')!r} "
                            f"does not match its directory name")
        claims = []
        for fn in sorted(os.listdir(d)):
            if fn == "TASK.json":
                continue
            m = CLAIM_RE.match(fn)
            if not m:
                problems.append(f"{name}/{fn}: not TASK.json and not claim-<agent>.json")
                continue
            try:
                claim = load_json(os.path.join(d, fn))
            except ValueError as exc:
                problems.append(f"{name}/{fn}: {exc}")
                continue
            problems += [f"{name}/{fn}: {p}"
                         for p in validate_instance(claim, schema["$defs"]["claim"])]
            if claim.get("agent") != m.group(1):
                problems.append(f"{name}/{fn}: agent {claim.get('agent')!r} does not "
                                f"match the filename; one claim file per agent")
            if claim.get("task_id") != name:
                problems.append(f"{name}/{fn}: task_id does not match its directory")
            claims.append(claim)
        board.append({"task": task, "claims": claims, "dir": d})
    return board, problems


def honesty_problems(board: list[dict]) -> list[str]:
    """The board schedules work; it never reports a result."""
    out: list[str] = []
    for entry in board:
        for label, doc in ([("TASK.json", entry["task"])] +
                           [(f"claim-{c.get('agent')}.json", c) for c in entry["claims"]]):
            tid = entry["task"].get("task_id", "?")
            for where, text in walk_strings(doc):
                if where.endswith("does_not_establish"):
                    continue
                flat = normalize(text)
                for phrase in PROMOTION_PHRASES:
                    if phrase in flat:
                        out.append(f"{tid}/{label}: {where} says {phrase!r}; a board entry "
                                   f"declares intent and cannot report a result")
                for pattern in CONFIDENCE_PATTERNS:
                    if pattern.search(flat):
                        out.append(f"{tid}/{label}: {where} votes confidence; the board "
                                   f"schedules work and does not weigh evidence")
            for field in FORBIDDEN_FIELDS:
                if isinstance(doc, dict) and field in doc:
                    out.append(f"{tid}/{label}: carries {field!r}; a scheduling record "
                               f"never carries a status field")
    return out


def overlap_problems(board: list[dict]) -> list[str]:
    """Two live tasks must not declare the same paths unless they share an agent.

    Sharing a claimant is the coordination: the same agent holding both knows
    about both. Disjoint agent sets over the same files is the collision.
    """
    out: list[str] = []
    live = [e for e in board if e["claims"]]
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            agents_a = {c.get("agent") for c in a["claims"]}
            agents_b = {c.get("agent") for c in b["claims"]}
            if agents_a & agents_b:
                continue
            hits = sorted({(pa, pb)
                           for pa in a["task"].get("paths", [])
                           for pb in b["task"].get("paths", [])
                           if globs_overlap(pa, pb)})
            for pa, pb in hits:
                out.append(
                    f"{a['task']['task_id']} ({', '.join(sorted(agents_a))}) declares "
                    f"{pa!r} and {b['task']['task_id']} ({', '.join(sorted(agents_b))}) "
                    f"declares {pb!r}; no agent holds both, so nothing is coordinating "
                    f"these writes")
    return out


def staleness(board: list[dict], now: _dt.datetime) -> list[tuple[str, str, float]]:
    """(task_id, agent, hours_since_heartbeat) for every claim past the window."""
    out = []
    for entry in board:
        for claim in entry["claims"]:
            hb = claim.get("heartbeat_utc")
            if not isinstance(hb, str):
                continue
            try:
                age = (now - parse_utc(hb)).total_seconds() / 3600.0
            except ValueError:
                continue
            if age > STALE_AFTER_HOURS:
                out.append((entry["task"]["task_id"], claim.get("agent", "?"), age))
    return sorted(out, key=lambda r: -r[2])


def ranking(board: list[dict]) -> list[tuple[str, str, int, str]]:
    """Joinable tasks first: most important, fewest claimants."""
    rows = []
    for entry in board:
        task = entry["task"]
        rows.append((task.get("importance", "NORMAL"), task.get("task_id", "?"),
                     len(entry["claims"]), task.get("title", "")))
    return sorted(rows, key=lambda r: (IMPORTANCE_ORDER.get(r[0], 9), r[2]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--done", default=DEFAULT_DONE)
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--now", default=None, help="UTC instant to age heartbeats against")
    ap.add_argument("--rank", action="store_true", help="print the join-me-next ranking")
    args = ap.parse_args(argv)

    schema = load_json(args.schema)
    board, problems = read_board(args.tasks, schema)
    problems += honesty_problems(board)
    problems += overlap_problems(board)

    now = parse_utc(args.now) if args.now else _dt.datetime.now(_dt.timezone.utc)
    stale = staleness(board, now)

    for p in problems:
        print(f"PROBLEM  {p}")
    for tid, agent, hours in stale:
        print(f"STALE    {tid}: {agent} last checked in {hours:.1f}h ago "
              f"(> {STALE_AFTER_HOURS}h); another agent may join or take it over")
    if args.rank:
        for importance, tid, n, title in ranking(board):
            print(f"  [{importance:8s}] claimants={n}  {tid}  {title}")

    unclaimed = sum(1 for e in board if not e["claims"])
    claims = sum(len(e["claims"]) for e in board)
    print(f"coordination_check: tasks={len(board)} claims={claims} "
          f"unclaimed={unclaimed} stale={len(stale)} problems={len(problems)}")
    print("A board entry is a work-scheduling record. It verifies no mathematics, "
          "moves no gate, discharges no premise and earns no independence credit.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
