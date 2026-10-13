#!/usr/bin/env python3
"""Fail-closed checker for governance work-lease / collision ledger.

Scientific effect: NONE. Coordination coverage only — not claim status,
not review acceptance, not theorem discharge.

Catches:
- ACTIVE lease with empty evidence
- ACTIVE lease past expires_at (stale heartbeat)
- two ACTIVE leases with overlapping write_scope (unless delegates_to)
- ACTIVE lease whose source.pr_state is closed/merged
- scientific_authority not false
- duplicate work_id
- invalid state token
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ALLOWED_STATES = frozenset({"OFFERED", "ACTIVE", "BLOCKED", "RELEASED", "SUPERSEDED"})
CLOSED_PR_STATES = frozenset({"closed", "merged"})
ROOT = Path(__file__).resolve().parent
DEFAULT_LEDGER = ROOT / "CURRENT.json"


def parse_ts(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def path_overlap(a: str, b: str) -> bool:
    """True when two write_scope entries collide (exact or prefix)."""
    if not a or not b:
        return False
    a = a.rstrip("/")
    b = b.rstrip("/")
    if a == b:
        return True
    # Directory prefix: "docs/" overlaps "docs/foo.md"; file vs sibling file does not.
    if a.endswith((".py", ".md", ".json", ".yml", ".yaml")) or "/" not in a + "/":
        a_is_file = "." in Path(a).name
    else:
        a_is_file = False
    if b.endswith((".py", ".md", ".json", ".yml", ".yaml")) or "/" not in b + "/":
        b_is_file = "." in Path(b).name
    else:
        b_is_file = False
    if a_is_file and b_is_file:
        return a == b
    # Treat trailing-path as directory prefix when either side looks like a dir claim.
    a_pref = a if a_is_file else a + "/"
    b_pref = b if b_is_file else b + "/"
    if not a_is_file and (b == a or b.startswith(a + "/")):
        return True
    if not b_is_file and (a == b or a.startswith(b + "/")):
        return True
    # File under directory
    if a_is_file and not b_is_file and a.startswith(b + "/"):
        return True
    if b_is_file and not a_is_file and b.startswith(a + "/"):
        return True
    # Also: exact directory string without slash vs child
    if not a_is_file and b.startswith(a_pref):
        return True
    if not b_is_file and a.startswith(b_pref):
        return True
    return False


def scopes_overlap(left: list[str], right: list[str]) -> list[tuple[str, str]]:
    hits: list[tuple[str, str]] = []
    for a in left:
        for b in right:
            if path_overlap(a, b):
                hits.append((a, b))
    return hits


def delegation_allows(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("delegates_to") == b.get("work_id") or b.get("delegates_to") == a.get("work_id")


def check_lease(lease: dict[str, Any], now: datetime, problems: list[str]) -> None:
    wid = lease.get("work_id", "<missing>")
    state = lease.get("state")
    if state not in ALLOWED_STATES:
        problems.append(f"{wid}: invalid state {state!r}")
        return
    if lease.get("scientific_authority") is not False:
        problems.append(f"{wid}: scientific_authority must be false")
    if state != "ACTIVE":
        return
    evidence = lease.get("evidence") or []
    if not evidence:
        problems.append(f"{wid}: ACTIVE requires nonempty evidence")
    try:
        expires = parse_ts(lease["expires_at"])
        heartbeat = parse_ts(lease["heartbeat_at"])
    except (KeyError, ValueError) as exc:
        problems.append(f"{wid}: bad timestamps ({exc})")
        return
    if heartbeat >= expires:
        problems.append(f"{wid}: heartbeat_at must be before expires_at")
    if now >= expires:
        problems.append(f"{wid}: ACTIVE lease expired at {lease['expires_at']} (stale; move to BLOCKED/RELEASED)")
    source = lease.get("source") or {}
    pr_state = source.get("pr_state")
    if pr_state in CLOSED_PR_STATES:
        problems.append(
            f"{wid}: ACTIVE lease has pr_state={pr_state!r}; release/supersede when PR closes"
        )


def check_ledger(doc: dict[str, Any], now: datetime | None = None) -> list[str]:
    now = now or datetime.now(timezone.utc)
    problems: list[str] = []
    if doc.get("scientific_authority") is not False:
        problems.append("ledger scientific_authority must be false")
    leases = doc.get("leases")
    if not isinstance(leases, list):
        return problems + ["leases must be a list"]
    seen: dict[str, dict[str, Any]] = {}
    for lease in leases:
        if not isinstance(lease, dict):
            problems.append("lease entry must be an object")
            continue
        wid = lease.get("work_id")
        if not isinstance(wid, str) or not wid:
            problems.append("lease missing work_id")
            continue
        if wid in seen:
            problems.append(f"duplicate work_id {wid}")
        seen[wid] = lease
        check_lease(lease, now, problems)

    active = [L for L in seen.values() if L.get("state") == "ACTIVE"]
    for i, a in enumerate(active):
        for b in active[i + 1 :]:
            hits = scopes_overlap(a.get("write_scope") or [], b.get("write_scope") or [])
            if hits and not delegation_allows(a, b):
                sample = ", ".join(f"{x}~{y}" for x, y in hits[:3])
                problems.append(
                    f"ACTIVE write_scope overlap: {a['work_id']} vs {b['work_id']} ({sample})"
                )
    return problems


def self_test() -> None:
    """Synthetic fixtures proving the two required catches."""
    now = parse_ts("2026-09-25T18:30:00Z")
    base = {
        "schema": "governance.work-leases/v1",
        "scientific_authority": False,
        "leases": [],
    }

    overlap_doc = {
        **base,
        "leases": [
            {
                "work_id": "a",
                "state": "ACTIVE",
                "scientific_authority": False,
                "write_scope": ["tools/foo.py"],
                "evidence": ["https://example.test/a"],
                "claimed_at": "2026-09-25T18:00:00Z",
                "heartbeat_at": "2026-09-25T18:10:00Z",
                "expires_at": "2026-09-26T00:00:00Z",
                "source": {"pr_state": "open"},
                "delegates_to": None,
            },
            {
                "work_id": "b",
                "state": "ACTIVE",
                "scientific_authority": False,
                "write_scope": ["tools/"],
                "evidence": ["https://example.test/b"],
                "claimed_at": "2026-09-25T18:00:00Z",
                "heartbeat_at": "2026-09-25T18:10:00Z",
                "expires_at": "2026-09-26T00:00:00Z",
                "source": {"pr_state": "open"},
                "delegates_to": None,
            },
        ],
    }
    overlap_problems = check_ledger(overlap_doc, now=now)
    if not any("write_scope overlap" in p for p in overlap_problems):
        raise AssertionError(f"expected overlap catch, got {overlap_problems}")

    closed_doc = {
        **base,
        "leases": [
            {
                "work_id": "c",
                "state": "ACTIVE",
                "scientific_authority": False,
                "write_scope": ["docs/x.md"],
                "evidence": ["https://example.test/c"],
                "claimed_at": "2026-09-25T18:00:00Z",
                "heartbeat_at": "2026-09-25T18:10:00Z",
                "expires_at": "2026-09-26T00:00:00Z",
                "source": {"pr": 1, "pr_state": "closed"},
                "delegates_to": None,
            }
        ],
    }
    closed_problems = check_ledger(closed_doc, now=now)
    if not any("pr_state='closed'" in p or 'pr_state="closed"' in p or "pr_state='closed'" in p for p in closed_problems):
        if not any("pr_state=" in p and "closed" in p for p in closed_problems):
            raise AssertionError(f"expected closed-PR catch, got {closed_problems}")

    offered_doc = {
        **base,
        "leases": [
            {
                "work_id": "d",
                "state": "OFFERED",
                "scientific_authority": False,
                "write_scope": ["anywhere/"],
                "evidence": [],
                "claimed_at": "2026-09-25T18:00:00Z",
                "heartbeat_at": "2026-09-25T18:00:00Z",
                "expires_at": "2026-09-26T00:00:00Z",
                "source": {"pr_state": None},
                "delegates_to": None,
            }
        ],
    }
    if check_ledger(offered_doc, now=now):
        raise AssertionError("OFFERED lease must not be treated as ACTIVE failure")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--now",
        default=None,
        help="ISO-8601 UTC override for expiry checks (tests/replay)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        print("work_leases self-test: OK")
        return 0

    doc = json.loads(args.ledger.read_text(encoding="utf-8"))
    now = parse_ts(args.now) if args.now else datetime.now(timezone.utc)
    problems = check_ledger(doc, now=now)
    if problems:
        print("work_leases: FAIL", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    n = len(doc.get("leases") or [])
    active = sum(1 for L in doc.get("leases") or [] if L.get("state") == "ACTIVE")
    offered = sum(1 for L in doc.get("leases") or [] if L.get("state") == "OFFERED")
    print(
        f"work_leases: OK leases={n} ACTIVE={active} OFFERED={offered} "
        f"scientific_authority=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
