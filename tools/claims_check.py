#!/usr/bin/env python3
"""Check the claim/premise dependency graph against the program's firewalls.

`claims/graph.json` encodes, as data, what the Drive sources state in prose:
which claims exist, what grade each carries, which named premises each rests on,
and which compositions are forbidden. This tool turns those firewalls into
assertions a CI run can fail on, so that a future edit cannot quietly:

  * mark a claim unconditional while it still rests on an OPEN premise;
  * compose the 2D track with the 3D lifetime track;
  * let a prize-track claim leak into the q0 dependency graph;
  * drop `original_prize_closed: false` from a prize claim;
  * attach a numerical constant to the qualitative rate or to Theorem B.

It checks the shape of the recorded dependencies. It does not verify any
mathematics, and it never changes a status.

Exit status is non-zero when any firewall is violated.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "claims", "graph.json")

OPEN_STATUSES = {"OPEN", "NOT_CLOSED"}
UNCONDITIONAL_GRADES = {"LIVE_ROOT_THEOREM", "FROZEN_CERTIFICATE", "RATIFIED_3D_ONLY"}
Q0_TRACKS = {"UPPER2D", "LOWER2D"}


def load(path: str | None = None) -> dict:
    # Resolve the module global at call time so tests (and a --graph override)
    # can point the checker at a different graph file.
    with open(path or GRAPH, encoding="utf-8") as f:
        return json.load(f)


def closure(g: dict, name: str, seen: set[str] | None = None) -> set[str]:
    """Transitive set of node names reachable from `name` via depends_on."""
    seen = seen if seen is not None else set()
    if name in seen:
        return seen
    seen.add(name)
    node = g["claims"].get(name) or g["premises"].get(name) or {}
    for dep in node.get("depends_on", []) or []:
        closure(g, dep, seen)
    for sub in node.get("sub_obligations", []) or []:
        closure(g, sub, seen)
    return seen


def main(argv: list[str] | None = None) -> int:
    global GRAPH
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) == 2 and argv[0] == "--graph":
        GRAPH = argv[1]
    g = load()
    claims, premises = g["claims"], g["premises"]
    known = set(claims) | set(premises)
    problems: list[str] = []

    # 0. referential integrity and acyclicity
    for name, node in list(claims.items()) + list(premises.items()):
        for dep in (node.get("depends_on") or []) + (node.get("sub_obligations") or []):
            if dep not in known:
                problems.append(f"{name}: depends on unknown node {dep!r}")

    def find_cycle(name, stack):
        node = claims.get(name) or premises.get(name) or {}
        for dep in (node.get("depends_on") or []) + (node.get("sub_obligations") or []):
            if dep in stack:
                return stack[stack.index(dep):] + [dep]
            if dep in known:
                c = find_cycle(dep, stack + [dep])
                if c:
                    return c
        return None

    for name in known:
        c = find_cycle(name, [name])
        if c:
            problems.append(f"dependency cycle: {' -> '.join(c)}")
            break

    # FW-UNCONDITIONAL
    for name, claim in claims.items():
        if claim.get("grade") not in UNCONDITIONAL_GRADES:
            continue
        for node in closure(g, name) - {name}:
            p = premises.get(node)
            if p and p.get("status_frozen_v2_2") in OPEN_STATUSES:
                problems.append(
                    f"FW-UNCONDITIONAL: {name} is graded {claim['grade']} but rests on "
                    f"{node} (frozen status {p['status_frozen_v2_2']})")

    # FW-2D-3D-COMPOSITION
    for name, claim in claims.items():
        tracks = set()
        for node in closure(g, name):
            t = (claims.get(node) or premises.get(node) or {}).get("track")
            if t:
                tracks.add(t)
        if tracks & Q0_TRACKS and "LIFETIME3D" in tracks:
            problems.append(
                f"FW-2D-3D-COMPOSITION: {name} composes 2D and 3D tracks ({sorted(tracks)})")

    # FW-PRIZE-ISOLATION
    for name, claim in claims.items():
        tracks = set()
        for node in closure(g, name):
            t = (claims.get(node) or premises.get(node) or {}).get("track")
            if t:
                tracks.add(t)
        if "NUMBER_THEORY" in tracks and tracks & (Q0_TRACKS | {"LIFETIME3D"}):
            problems.append(
                f"FW-PRIZE-ISOLATION: {name} mixes the prize track with q0/3D ({sorted(tracks)})")

    # FW-NO-PRIZE-CLOSURE
    for name, claim in claims.items():
        if claim.get("track") == "NUMBER_THEORY" and claim.get("original_prize_closed") is not False:
            problems.append(
                f"FW-NO-PRIZE-CLOSURE: {name} does not carry original_prize_closed: false")

    # FW-DECIMAL-KILL
    for name, forbidden in (("Q0-C101-QUALITATIVE-RATE", "decimal"),
                            ("Q0-C104-THEOREM-B", "numerical")):
        claim = claims.get(name)
        if not claim:
            problems.append(f"FW-DECIMAL-KILL: {name} missing from the graph")
            continue
        if not any(forbidden in x for x in claim.get("forbidden_extrapolations", [])):
            problems.append(
                f"FW-DECIMAL-KILL: {name} does not forbid a {forbidden} constant")

    # A conditional claim must actually name at least one premise.
    for name, claim in claims.items():
        if claim.get("grade") == "CONDITIONAL" and not claim.get("depends_on"):
            problems.append(f"{name} is graded CONDITIONAL but names no premise")

    for p in problems:
        print(p)
    print(f"claims={len(claims)} premises={len(premises)} "
          f"firewalls={len(g['firewalls'])} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
