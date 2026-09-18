#!/usr/bin/env python3
"""Rank the open work lanes, so the repository can say what to work on next.

Reads `engine/lanes/*.json`, `claims/graph.json` and, when it exists,
`engine/carriers/MANIFEST.json`, and prints the lanes in a ranked order with the
ranking rule stated in the output.

What a rank is: a work-scheduling heuristic over four integers this repository
can compute — how many claims the claim graph says a lane blocks, whether the
source names a falsifier for it, whether its inputs are bound in this
repository, and how much code for it exists here. Nothing else.

What a rank is not: any kind of mathematical authority. Ranking a lane first
does not make its object more nearly closed; ranking it last does not make it
open. A lane's `status` is transcribed from the sources by `engine/lanes/`, and
`tools/lanes_check.py` refuses any lane that reads stronger than the same object
in `claims/graph.json`. The only thing that can change a mathematical status is
an operator decision under the protocols in `governance/`.

No floating-point arithmetic is used here: every ranking key is an integer or a
boolean, and no number in the output is a bound on anything.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANES = os.path.join(ROOT, "engine", "lanes")
GRAPH = os.path.join(ROOT, "claims", "graph.json")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")

REPO_STATE_RANK = {"running": 3, "partial": 2, "scaffolded": 1, "none": 0}

RANKING_RULE = [
    "1. claims blocked: the number of distinct claims in claims/graph.json that "
    "depend, transitively, on an object this lane blocks (more first).",
    "2. falsifier: a lane the sources give you a way to be proven wrong on ranks "
    "above one they do not (defined first).",
    "3. inputs bound here: a lane whose carrier inputs all resolve in "
    "engine/carriers/MANIFEST.json ranks above one whose inputs are unbound.",
    "4. repo_state: more code for the lane in this repository ranks first "
    "(running > partial > scaffolded > none). This is a code state. It is not a "
    "mathematical status and no verdict may be read from it.",
    "5. lane key, so the order is deterministic.",
]

NO_AUTHORITY = [
    "A ranking is a work-scheduling heuristic and carries NO MATHEMATICAL AUTHORITY WHATSOEVER.",
    "It promotes, closes, discharges and reclassifies nothing. Lane statuses are transcribed",
    "from the sources, never decided here. repo_state describes code in this repository only.",
    "The one thing that can change a mathematical status is an operator decision under the",
    "protocols in governance/. Running this program is not that, and neither is a green CI run.",
]


def load_lanes(lanes_dir: str) -> dict[str, dict]:
    lanes = {}
    for path in sorted(glob.glob(os.path.join(lanes_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            lane = json.load(f)
        lanes[lane.get("key", os.path.splitext(os.path.basename(path))[0])] = lane
    return lanes


def dependents(graph: dict) -> dict[str, set[str]]:
    """Reverse edges of the claim graph: object -> things that rest on it."""
    rev: dict[str, set[str]] = {}
    nodes = list(graph.get("claims", {}).items()) + list(graph.get("premises", {}).items())
    for name, node in nodes:
        for dep in (node.get("depends_on") or []) + (node.get("sub_obligations") or []):
            rev.setdefault(dep, set()).add(name)
    return rev


def blocked_claims(graph: dict, rev: dict[str, set[str]], blocks: list[str]) -> list[str]:
    """Claims reachable upward from the objects a lane blocks, including them."""
    seen: set[str] = set()
    stack = list(blocks)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(rev.get(node, ()))
    return sorted(n for n in seen if n in graph.get("claims", {}))


def carrier_ids(manifest_path: str) -> set[str] | None:
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    entries = man
    if isinstance(man, dict):
        for field in ("carriers", "entries", "members", "items"):
            if field in man:
                entries = man[field]
                break
    if isinstance(entries, dict):
        return set(entries)
    if isinstance(entries, list):
        ids = set()
        for e in entries:
            if isinstance(e, str):
                ids.add(e)
            elif isinstance(e, dict):
                for field in ("carrier_id", "id", "carrier"):
                    if field in e:
                        ids.add(e[field])
                        break
        return ids
    return set()


def assess(lanes: dict[str, dict], graph: dict, manifest_path: str) -> list[dict]:
    rev = dependents(graph)
    carriers = carrier_ids(manifest_path)
    rows = []
    for key, lane in lanes.items():
        blocks = lane.get("blocks") or []
        claims = blocked_claims(graph, rev, blocks)
        inputs = lane.get("inputs") or []
        if carriers is None:
            bound = False
            bound_note = "engine/carriers/MANIFEST.json absent; no carrier input is bound here"
        elif not inputs:
            bound = False
            bound_note = "lane declares no carrier inputs"
        else:
            missing = [c for c in inputs if c not in carriers]
            bound = not missing
            bound_note = "all inputs bound" if bound else f"unbound: {missing}"
        rows.append({
            "key": key,
            "title": lane.get("title"),
            "status": lane.get("status"),
            "status_is_transcribed_not_decided": True,
            "blocks": blocks,
            "blocked_claims": claims,
            "blocked_claim_count": len(claims),
            "falsifier": lane.get("falsifier"),
            "falsifier_defined": lane.get("falsifier") is not None,
            "inputs": inputs,
            "inputs_bound_here": bound,
            "inputs_note": bound_note,
            "repo_state": lane.get("repo_state"),
            "repo_state_is_not_a_mathematical_status": True,
            "next_exact_action": lane.get("next_exact_action"),
            "does_not_establish": lane.get("does_not_establish"),
            "artifacts": lane.get("artifacts") or [],
        })
    rows.sort(key=lambda r: (-r["blocked_claim_count"], not r["falsifier_defined"],
                             not r["inputs_bound_here"],
                             -REPO_STATE_RANK.get(r["repo_state"], -1), r["key"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def wrap(text: str, width: int, indent: str) -> list[str]:
    words, lines, cur = str(text).split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(indent + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(indent + cur)
    return lines


def print_banner() -> None:
    print("=" * 78)
    print("q0 / SIDE24 work dispatcher — engine/next_action.py")
    print("=" * 78)
    print("RANKING RULE (applied in this order):")
    for line in RANKING_RULE:
        parts = wrap(line, 72, "")
        print("  " + parts[0])
        for out in parts[1:]:
            print("     " + out)
    print()
    for line in NO_AUTHORITY:
        print(line)
    print("=" * 78)


def print_ranked(rows: list[dict]) -> None:
    print_banner()
    print(f"{'#':>2}  {'LANE':<4} {'BLOCKS':>6} {'FALSIF':>6} {'INPUTS':>6}  "
          f"{'REPO_STATE':<11} {'STATUS (transcribed)':<22} TITLE")
    print("-" * 78)
    for r in rows:
        print(f"{r['rank']:>2}  {r['key']:<4} {r['blocked_claim_count']:>6} "
              f"{'yes' if r['falsifier_defined'] else 'no':>6} "
              f"{'yes' if r['inputs_bound_here'] else 'no':>6}  "
              f"{r['repo_state']:<11} {str(r['status'] or '(none in source)'):<22} "
              f"{r['title']}")
    print("-" * 78)
    print("BLOCKS = claims in claims/graph.json that rest, transitively, on this lane.")
    print("STATUS = transcribed from the sources; null means the source assigns none.")
    print("Use --lane KEY for one lane in full, --json for machine consumption.")


def print_lane(row: dict, lane: dict) -> None:
    print_banner()
    print(f"LANE {row['key']} — {row['title']}")
    src = lane.get("source_section", {})
    print(f"  source section : {src.get('document')}  {src.get('heading')}")
    print(f"  status         : {row['status'] or '(the source assigns none)'}"
          f"   [transcribed from source, not decided here]")
    if lane.get("status_absent_reason"):
        for out in wrap(lane["status_absent_reason"], 72, "                   "):
            print(out)
    print(f"  blocks         : {', '.join(row['blocks']) or '(none)'}")
    if lane.get("blocks_note"):
        for out in wrap(lane["blocks_note"], 72, "                   "):
            print(out)
    print(f"  blocked claims : {', '.join(row['blocked_claims']) or '(none)'}")
    print(f"  falsifier      : {row['falsifier'] or '(the source names none)'}")
    print(f"  inputs         : {', '.join(row['inputs']) or '(none)'}  — {row['inputs_note']}")
    print(f"  repo_state     : {row['repo_state']}  [code in this repository; NOT a status]")
    for out in wrap(lane.get("repo_state_note", ""), 72, "                   "):
        print(out)
    print("  next exact action (quoted from the source):")
    for out in wrap(row["next_exact_action"], 72, "      "):
        print(out)
    if lane.get("not_a_substitute"):
        print("  not a substitute (the source's own caveats):")
        for item in lane["not_a_substitute"]:
            for out in wrap(f"- {item.get('quote')}  [{item.get('source')}]", 72, "      "):
                print(out)
    print("  artifacts      : " + (", ".join(row["artifacts"]) or "(none in this repository)"))
    print("  DOES NOT ESTABLISH:")
    for out in wrap(row["does_not_establish"], 72, "      "):
        print(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", default=LANES)
    ap.add_argument("--graph", default=GRAPH)
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--lane", metavar="KEY", help="detail for one lane")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args(argv)

    lanes = load_lanes(args.lanes)
    with open(args.graph, encoding="utf-8") as f:
        graph = json.load(f)
    rows = assess(lanes, graph, args.manifest)

    if args.lane:
        key = args.lane
        if key not in lanes:
            print(f"no such lane: {key!r} (have {', '.join(sorted(lanes))})", file=sys.stderr)
            return 2
        row = next(r for r in rows if r["key"] == key)
        if args.json:
            print(json.dumps({"ranking_rule": RANKING_RULE, "no_authority": NO_AUTHORITY,
                              "lane": row, "lane_file": lanes[key]}, indent=2, ensure_ascii=False))
        else:
            print_lane(row, lanes[key])
        return 0

    if args.json:
        print(json.dumps({"ranking_rule": RANKING_RULE, "no_authority": NO_AUTHORITY,
                          "generated_by": "engine/next_action.py",
                          "ranking_carries_no_mathematical_authority": True,
                          "lanes": rows}, indent=2, ensure_ascii=False))
    else:
        print_ranked(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
