#!/usr/bin/env python3
"""Invariants for the work ledger in `engine/lanes/`.

`engine/lanes/<KEY>.json` turns the prose of `docs/OPEN_PROBLEMS.md` into work
items a dispatcher can rank. That is a scheduling convenience, and it introduces
exactly one danger worth building a checker around: a work ledger is an easy
place to quietly promote something. A lane that says `CLOSED` where the claim
graph says `OPEN`, or that reads its own `repo_state` as a mathematical status,
would launder a status change through a file nobody reviews as mathematics.

This tool refuses that. It checks, and exits non-zero on any failure:

  1. the lane set matches the section set of `docs/OPEN_PROBLEMS.md`, in both
     directions, with the section list parsed from the document's headings
     rather than hardcoded here;
  2. every id in `blocks` exists in `claims/graph.json`;
  3. every `carrier_id` in `inputs` exists in `engine/carriers/MANIFEST.json`
     when that manifest exists, and `inputs` is empty with a stated reason when
     it does not;
  4. every `status` value is one the registers (or the claim graph, which
     transcribes the same sources) actually use;
  5. **no lane declares a status stronger than the same object's status in
     `claims/graph.json`** — the firewall that stops the ledger from promoting
     anything;
  6. `repo_state` is a fixed code-state vocabulary, never used as a status, and
     absent from every structure in this repository that carries evidentiary
     meaning;
  7. lane D's per-route sub-items still match `registers/json/review_queue.json`
     verbatim.

It checks the shape of transcribed statements. It verifies no mathematics, it
grades nothing, and passing it is not evidence of anything.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Lane fields. A lane that carries an unlisted key fails: a work ledger that can
# grow fields silently can grow a status field silently.
REQUIRED = {
    "key", "title", "source_section", "blocks", "status", "status_source",
    "next_exact_action", "falsifier", "inputs", "inputs_note", "not_a_substitute",
    "repo_state", "repo_state_note", "artifacts", "does_not_establish",
}
OPTIONAL = {
    "obligation", "status_absent_reason", "blocks_note", "next_exact_action_note",
    "additional_source_directives", "register_next_decisive_action", "firewalls",
    "source_note", "sub_items", "sub_items_source",
}

REPO_STATES = ("none", "scaffolded", "partial", "running")

# Ordering used for ONE purpose: refusing a lane that claims more than the claim
# graph records for the same object. It is not a mathematical hierarchy, it
# grades nothing, and an unknown value is a failure rather than a default.
#   0 = the object is open, not closed, unreviewed, or a hypothesis carried
#   1 = work is recorded against the object; nothing is discharged
#   2 = the source records a discharge, closure or promotion
STATUS_STRENGTH = {
    "OPEN": 0, "NOT_CLOSED": 0, "NEEDS_RECONCILIATION": 0, "UNASSIGNED": 0,
    "HOLD": 0, "NAMED_HYPOTHESIS": 0, "REFUTED_AS_WRITTEN": 0, "CANNOT_VERIFY": 0,
    "OPEN-RESPONSE-REQUIRED": 0, "REPAIR-REQUIRED": 0,
    "PARTIAL": 1, "RESTATED": 1, "REFINEMENT": 1, "PROPOSED": 1, "AMEND": 1,
    "AMEND_REQUIRED": 1, "READY": 1, "PASS_TECHNICAL": 1, "CONDITIONAL": 1,
    "RETRACTED_TO_CANDIDATE": 1,
    "CERTIFIED_RUNG": 1, "ACCEPTED_AT_REVIEW_SCOPE": 1, "AUTHOR_SIDE_PARTIAL": 1,
    "AUTHOR_SIDE_CERTIFIED": 1, "AUTHOR_SIDE_PROOF_PRESENT": 1,
    "AUTHOR_SIDE_COMPLETE_ARGUMENTS_WITH_EXACT_FINITE_COMPANIONS": 1,
    "DISCHARGED": 2, "CLOSED": 2, "PROMOTED": 2, "LIVE_ROOT_THEOREM": 2,
    "FROZEN_CERTIFICATE": 2, "RATIFIED_3D_ONLY": 2,
}

HEADING = re.compile(r"^(#{2,3})\s+([A-Z][0-9]*)\.\s+(.+?)\s*$")
TOKEN = re.compile(r"^[A-Z][A-Z0-9_().|+-]*$")

# Files that carry evidentiary meaning. `repo_state` describes code, so it must
# never appear in any of them.
EVIDENTIARY_GLOBS = ("claims/graph.json", "registers/json/*.json", "reviews/records/*.json",
                     "quarantine/EXCLUSIONS.json", "governance/PROVENANCE.json")


def parse_sections(doc_path: str) -> dict[str, dict]:
    """Section keys of the open-problems document, parsed from its headings.

    The rule, applied to the document rather than to a hardcoded list: a level-2
    section that contains level-3 subsections is a container and is represented
    by those subsections; every other section is a lane of its own.
    """
    heads = []
    with open(doc_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            m = HEADING.match(line.rstrip("\n"))
            if m:
                heads.append({"level": len(m.group(1)), "key": m.group(2),
                              "title": m.group(3), "line": n, "heading": line.rstrip("\n")})
    container = set()
    for i, h in enumerate(heads):
        if h["level"] == 2:
            for nxt in heads[i + 1:]:
                if nxt["level"] == 2:
                    break
                if nxt["level"] == 3:
                    container.add(h["key"])
                    break
    return {h["key"]: h for h in heads if h["key"] not in container}


def register_status_vocabulary(registers_dir: str) -> set[str]:
    """Status tokens the exported registers actually use."""
    vocab: set[str] = set()
    for path in sorted(glob.glob(os.path.join(registers_dir, "*.json"))):
        with open(path, encoding="utf-8") as f:
            tab = json.load(f)
        if not isinstance(tab, dict) or "header" not in tab or "rows" not in tab:
            continue
        cols = [i for i, c in enumerate(tab["header"]) if "status" in str(c).lower()]
        for row in tab["rows"]:
            for i in cols:
                if i < len(row) and isinstance(row[i], str) and TOKEN.match(row[i].strip()):
                    vocab.add(row[i].strip())
    return vocab


def graph_status_vocabulary(graph: dict) -> set[str]:
    vocab: set[str] = set()
    for p in graph.get("premises", {}).values():
        for k in ("status_frozen_v2_2", "status_register_note"):
            if isinstance(p.get(k), str):
                vocab.add(p[k])
    for c in graph.get("claims", {}).values():
        if isinstance(c.get("grade"), str):
            vocab.add(c["grade"])
    return vocab


def graph_statuses_for(graph: dict, node_id: str) -> list[str]:
    """Every status the graph records for one object, as separate layers."""
    if node_id in graph.get("premises", {}):
        p = graph["premises"][node_id]
        return [p[k] for k in ("status_frozen_v2_2", "status_register_note") if k in p]
    if node_id in graph.get("claims", {}):
        c = graph["claims"][node_id]
        return [c["grade"]] if "grade" in c else []
    return []


def find_key(obj, key: str, path: str = "") -> list[str]:
    """Every location of `key` inside a nested structure."""
    hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{path}.{k}" if path else k
            if k == key:
                hits.append(here)
            hits += find_key(v, key, here)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits += find_key(v, key, f"{path}[{i}]")
    return hits


def manifest_carrier_ids(path: str) -> set[str] | None:
    """Carrier ids in engine/carriers/MANIFEST.json, or None if it does not exist.

    The manifest is owned elsewhere in this repository. Several plausible shapes
    are read; an unrecognised shape raises, because silently reading it as "no
    carriers" would turn invariant 3 into a no-op exactly when it matters.
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    entries = man
    if isinstance(man, dict):
        for field in ("carriers", "entries", "members", "items"):
            if field in man:
                entries = man[field]
                break
    if isinstance(entries, dict):
        return set(entries.keys())
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
                else:
                    raise ValueError(f"manifest entry without a carrier id: {sorted(e)[:6]}")
        return ids
    raise ValueError(f"unrecognised manifest shape: {type(entries).__name__}")


def binding_member_ids(path: str) -> set[str]:
    """Archive-member ids in engine/rn_engine/BINDING.json, or an empty set.

    A second index exists on purpose. MANIFEST.json records Drive files, each
    validated against its own inventory row; BINDING.json records files
    recovered from inside ZIP carriers, which have no inventory row and are
    validated against drive/source_map/Archive_Members.csv by
    engine/rn_engine/verify_recovery.py instead. A lane input may name either.
    Absence of the binding file is not an error — it simply resolves nothing.
    """
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        b = json.load(f)
    entries = b.get("carriers", b) if isinstance(b, dict) else b
    ids: set[str] = set()
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("carrier_id"):
            ids.add(e["carrier_id"])
    return ids


def check(lanes_dir: str, doc: str, graph_path: str, registers_dir: str,
          review_queue: str, manifest: str, repo_root: str,
          binding: str | None = None) -> list[str]:
    if binding is None:
        binding = os.path.join(ROOT, "engine", "rn_engine", "BINDING.json")
    problems: list[str] = []

    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)
    known_ids = set(graph.get("claims", {})) | set(graph.get("premises", {}))
    vocabulary = register_status_vocabulary(registers_dir) | graph_status_vocabulary(graph)

    sections = parse_sections(doc)
    lanes: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(lanes_dir, "*.json"))):
        key = os.path.splitext(os.path.basename(path))[0]
        with open(path, encoding="utf-8") as f:
            lanes[key] = json.load(f)

    # 1. lane set <-> section set, both directions
    for key in sorted(set(sections) - set(lanes)):
        problems.append(
            f"section {key} ({sections[key]['heading']!r}, {doc} line {sections[key]['line']}) "
            f"has no lane file {key}.json")
    for key in sorted(set(lanes) - set(sections)):
        problems.append(f"lane {key}.json has no section in {doc}")

    for key in sorted(set(lanes) & set(sections)):
        lane, sec = lanes[key], sections[key]
        where = f"lane {key}"

        # schema
        if lane.get("key") != key:
            problems.append(f"{where}: 'key' is {lane.get('key')!r}, filename says {key!r}")
        missing = REQUIRED - set(lane)
        if missing:
            problems.append(f"{where}: missing required fields {sorted(missing)}")
        unknown = set(lane) - REQUIRED - OPTIONAL
        if unknown:
            problems.append(f"{where}: unknown fields {sorted(unknown)}")
        if lane.get("source_section", {}).get("heading") != sec["heading"]:
            problems.append(
                f"{where}: source_section.heading does not match the document heading "
                f"({lane.get('source_section', {}).get('heading')!r} vs {sec['heading']!r})")
        if not str(lane.get("next_exact_action") or "").strip():
            problems.append(f"{where}: next_exact_action is empty")
        if not str(lane.get("does_not_establish") or "").strip():
            problems.append(f"{where}: does_not_establish is empty — every lane must state what "
                            f"it does not establish")
        for field in ("blocks", "inputs", "not_a_substitute", "artifacts"):
            if not isinstance(lane.get(field), list):
                problems.append(f"{where}: {field} must be a list")

        # 2. blocks ids exist in the claim graph
        for node_id in lane.get("blocks", []) or []:
            if node_id not in known_ids:
                problems.append(f"{where}: blocks unknown id {node_id!r} (not in {graph_path})")

        # 3. inputs against the carrier manifest
        try:
            carriers = manifest_carrier_ids(manifest)
        except ValueError as exc:
            problems.append(f"{manifest}: {exc}; update manifest_carrier_ids in this checker "
                            f"rather than letting the inputs invariant pass vacuously")
            carriers = set()
        if carriers is None:
            if lane.get("inputs"):
                problems.append(f"{where}: declares inputs {lane['inputs']} but {manifest} "
                                f"does not exist, so no carrier_id can be verified")
            if not str(lane.get("inputs_note") or "").strip():
                problems.append(f"{where}: inputs are unbound and inputs_note does not say why")
        else:
            members = binding_member_ids(binding)
            for cid in lane.get("inputs", []) or []:
                if cid in carriers or cid in members:
                    continue
                problems.append(
                    f"{where}: input carrier_id {cid!r} is in neither index — not a Drive "
                    f"file in {manifest} and not an archive member in {binding}"
                )

        # 4. status vocabulary
        status = lane.get("status")
        if status is None:
            if not str(lane.get("status_absent_reason") or "").strip():
                problems.append(f"{where}: status is null without a status_absent_reason. A "
                                f"missing status must be explained, never assumed.")
        elif status not in vocabulary:
            problems.append(f"{where}: status {status!r} is not a status the registers or the "
                            f"claim graph use")
        elif status not in STATUS_STRENGTH:
            problems.append(f"{where}: status {status!r} has no entry in STATUS_STRENGTH, so it "
                            f"cannot be compared against the claim graph")

        # 5. THE FIREWALL: no lane may be stronger than the graph
        if status in STATUS_STRENGTH:
            for node_id in lane.get("blocks", []) or []:
                layers = graph_statuses_for(graph, node_id)
                for layer in layers:
                    if layer not in STATUS_STRENGTH:
                        problems.append(f"{where}: claim-graph status {layer!r} on {node_id} has "
                                        f"no strength entry; cannot verify the firewall")
                bound = [STATUS_STRENGTH[s] for s in layers if s in STATUS_STRENGTH]
                if bound and STATUS_STRENGTH[status] > min(bound):
                    problems.append(
                        f"{where}: PROMOTION REFUSED — lane status {status!r} is stronger than "
                        f"{node_id} in {graph_path} (layers {layers}). A work ledger may not "
                        f"promote, close or discharge anything; only an operator decision under "
                        f"governance/ can change a mathematical status.")

        # 6. repo_state is a code state, never a status
        rs = lane.get("repo_state")
        if rs not in REPO_STATES:
            problems.append(f"{where}: repo_state {rs!r} is not one of {list(REPO_STATES)}")
        if status in REPO_STATES:
            problems.append(f"{where}: status {status!r} is a repo_state value; code state is "
                            f"not a mathematical status")
        hits = find_key(lane, "repo_state")
        if hits != ["repo_state"]:
            problems.append(f"{where}: repo_state must appear exactly once, at the top level of "
                            f"the lane; found {hits}")
        if not str(lane.get("repo_state_note") or "").strip():
            problems.append(f"{where}: repo_state_note is empty")
        for item in lane.get("sub_items", []) or []:
            for field in ("technical_status", "label", "status"):
                if isinstance(item, dict) and item.get(field) in REPO_STATES:
                    problems.append(f"{where}: sub-item {field} {item[field]!r} is a repo_state "
                                    f"value used as a status")

    # 6 (continued). repo_state absent from every evidentiary structure
    for pattern in EVIDENTIARY_GLOBS:
        for path in sorted(glob.glob(os.path.join(repo_root, pattern))):
            with open(path, encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue
            hits = find_key(data, "repo_state")
            if hits:
                rel = os.path.relpath(path, repo_root)
                problems.append(
                    f"{rel}: carries evidentiary meaning and must not contain repo_state "
                    f"(found at {hits}). repo_state describes code in this repository; it is "
                    f"not a mathematical status.")

    # 7. lane D sub-items still match the register verbatim
    for key, lane in sorted(lanes.items()):
        if lane.get("sub_items_source") != "registers/json/review_queue.json":
            continue
        with open(review_queue, encoding="utf-8") as f:
            rq = json.load(f)
        h = rq["header"]
        rows = {r[h.index("Review key")]: r for r in rq["rows"]}
        items = {i["review_key"]: i for i in lane.get("sub_items", [])}
        for k in sorted(set(rows) - set(items)):
            problems.append(f"lane {key}: review route {k} is in the register and not in the lane")
        for k in sorted(set(items) - set(rows)):
            problems.append(f"lane {key}: sub-item {k} is not a route in {review_queue}")
        for k in sorted(set(rows) & set(items)):
            for field, col in (("technical_status", "Technical status"),
                               ("exact_object", "Exact object"),
                               ("next_action", "Next action"),
                               ("aging_action", "Aging action"),
                               ("independence_status", "Independence status")):
                want = rows[k][h.index(col)]
                got = items[k].get(field)
                if got != want:
                    problems.append(
                        f"lane {key}: sub-item {k} {field} is {got!r}; the register says {want!r}. "
                        f"Statuses are transcribed, never edited here.")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--lanes", default=os.path.join(ROOT, "engine", "lanes"))
    ap.add_argument("--doc", default=os.path.join(ROOT, "docs", "OPEN_PROBLEMS.md"))
    ap.add_argument("--graph", default=os.path.join(ROOT, "claims", "graph.json"))
    ap.add_argument("--registers", default=os.path.join(ROOT, "registers", "json"))
    ap.add_argument("--review-queue",
                    default=os.path.join(ROOT, "registers", "json", "review_queue.json"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "engine", "carriers", "MANIFEST.json"))
    ap.add_argument("--binding", default=os.path.join(ROOT, "engine", "rn_engine", "BINDING.json"),
                    help="archive-member index; a lane input may resolve here instead of the manifest")
    ap.add_argument("--repo-root", default=ROOT,
                    help="root scanned for evidentiary structures that must not carry repo_state")
    args = ap.parse_args(argv)

    problems = check(args.lanes, args.doc, args.graph, args.registers,
                     args.review_queue, args.manifest, args.repo_root,
                     binding=args.binding)
    if problems:
        print(f"lanes_check: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    n = len(glob.glob(os.path.join(args.lanes, "*.json")))
    print(f"lanes_check: OK — {n} lanes, one per section of {os.path.relpath(args.doc, ROOT)}")
    print("A pass here is a structural check on transcription. It verifies no mathematics, "
          "grades nothing, and is not evidence.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
