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
  * attach a numerical constant to the qualitative rate or to Theorem B;
  * restore an unconditional grade on a claim whose register carries a
    retraction (Theorem B: GP-AUD-187 retracted the historical PROVEN-HERE
    label; "Do not cite ... historical PROVEN-HERE labels as current proof");
  * mark the RV-LM011 synthesis route satisfiable while a prerequisite is open,
    or count a same-provider technical pass as organizational independence;
  * let a receipt, a green test run, a reproduction or a carrier binding raise a
    grade or move a status;
  * let a floating-point computation, however precise, stand as a certification.

It checks the shape of the recorded dependencies. It does not verify any
mathematics, and it never changes a status. Every rule below can only REFUSE;
none of them marks anything satisfied, discharged, closed or certified.

Exit status is non-zero when any firewall is violated.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "claims", "graph.json")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")

OPEN_STATUSES = {"OPEN", "NOT_CLOSED"}
UNCONDITIONAL_GRADES = {"LIVE_ROOT_THEOREM", "FROZEN_CERTIFICATE", "RATIFIED_3D_ONLY"}
# P0.2 is a distinct lane from the q0 pair-Palm law; it is grouped with the 2D
# tracks here so that the 2D/3D and prize firewalls cover it too. Grouping only
# ever adds refusals.
Q0_TRACKS = {"UPPER2D", "LOWER2D", "P02_ADJACENCY"}

# ---------------------------------------------------------------- FW-LM011 --
# Technical statuses the registers use for a route that has passed. Transcribed
# vocabulary, not a judgement: anything outside this set is simply not a pass.
PASSING_TECHNICAL_STATUSES = {"PASS_TECHNICAL", "APPROVE", "CLOSED", "TERMINAL"}
# Independent-review states that record a verdict actually obtained.
SATISFIED_INDEPENDENCE = {"SATISFIED", "COMPLETED"}
SYNTHESIS_ROUTE = "RV-LM011-MAIN"

# ------------------------------------- FW-NO-RECEIPT-PROMOTION / FW-FLOAT ----
EVIDENCE_KINDS = {
    "proof_body", "frozen_certificate_body", "certificate_set", "register_row",
    "review_record", "exact_rational_module",
    # the four kinds that establish nothing mathematical:
    "receipt", "test", "carrier_binding", "reproduction",
}
NON_ESTABLISHING_EVIDENCE_KINDS = {"receipt", "test", "carrier_binding", "reproduction"}
DISCHARGING_STATUSES = {"CLOSED", "DISCHARGED", "PROMOTED", "SATISFIED", "CERTIFIED"}
CERTIFYING_GRADES = {"FROZEN_CERTIFICATE", "CERTIFIED_RUNG", "AUTHOR_SIDE_CERTIFIED"}

# Ordering used for ONE purpose: refusing a record that claims more than its
# evidence can carry. It is not a mathematical hierarchy and it grades nothing.
# A grade absent from this table is a failure, never a default, so that a new
# grade must be placed explicitly rather than slipping in unranked.
#   0 open / refuted / under review   1 conditional, or work recorded
#   2 author-side argument present    3 certified at a scope   4 root / frozen
GRADE_STRENGTH = {
    "OPEN": 0, "REFUTED_AS_WRITTEN": 0, "NEEDS_RECONCILIATION": 0,
    "AMEND_REQUIRED": 0, "PROPOSED": 0,
    "CONDITIONAL": 1, "AUTHOR_SIDE_PARTIAL": 1, "PASS_TECHNICAL": 1,
    "READY": 1, "AMEND": 1,   # review-queue words: a route ready for or needing review; work recorded, nothing discharged
    "RETRACTED_TO_CANDIDATE": 1,   # an unconditional label withdrawn by audit; what survives is conditional
    "AUTHOR_SIDE_PROOF_PRESENT": 2,
    "AUTHOR_SIDE_COMPLETE_ARGUMENTS_WITH_EXACT_FINITE_COMPANIONS": 2,
    "ACCEPTED_AT_REVIEW_SCOPE": 3, "AUTHOR_SIDE_CERTIFIED": 3, "CERTIFIED_RUNG": 3,
    "LIVE_ROOT_THEOREM": 4, "FROZEN_CERTIFICATE": 4, "RATIFIED_3D_ONLY": 4,
}
CONDITIONAL_STRENGTH = GRADE_STRENGTH["CONDITIONAL"]

# Arithmetic vocabulary. Matching is on substrings so that a carrier manifest's
# own prose ("mpmath binary floating point at mp.dps = 100") classifies the same
# way as the graph's token ("mpmath_float").
FLOAT_ARITHMETIC_TOKENS = ("float", "mpmath", "numpy", "double")
EXACT_ARITHMETIC_TOKENS = ("fraction", "rational", "interval", "arb", "decimal", "exact")


def is_float_arithmetic(arithmetic) -> bool:
    """True when the declared arithmetic is binary floating point.

    An exact/interval token anywhere wins: `engine/rn_engine/BINDING.json`
    records arithmetic as a sentence, and a sentence that mentions intervals or
    `fractions.Fraction` is not being described as plain float. Unknown or
    missing arithmetic is NOT reported as float — it is simply unchecked, which
    is why an unrecorded arithmetic is never a pass either.
    """
    if not isinstance(arithmetic, str):
        return False
    s = arithmetic.lower()
    if any(t in s for t in EXACT_ARITHMETIC_TOKENS):
        return False
    return any(t in s for t in FLOAT_ARITHMETIC_TOKENS)


def load(path: str | None = None) -> dict:
    # Resolve the module global at call time so tests (and a --graph override)
    # can point the checker at a different graph file.
    with open(path or GRAPH, encoding="utf-8") as f:
        return json.load(f)


def load_carrier_manifest(path: str | None = None) -> dict:
    """{carrier_id: record} from `engine/carriers/MANIFEST.json`, or {}.

    That manifest is owned by another part of this repository and may not exist.
    When it is absent, unreadable or shaped in a way this function does not
    recognise, the FW-FLOAT-NOT-CERTIFIED check falls back to the graph's own
    evidence records and skips the manifest cleanly. Skipping it can only lose a
    refusal that the graph's own record would have to state anyway; it can never
    manufacture a pass, because an evidence record that declares float
    arithmetic is refused on the graph's own fields.
    """
    p = path or MANIFEST
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            man = json.load(f)
    except (OSError, ValueError):
        return {}
    entries = man
    if isinstance(man, dict):
        for field in ("carriers", "entries", "members", "items"):
            if field in man:
                entries = man[field]
                break
    out: dict[str, dict] = {}
    if isinstance(entries, dict):
        for k, v in entries.items():
            if isinstance(v, dict):
                out[k] = v
    elif isinstance(entries, list):
        for e in entries:
            if not isinstance(e, dict):
                continue
            for field in ("carrier_id", "id", "carrier"):
                if isinstance(e.get(field), str):
                    out[e[field]] = e
                    break
    return out


def evidence_arithmetic(ev: dict, manifest: dict) -> tuple[object, object, str]:
    """(arithmetic, certifying, where) for one evidence record.

    The carrier manifest wins over the graph for any evidence naming a
    `carrier_id` it lists: the carrier's own record is what the run actually
    used.
    """
    arithmetic, certifying, where = ev.get("arithmetic"), ev.get("certifying"), "graph"
    carrier = manifest.get(ev.get("carrier_id")) if ev.get("carrier_id") else None
    if carrier:
        if carrier.get("arithmetic") is not None:
            arithmetic, where = carrier["arithmetic"], "engine/carriers/MANIFEST.json"
        if carrier.get("certifying") is not None:
            certifying, where = carrier["certifying"], "engine/carriers/MANIFEST.json"
    return arithmetic, certifying, where


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
    global GRAPH, MANIFEST
    argv = argv if argv is not None else sys.argv[1:]
    # Parsed by hand and strictly: an unrecognised flag is an error rather than
    # a silent fall-back to the committed graph. A default-argument bug once
    # made the mutation tests re-check the good graph and pass regardless of the
    # mutation, so the override path is kept explicit and greedy.
    rest = list(argv)
    while rest:
        flag = rest.pop(0)
        if flag in ("--graph", "--manifest"):
            if not rest:
                print(f"{flag} needs a path")
                return 2
            if flag == "--graph":
                GRAPH = rest.pop(0)
            else:
                MANIFEST = rest.pop(0)
        else:
            print(f"unknown argument {flag!r}; usage: claims_check.py "
                  f"[--graph PATH] [--manifest PATH]")
            return 2
    g = load()
    manifest = load_carrier_manifest()
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

    # FW-RETRACTED-NOT-UNCONDITIONAL. A claim that carries a retraction record,
    # or whose transcribed register status says RETRACTED, cannot carry an
    # unconditional grade: the historical label is exactly what the audit
    # withdrew, and a graph edit must not quietly put it back.
    for name, claim in claims.items():
        retracted = claim.get("retraction") is not None or \
            "RETRACTED" in str(claim.get("register_status", "")).upper()
        if not retracted:
            continue
        if claim.get("retraction") is not None and not claim.get("register_status"):
            problems.append(f"FW-RETRACTED-NOT-UNCONDITIONAL: {name} carries a retraction record "
                            f"but no register_status transcribing the register's current word")
        if claim.get("grade") in UNCONDITIONAL_GRADES:
            problems.append(
                f"FW-RETRACTED-NOT-UNCONDITIONAL: {name} is graded {claim['grade']} but its register "
                f"status is a retraction ({str(claim.get('register_status'))[:60]}...); the historical "
                f"unconditional label is not current proof")

    # A conditional claim must actually name at least one premise.
    for name, claim in claims.items():
        if claim.get("grade") == "CONDITIONAL" and not claim.get("depends_on"):
            problems.append(f"{name} is graded CONDITIONAL but names no premise")

    # A record's `technical_status`, where it has one, is the same transcribed
    # word as its `grade`. Two fields that can drift are two chances to promote.
    for name, claim in claims.items():
        ts = claim.get("technical_status")
        if ts is not None and ts != claim.get("grade"):
            problems.append(
                f"{name}: technical_status {ts!r} and grade {claim.get('grade')!r} disagree; "
                f"both transcribe the same register word")

    # FW-LM011-PRECONDITION
    def unsatisfied_reason(dep: str) -> str | None:
        """Why `dep` does not discharge a precondition, or None if it does.

        This function can only withhold satisfaction. It never records one.
        """
        if dep in premises:
            p = premises[dep]
            # Both layers must record a discharge. One column alone is the
            # register note running ahead of the frozen body, which is exactly
            # the disagreement this graph refuses to collapse.
            for column in ("status_frozen_v2_2", "status_register_note"):
                if p.get(column) not in DISCHARGING_STATUSES:
                    return f"premise {column} = {p.get(column)!r}"
            return None
        node = claims.get(dep)
        if node is None:
            return "not in the graph"
        status = node.get("technical_status", node.get("grade"))
        if status not in PASSING_TECHNICAL_STATUSES:
            return f"technical status {status!r} is not a pass"
        if node.get("requires_independent_verdict"):
            state = node.get("independent_review_state")
            if state not in SATISFIED_INDEPENDENCE:
                return f"independent review {state!r}"
            if not node.get("independence_credit"):
                return ("technical pass at ZERO organizational independence credit; a "
                        "same-provider pass does not discharge an independence-requiring gate")
        return None

    synthesis = claims.get(SYNTHESIS_ROUTE)
    if synthesis is None:
        problems.append(
            f"FW-LM011-PRECONDITION: {SYNTHESIS_ROUTE} is missing from the graph")
    elif not synthesis.get("precondition_routes"):
        problems.append(
            f"FW-LM011-PRECONDITION: {SYNTHESIS_ROUTE} names no precondition_routes; the "
            f"register's 'LM003, LM004-v1.1, LM006, LM009, LM010-v1.1, LM012-v1.1 and "
            f"LM013 Carrier-B-v1.1 plus joint stack FIRST' must be wired, not remembered")

    for name, claim in claims.items():
        routes = claim.get("precondition_routes")
        if not routes:
            continue
        declared = claim.get("depends_on") or []
        for route in routes:
            if route not in declared:
                problems.append(
                    f"FW-LM011-PRECONDITION: {name} names {route} as a precondition route "
                    f"but does not depend on it")
        if "synthesis_route_satisfiable" not in claim:
            problems.append(
                f"FW-LM011-PRECONDITION: {name} names precondition routes but does not "
                f"declare synthesis_route_satisfiable")
        if claim.get("synthesis_route_satisfiable") is True:
            for route in routes:
                reason = unsatisfied_reason(route)
                if reason:
                    problems.append(
                        f"FW-LM011-PRECONDITION: {name} is marked satisfiable while "
                        f"{route} is unsatisfied ({reason})")

    # FW-NO-RECEIPT-PROMOTION and FW-FLOAT-NOT-CERTIFIED
    for name, node in list(claims.items()) + list(premises.items()):
        evidence = node.get("evidence")
        if not evidence:
            continue
        is_claim = name in claims
        for i, ev in enumerate(evidence):
            if ev.get("kind") not in EVIDENCE_KINDS:
                problems.append(
                    f"{name}: evidence[{i}] kind {ev.get('kind')!r} is not in the evidence "
                    f"vocabulary {sorted(EVIDENCE_KINDS)}")
            arithmetic, certifying, where = evidence_arithmetic(ev, manifest)
            if certifying is True and is_float_arithmetic(arithmetic):
                problems.append(
                    f"FW-FLOAT-NOT-CERTIFIED: {name} evidence[{i}] declares certifying: true "
                    f"with arithmetic {arithmetic!r} (from {where}). High precision is not "
                    f"certification.")

        kinds = {ev.get("kind") for ev in evidence}
        if kinds and kinds <= NON_ESTABLISHING_EVIDENCE_KINDS:
            if is_claim:
                grade = node.get("grade")
                strength = GRADE_STRENGTH.get(grade)
                if strength is None:
                    problems.append(
                        f"FW-NO-RECEIPT-PROMOTION: {name} carries grade {grade!r}, which has no "
                        f"entry in GRADE_STRENGTH, so it cannot be compared against its evidence")
                elif strength > CONDITIONAL_STRENGTH:
                    problems.append(
                        f"FW-NO-RECEIPT-PROMOTION: {name} is graded {grade} (stronger than "
                        f"CONDITIONAL) on evidence of kinds {sorted(kinds)} only. A receipt, a "
                        f"green test run, a reproduction and a carrier binding are records that "
                        f"something ran, not that something is true.")
            else:
                for column in ("status_frozen_v2_2", "status_register_note"):
                    if node.get(column) in DISCHARGING_STATUSES:
                        problems.append(
                            f"FW-NO-RECEIPT-PROMOTION: premise {name} carries {column} = "
                            f"{node[column]} on evidence of kinds {sorted(kinds)} only. A "
                            f"receipt or a carrier binding discharges nothing.")

        if is_claim and node.get("grade") in CERTIFYING_GRADES:
            views = [evidence_arithmetic(ev, manifest) for ev in evidence]
            if views and all(is_float_arithmetic(a) for a, _c, _w in views):
                problems.append(
                    f"FW-FLOAT-NOT-CERTIFIED: {name} is graded {node['grade']} but every "
                    f"evidence record it cites is floating point "
                    f"({sorted({str(a) for a, _c, _w in views})}). A high-precision float "
                    f"computation is not a certified bound.")

    # FW-PROPOSED-LAYER-NOT-A-STATUS. The corpus transcribes some PROPOSED-tier
    # promotions verbatim, so a reader can see what a source proposed without
    # leaving this repository. `OBL-H5-ZBAND` carries
    # "OBL-H5-ZBAND: OPEN -> DISCHARGED (consumption grade)" beside two status
    # fields that both say OPEN, which is correct -- and until 2026-09-22 it was
    # correct by FIELD NAMING AND PROSE ALONE. Nothing here knew the field
    # existed. A word search for DISCHARGED in this graph finds a sentence a
    # source proposed and no operator granted, and the PR owner flagged exactly
    # that skim-trap on 2026-09-22: "do not promote from word search. Green CI
    # != discharge."
    #
    # Three rules, so the transcription can never become the status:
    #   (a) a proposed-layer transcription must name its source, and that source
    #       must record the tier and the absent authority;
    #   (b) the value the transition proposes must not appear in ANY status or
    #       grade field of that entry;
    #   (c) where the transition names a from-value, every status field of that
    #       entry must still carry it.
    # Plus a general rule: no status or grade field anywhere may hold a
    # TRANSITION at all. A status is a value, not an arrow.
    arrow = re.compile(r"\s*(?:->|\u2192)\s*")
    for kind, table in (("premise", premises), ("claim", claims)):
        for name, node in table.items():
            status_fields = {f: str(v) for f, v in node.items()
                             if ("status" in f or "grade" in f)
                             and isinstance(v, str)
                             and not f.startswith("proposed_layer")
                             and not f.endswith("_source")}
            for f, v in status_fields.items():
                if arrow.search(v):
                    problems.append(
                        f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} field {f!r} holds a "
                        f"transition ({v[:60]!r}). A status is a value, not an arrow; "
                        f"transcribe the transition in proposed_layer_verbatim instead")
            verbatim = node.get("proposed_layer_verbatim")
            if not verbatim:
                continue
            source = str(node.get("proposed_layer_source") or "")
            if not source:
                problems.append(
                    f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} transcribes a proposed-layer "
                    f"promotion and names no proposed_layer_source. An unsourced proposal is "
                    f"indistinguishable from an assertion")
            else:
                upper = source.upper()
                if "PROPOSED" not in upper:
                    problems.append(
                        f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} proposed_layer_source does "
                        f"not record the PROPOSED tier")
                if "AUTHORITY" not in upper:
                    problems.append(
                        f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} proposed_layer_source does "
                        f"not record the authority the proposal carries (none, unless an operator "
                        f"granted one)")
            parts = arrow.split(str(verbatim))
            if len(parts) < 2:
                problems.append(
                    f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} proposed_layer_verbatim names no "
                    f"transition ({str(verbatim)[:60]!r}); if it is not a proposed promotion it does "
                    f"not belong in this field")
                continue
            was = parts[0].split(":")[-1].strip().upper().split()
            proposes = parts[-1].strip().upper().split()
            proposed_token = proposes[0] if proposes else ""
            from_token = was[-1] if was else ""
            for f, v in status_fields.items():
                up = v.upper()
                if proposed_token and proposed_token in up.split():
                    problems.append(
                        f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} field {f!r} carries "
                        f"{proposed_token!r}, which is what proposed_layer_verbatim PROPOSES and no "
                        f"operator has granted. Only the operator applies the licensing predicate")
                if from_token and from_token not in up.split():
                    problems.append(
                        f"FW-PROPOSED-LAYER-NOT-A-STATUS: {kind} {name} proposed_layer_verbatim reads "
                        f"{from_token!r} -> {proposed_token!r} but field {f!r} is {v[:40]!r}. The "
                        f"pre-promotion value is what the sources record; a status that has moved "
                        f"off it needs its own transcribed source, not a proposal")

    for p in problems:
        print(p)
    print(f"claims={len(claims)} premises={len(premises)} "
          f"firewalls={len(g['firewalls'])} problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
