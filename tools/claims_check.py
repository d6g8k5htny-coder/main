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
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH = os.path.join(ROOT, "claims", "graph.json")
CLAIMS_README = os.path.join(ROOT, "claims", "README.md")
MANIFEST = os.path.join(ROOT, "engine", "carriers", "MANIFEST.json")
# The second carrier index, and the one the graph actually names. Until this
# existed the override below resolved nothing: the graph names exactly one
# carrier_id, RNENG-01, which is in BINDING and not in MANIFEST.
BINDING = os.path.join(ROOT, "engine", "rn_engine", "BINDING.json")

OPEN_STATUSES = {"OPEN", "NOT_CLOSED"}
UNCONDITIONAL_GRADES = {"LIVE_ROOT_THEOREM", "FROZEN_CERTIFICATE", "RATIFIED_3D_ONLY"}
# P0.2 is a distinct lane from the q0 pair-Palm law; it is grouped with the 2D
# tracks here so that the 2D/3D and prize firewalls cover it too. Grouping only
# ever adds refusals.
Q0_TRACKS = {"UPPER2D", "LOWER2D", "P02_ADJACENCY"}
# Every track a node may carry. Closed, because three firewalls read `track` as a
# membership test against a literal set -- FW-2D-3D-COMPOSITION,
# FW-PRIZE-ISOLATION and FW-NO-PRIZE-CLOSURE -- so an unrecognised value matches
# none of them and drops the node out of all three at once, silently. Nothing
# validated it: the graph happens to use exactly these five, and a typo or a
# rename would have been a quiet opt-out from rule 4.
TRACKS = Q0_TRACKS | {"LIFETIME3D", "NUMBER_THEORY"}

# Premise status vocabulary, closed for the same reason. FW-UNCONDITIONAL used to
# ask `status_frozen_v2_2 in {"OPEN", "NOT_CLOSED"}`, which is an open-word
# whitelist over ONE of the two columns, and it let two things through:
#
#   * NAMED_HYPOTHESIS read as discharged. A named hypothesis is by definition
#     not discharged, and two premises carry exactly that word.
#   * the register-note column was never read at all -- and the two-column shape
#     exists precisely because the columns disagree about four of five D1
#     premises. A premise open in its note and something else when frozen was
#     invisible.
#
# So the test is inverted: a premise is treated as discharged only when a column
# says so in a word listed here, and an unrecognised word is refused rather than
# assumed harmless. RESTATED and REFINEMENT are deliberately NOT in
# PREMISE_DISCHARGED: whether a restatement discharges the premise is a status
# question, and this file transcribes statuses rather than deciding them.
# The firewalls this file actually enforces. Three lists used to exist -- declared
# in graph.json, enforced here, documented in claims/README.md -- and nothing
# reconciled them, so claims/README.md documented eight where nine are enforced
# and said "24 claims" over a graph of twenty-six. A checker that cannot say
# which rules it applies cannot be audited, so the list is written down and
# compared against both the graph and the prose on every run.
ENFORCED_FIREWALLS = frozenset({
    "FW-UNCONDITIONAL", "FW-2D-3D-COMPOSITION", "FW-PRIZE-ISOLATION",
    "FW-NO-PRIZE-CLOSURE", "FW-DECIMAL-KILL", "FW-LM011-PRECONDITION",
    "FW-NO-RECEIPT-PROMOTION", "FW-FLOAT-NOT-CERTIFIED",
    "FW-RETRACTED-NOT-UNCONDITIONAL",
})

PREMISE_STATUS_COLUMNS = ("status_frozen_v2_2", "status_register_note")
PREMISE_DISCHARGED = {"CLOSED", "DISCHARGED", "PROMOTED", "SATISFIED", "CERTIFIED"}
PREMISE_UNDISCHARGED = {"OPEN", "NOT_CLOSED", "NAMED_HYPOTHESIS", "RESTATED",
                        "REFINEMENT", "PARTIAL", "REFUTED", "RETRACTED"}

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
# The third case, and it must be spelled rather than left blank: a review record
# or a register row performs no arithmetic at all. Nine evidence records say so.
NOT_APPLICABLE_ARITHMETIC_TOKENS = ("not_applicable", "not applicable")

FLOAT, EXACT, NOT_APPLICABLE = "float", "exact", "not_applicable"
UNRECOGNISED, AMBIGUOUS = "unrecognised", "ambiguous"


def classify_arithmetic(arithmetic) -> str:
    """One of FLOAT, EXACT, NOT_APPLICABLE, UNRECOGNISED, AMBIGUOUS.

    This was `is_float_arithmetic`, returning a bool, and the bool was a
    fail-open: an unlisted word ("IEEE 754", "binary64"), a missing field and a
    non-string all returned False, which the caller read as "not float" and so
    as nothing to refuse. The old docstring claimed "an unrecorded arithmetic is
    never a pass either". It was.

    AMBIGUOUS is the second fix and the less obvious one. The old rule was that
    an exact token anywhere wins, which a NEGATION defeats.
    `engine/rn_engine/BINDING.json` records RNENG-01 as

        "mpmath binary floating point at mp.dps = 100 ... no fractions.Fraction,
         no decimal.Decimal and no interval arithmetic occurs anywhere"

    -- genuinely float code, whose own denial mentions `fraction`, `decimal` and
    `interval`, so the old rule classified it EXACT. A string carrying tokens
    from both vocabularies is prose, not a classification, and the checker says
    so rather than picking one.
    """
    if not isinstance(arithmetic, str) or not arithmetic.strip():
        return UNRECOGNISED
    s = arithmetic.lower()
    float_hit = any(t in s for t in FLOAT_ARITHMETIC_TOKENS)
    exact_hit = any(t in s for t in EXACT_ARITHMETIC_TOKENS)
    if float_hit and exact_hit:
        return AMBIGUOUS
    if float_hit:
        return FLOAT
    if exact_hit:
        return EXACT
    if any(t in s for t in NOT_APPLICABLE_ARITHMETIC_TOKENS):
        return NOT_APPLICABLE
    return UNRECOGNISED


def is_float_arithmetic(arithmetic) -> bool:
    """Only the float question. Prefer the classifier: this still answers False
    for UNRECOGNISED and AMBIGUOUS, which is the fail-open it exists to close."""
    return classify_arithmetic(arithmetic) == FLOAT


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


def carrier_indexes(manifest_path: str | None = None,
                    binding_path: str | None = None) -> dict:
    """{carrier_id: (record, source label)} over BOTH carrier indexes.

    `tools/quarantine_check.py` and `tools/lanes_check.py` already resolve a
    carrier against both; this one read only `engine/carriers/MANIFEST.json`,
    and the graph names exactly one carrier_id, which lives in the other file.
    So the override that says "the carrier's own record is what the run actually
    used" resolved nothing at all: one lookup, one miss, every run.

    MANIFEST wins a duplicate id, and a duplicate that disagrees is reported by
    the caller rather than silently resolved.
    """
    out: dict[str, tuple[dict, str]] = {}
    for path, label in ((binding_path or BINDING, "engine/rn_engine/BINDING.json"),
                        (manifest_path or MANIFEST, "engine/carriers/MANIFEST.json")):
        for cid, rec in load_carrier_manifest(path).items():
            out[cid] = (rec, label)
    return out


def evidence_arithmetic(ev: dict, manifest: dict) -> tuple[object, object, str]:
    """(arithmetic, certifying, where) for one evidence record.

    A carrier index wins over the graph for any evidence naming a `carrier_id`
    it lists: the carrier's own record is what the run actually used. `manifest`
    maps a carrier_id to either a bare record or a (record, label) pair, so a
    caller that has only one index still works.
    """
    arithmetic, certifying, where = ev.get("arithmetic"), ev.get("certifying"), "graph"
    found = manifest.get(ev.get("carrier_id")) if ev.get("carrier_id") else None
    if isinstance(found, tuple):
        carrier, label = found
    else:
        carrier, label = found, "engine/carriers/MANIFEST.json"
    if carrier:
        if carrier.get("arithmetic") is not None:
            arithmetic, where = carrier["arithmetic"], label
        if carrier.get("certifying") is not None:
            certifying, where = carrier["certifying"], label
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
    global GRAPH, MANIFEST, BINDING, CLAIMS_README
    argv = argv if argv is not None else sys.argv[1:]
    # Parsed by hand and strictly: an unrecognised flag is an error rather than
    # a silent fall-back to the committed graph. A default-argument bug once
    # made the mutation tests re-check the good graph and pass regardless of the
    # mutation, so the override path is kept explicit and greedy.
    rest = list(argv)
    while rest:
        flag = rest.pop(0)
        if flag in ("--graph", "--manifest", "--binding", "--readme"):
            if not rest:
                print(f"{flag} needs a path")
                return 2
            if flag == "--graph":
                GRAPH = rest.pop(0)
            elif flag == "--manifest":
                MANIFEST = rest.pop(0)
            elif flag == "--binding":
                BINDING = rest.pop(0)
            else:
                CLAIMS_README = rest.pop(0)
        else:
            print(f"unknown argument {flag!r}; usage: claims_check.py "
                  f"[--graph PATH] [--manifest PATH] [--binding PATH] "
                  f"[--readme PATH]")
            return 2
    g = load()
    manifest = carrier_indexes()
    claims, premises = g["claims"], g["premises"]
    known = set(claims) | set(premises)
    problems: list[str] = []
    unreadable_arithmetic = 0

    # Declared / enforced / documented, reconciled three ways.
    declared = {f["id"] if isinstance(f, dict) else f for f in g.get("firewalls", [])}
    for fid in sorted(declared - ENFORCED_FIREWALLS):
        problems.append(
            f"{fid} is declared in claims/graph.json and is not in ENFORCED_FIREWALLS, "
            f"so nothing in this file applies it")
    for fid in sorted(ENFORCED_FIREWALLS - declared):
        problems.append(
            f"{fid} is enforced here and is not declared in claims/graph.json")
    # A missing prose document is a refusal, not a skip. `if os.path.isfile(...)`
    # would have been the next fail-open in this file: delete claims/README.md
    # and the entire reconciliation below disappears without a word.
    if not os.path.isfile(CLAIMS_README):
        problems.append(
            f"{os.path.relpath(CLAIMS_README, ROOT)} is absent, so the firewalls this "
            f"file enforces cannot be reconciled against the prose that documents them. "
            f"A missing document is not a pass.")
    else:
        with open(CLAIMS_README, encoding="utf-8") as handle:
            prose = handle.read()
        for fid in sorted(ENFORCED_FIREWALLS):
            if fid not in prose:
                problems.append(
                    f"{fid} is enforced and claims/README.md does not name it; a firewall "
                    f"nobody can read about is not a safeguard a reader can check")
        for want, what in ((f"{len(claims)} claims", "claim count"),
                           (f"{len(ENFORCED_FIREWALLS)} firewalls", "firewall count")):
            if want not in prose:
                problems.append(
                    f"claims/README.md does not state {want!r}; its {what} has drifted "
                    f"from the graph this checker reads")

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

    # Every premise status word must be in the closed vocabulary, on both
    # columns. An unrecognised word is not evidence of discharge and must not be
    # read as one.
    for name, p in premises.items():
        for col in PREMISE_STATUS_COLUMNS:
            v = p.get(col)
            if v is None:
                continue
            if v not in PREMISE_DISCHARGED and v not in PREMISE_UNDISCHARGED:
                problems.append(
                    f"FW-UNCONDITIONAL: {name} carries {col} {v!r}, which is in neither "
                    f"PREMISE_DISCHARGED nor PREMISE_UNDISCHARGED. A status word this file "
                    f"does not know is not a discharge; place it explicitly, transcribing "
                    f"the register rather than deciding anything.")

    # FW-UNCONDITIONAL
    for name, claim in claims.items():
        if claim.get("grade") not in UNCONDITIONAL_GRADES:
            continue
        for node in sorted(closure(g, name) - {name}):
            p = premises.get(node)
            if not p:
                continue
            for col in PREMISE_STATUS_COLUMNS:
                v = p.get(col)
                if v is not None and v not in PREMISE_DISCHARGED:
                    problems.append(
                        f"FW-UNCONDITIONAL: {name} is graded {claim['grade']} but rests on "
                        f"{node}, whose {col} is {v!r} and is not a discharge")

    # A node's `track` is read by three firewalls as a membership test, so an
    # unrecognised value is an opt-out from all three rather than an error.
    for name, node in list(claims.items()) + list(premises.items()):
        t = node.get("track")
        if t is None:
            problems.append(
                f"{name}: carries no `track`, so FW-2D-3D-COMPOSITION, "
                f"FW-PRIZE-ISOLATION and FW-NO-PRIZE-CLOSURE all skip it")
        elif t not in TRACKS:
            problems.append(
                f"{name}: track {t!r} is not one of {sorted(TRACKS)}. Three firewalls "
                f"test `track` by membership, so an unrecognised value drops this node "
                f"out of all three at once rather than failing anything")

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
            kind = classify_arithmetic(arithmetic)
            if kind in (UNRECOGNISED, AMBIGUOUS):
                unreadable_arithmetic += 1
            # The old test was `certifying is True and is_float_arithmetic(...)`,
            # and it was a fail-open three ways over: `is True` is an identity
            # test the JSON string "true" walks past; an unlisted word and a
            # missing field both answered "not float"; and a sentence denying
            # exact arithmetic classified as exact. So the rule is inverted.
            # A record that claims certification must SHOW exactness; every
            # other class is refused, by name. A record that claims none is
            # free to describe its arithmetic in prose, because nothing rests
            # on it.
            if not isinstance(certifying, bool):
                problems.append(
                    f"FW-FLOAT-NOT-CERTIFIED: {name} evidence[{i}] declares certifying "
                    f"{certifying!r} (from {where}), which is not a boolean. The test that "
                    f'guards this is an identity test, so the string "true" reads as '
                    f"not-certifying and refuses nothing.")
            elif certifying is True and kind != EXACT:
                why = {
                    FLOAT: "High precision is not certification.",
                    NOT_APPLICABLE: "A record that performs no arithmetic certifies nothing.",
                    UNRECOGNISED: ("It is in neither vocabulary, so it is not evidence of "
                                   "exactness; say which it is."),
                    AMBIGUOUS: ("It carries tokens from both vocabularies, which is prose "
                                "rather than a classification -- a sentence saying 'no "
                                "fractions.Fraction ... no interval arithmetic' describes "
                                "float code in words that used to classify it exact."),
                }[kind]
                problems.append(
                    f"FW-FLOAT-NOT-CERTIFIED: {name} evidence[{i}] declares certifying: true "
                    f"with arithmetic {arithmetic!r} (from {where}), which classifies as "
                    f"{kind}. {why}")

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

    for p in problems:
        print(p)
    print(f"claims={len(claims)} premises={len(premises)} "
          f"firewalls={len(g['firewalls'])} enforced={len(ENFORCED_FIREWALLS)} "
          f"evidence_arithmetic_unreadable={unreadable_arithmetic} "
          f"problems={len(problems)}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
