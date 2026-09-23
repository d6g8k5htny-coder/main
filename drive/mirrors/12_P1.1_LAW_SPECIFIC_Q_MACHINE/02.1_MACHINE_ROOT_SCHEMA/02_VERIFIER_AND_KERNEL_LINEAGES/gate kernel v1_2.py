#!/usr/bin/env python3
"""
GATE KERNEL v1.2 — executable enforcement of the Master Gate File (v1.1) SHELL,
with graph validation, JSON (de)serialization, and a minimal admission registry.

================================  SCOPE (read first)  ========================
Unchanged from v1.1 and non-negotiable: this kernel checks the mechanical SHELL
of every gate — tag presence/compatibility, graph reachability, dependability of
every cited node (including bridge nodes), hash-drift. It returns the 13-gate
vector + constitutional check + hash-integrity + an effective verdict.

It DOES NOT and CANNOT check the CORE of any gate — whether a tagged claim is
TRUE, whether a cited change-of-measure is VALID, whether a failure-region list
is COMPLETE. Per Master File §0.5 the kernel certifies tag-CONSISTENCY, not
tag-to-content FIDELITY. Fidelity is produced only by the §7 disciplines
(adversarial reimplementation, separation of duties). A PASS means "consistent
under the declared tags," never "correct." The wrong-but-named-witness failure
(a false but properly-typed, sufficiently-graded Independence node) is invisible
here by design; §3.8 forces it into the §7 verification queue by requiring it to
be graded, but the kernel cannot adjudicate it.

NEW IN v1.2
  - g_model defined with the other gates; GATES built once (no post-hoc patch).
  - validate_graph(): missing targets, illegal cycles, duplicate content hashes,
    inconsistent status/grade — a hard pre-pass before gating.
  - JSON serialize/deserialize for claims and whole graphs (registry files).
  - Registry: stores claims by content hash, admits leaf-up, and REFUSES any
    claim whose shell verdict is not PASS or an explicit PROVISIONAL (ceiling
    recorded). Persists to / loads from JSON.

DELIBERATELY LEFT AS-IS (per review; both defensible)
  - DOMAIN skips a support edge that declares no parameter_domain (shell choice;
    a `strict_domain` pass could tighten it, but non-declaration != violation).
  - COVERAGE / ENDPOINT do only the mechanical part; their large cores
    (is the region list complete? are these the true extrema?) remain §7 work.
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# ----------------------------------------------------------------------------
# §3.1  Grade ladder and the three dependability tiers
# ----------------------------------------------------------------------------
class Grade(str, Enum):
    PRIMITIVE = "Primitive"
    PROVEN = "Proven"
    PROVEN_MODULO = "Proven-Modulo"
    DERIVED = "Derived"
    LITERATURE = "Literature-Supported"
    PLAUSIBLE = "Plausible"
    CONJECTURE = "Conjecture"
    OPEN = "Open"
    KILLED = "Killed"


_FULLY = {Grade.PRIMITIVE, Grade.PROVEN, Grade.PROVEN_MODULO, Grade.LITERATURE}


def dependability_level(g: Grade) -> int:
    if g in _FULLY:
        return 3
    if g == Grade.DERIVED:
        return 2
    if g == Grade.PLAUSIBLE:
        return 1
    if g in (Grade.CONJECTURE, Grade.OPEN):
        return 0
    return -1  # Killed


def required_level(parent: Grade) -> int:
    if parent in _FULLY:
        return 3
    if parent == Grade.DERIVED:
        return 2
    if parent == Grade.PLAUSIBLE:
        return 1
    return 0


def edge_ok(dep_grade: Grade, parent_grade: Grade, conditional: bool) -> bool:
    if dep_grade == Grade.KILLED:
        return False
    if conditional and parent_grade == Grade.PROVEN_MODULO:
        return True
    return dependability_level(dep_grade) >= required_level(parent_grade)


# ----------------------------------------------------------------------------
# Verdicts
# ----------------------------------------------------------------------------
class V(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PROVISIONAL = "PROVISIONAL"
    NA = "n/a"


# ----------------------------------------------------------------------------
# Data model — the §2 proof object
# ----------------------------------------------------------------------------
class EdgeKind(str, Enum):
    SUPPORT = "support"
    CHANGE_OF_MEASURE = "change_of_measure"
    COMPOSITION_WITNESS = "composition_witness"
    MARK_TRANSFER = "mark_transfer"
    TRANSFER_CERTIFICATE = "transfer_certificate"
    HEURISTIC_BRIDGE = "heuristic_bridge"


@dataclass
class Edge:
    target: str
    kind: EdgeKind = EdgeKind.SUPPORT
    pinned_hash: str = ""
    conditional: bool = False
    scope_covers: bool = True

    def to_dict(self) -> dict:
        return {"target": self.target, "kind": self.kind.value,
                "pinned_hash": self.pinned_hash, "conditional": self.conditional,
                "scope_covers": self.scope_covers}

    @staticmethod
    def from_dict(d: dict) -> "Edge":
        return Edge(target=d["target"], kind=EdgeKind(d.get("kind", "support")),
                    pinned_hash=d.get("pinned_hash", ""),
                    conditional=d.get("conditional", False),
                    scope_covers=d.get("scope_covers", True))


@dataclass
class Claim:
    id: str
    statement: str
    grade: Grade
    content_hash: str
    status: str = "live"

    measure_tag: str = ""
    model_tag: str = ""
    parameter_domain: dict = field(default_factory=dict)     # param -> (lo, hi)
    required_marks: set = field(default_factory=set)
    supplied_marks: set = field(default_factory=set)

    direction: str = "none"
    uncertainty_side: str = "none"

    measured_values: list = field(default_factory=list)      # (name, value, scale_tag)

    is_combination: bool = False

    finite_range_bound: bool = False
    registered_endpoints: list = field(default_factory=list) # (point, value, satisfies)

    failure_regions: dict = field(default_factory=dict)      # region -> [detector ids]
    global_suppression_argument: bool = False

    verification: str = "none"                               # single_exact|agreement_based|none
    error_independence_arg: bool = False
    external_crosscheck: bool = False

    heuristic_step: bool = False
    closed_core: bool = False

    edges: list = field(default_factory=list)

    # -- serialization -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.id, "statement": self.statement, "grade": self.grade.value,
            "content_hash": self.content_hash, "status": self.status,
            "measure_tag": self.measure_tag, "model_tag": self.model_tag,
            "parameter_domain": {k: list(v) for k, v in self.parameter_domain.items()},
            "required_marks": sorted(self.required_marks),
            "supplied_marks": sorted(self.supplied_marks),
            "direction": self.direction, "uncertainty_side": self.uncertainty_side,
            "measured_values": [list(m) for m in self.measured_values],
            "is_combination": self.is_combination,
            "finite_range_bound": self.finite_range_bound,
            "registered_endpoints": [list(e) for e in self.registered_endpoints],
            "failure_regions": self.failure_regions,
            "global_suppression_argument": self.global_suppression_argument,
            "verification": self.verification,
            "error_independence_arg": self.error_independence_arg,
            "external_crosscheck": self.external_crosscheck,
            "heuristic_step": self.heuristic_step, "closed_core": self.closed_core,
            "edges": [e.to_dict() for e in self.edges],
        }

    @staticmethod
    def from_dict(d: dict) -> "Claim":
        return Claim(
            id=d["id"], statement=d.get("statement", ""),
            grade=Grade(d["grade"]), content_hash=d["content_hash"],
            status=d.get("status", "live"),
            measure_tag=d.get("measure_tag", ""), model_tag=d.get("model_tag", ""),
            parameter_domain={k: tuple(v) for k, v in d.get("parameter_domain", {}).items()},
            required_marks=set(d.get("required_marks", [])),
            supplied_marks=set(d.get("supplied_marks", [])),
            direction=d.get("direction", "none"),
            uncertainty_side=d.get("uncertainty_side", "none"),
            measured_values=[tuple(m) for m in d.get("measured_values", [])],
            is_combination=d.get("is_combination", False),
            finite_range_bound=d.get("finite_range_bound", False),
            registered_endpoints=[tuple(e) for e in d.get("registered_endpoints", [])],
            failure_regions=d.get("failure_regions", {}),
            global_suppression_argument=d.get("global_suppression_argument", False),
            verification=d.get("verification", "none"),
            error_independence_arg=d.get("error_independence_arg", False),
            external_crosscheck=d.get("external_crosscheck", False),
            heuristic_step=d.get("heuristic_step", False),
            closed_core=d.get("closed_core", False),
            edges=[Edge.from_dict(e) for e in d.get("edges", [])],
        )


Graph = dict  # id -> Claim


# ----------------------------------------------------------------------------
# graph helpers  (reachability is iterative + seen-guarded => cycle-terminating)
# ----------------------------------------------------------------------------
def _reachable(start: str, g: Graph) -> set:
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        node = g.get(cur)
        if not node:
            continue
        for e in node.edges:
            if e.target not in seen and e.target in g:
                seen.add(e.target)
                stack.append(e.target)
    return seen


def _bridges(c: Claim, kind: EdgeKind) -> list:
    return [e for e in c.edges if e.kind == kind]


def _bridge_sufficient(c: Claim, kind: EdgeKind, g: Graph) -> bool:
    for e in _bridges(c, kind):
        node = g.get(e.target)
        if node and edge_ok(node.grade, c.grade, conditional=False):
            if kind != EdgeKind.TRANSFER_CERTIFICATE or e.scope_covers:
                return True
    return False


def _supports(c: Claim) -> list:
    return [e for e in c.edges if e.kind == EdgeKind.SUPPORT]


# ----------------------------------------------------------------------------
# The gates (SHELL only). Each returns (verdict, reason).  ALL defined here,
# then GATES is assembled once.
# ----------------------------------------------------------------------------
def g_domain(c: Claim, g: Graph):
    if not c.parameter_domain:
        return V.NA, "no parameter domain declared"
    gaps = []
    for e in _supports(c):
        dep = g.get(e.target)
        if not dep or not dep.parameter_domain:
            continue  # deliberate shell choice: non-declaration is not a violation
        for p, (lo, hi) in c.parameter_domain.items():
            if p in dep.parameter_domain:
                dlo, dhi = dep.parameter_domain[p]
                if dlo > lo or dhi < hi:
                    gaps.append(f"{e.target}:{p} covers [{dlo},{dhi}] < [{lo},{hi}]")
    if gaps:
        return V.FAIL, "domain not covered: " + "; ".join(gaps)
    return V.PASS, "dependencies cover the asserted domain"


def g_endpoint(c: Claim, g: Graph):
    if not c.finite_range_bound:
        return V.NA, "not a finite-range bound"
    if not c.registered_endpoints:
        return V.FAIL, "finite-range bound with no registered endpoints"
    bad = [str(pt) for (pt, val, ok) in c.registered_endpoints if not ok]
    if bad:
        return V.FAIL, "bound violated at endpoints: " + ", ".join(bad)
    return V.PASS, "bound holds at every registered endpoint"


def g_polarity(c: Claim, g: Graph):
    if c.direction in ("none", "two-sided"):
        return V.NA, "claim not one-sided"
    if c.direction == "upper" and c.uncertainty_side not in ("plus", "none"):
        return V.FAIL, "upper claim carries a non-(+) band"
    if c.direction == "lower" and c.uncertainty_side not in ("minus", "none"):
        return V.FAIL, "lower claim carries a non-(-) band"
    bad = []
    for e in _supports(c):
        dep = g.get(e.target)
        if not dep or dep.direction in ("none", "two-sided"):
            continue
        if dep.direction != c.direction:
            bad.append(f"{e.target}({dep.direction})")
    if bad:
        return V.FAIL, f"{c.direction} claim fed by opposite-sided inputs: " + ", ".join(bad)
    return V.PASS, "input sides and band side match a one-sided claim"


def g_measure(c: Claim, g: Graph):
    if not c.measure_tag:
        return V.NA, "no measure tag (claim not measure-dependent)"
    mism = [e.target for e in _supports(c)
            if g.get(e.target) and g[e.target].measure_tag
            and g[e.target].measure_tag != c.measure_tag]
    if not mism:
        return V.PASS, f"all inputs under required law ({c.measure_tag})"
    if _bridge_sufficient(c, EdgeKind.CHANGE_OF_MEASURE, g):
        return V.PASS, "measure mismatch bridged by a graded change-of-measure node"
    return V.FAIL, "silent measure swap (no sufficient change-of-measure node): " + ", ".join(mism)


def g_mark(c: Claim, g: Graph):
    if not c.required_marks:
        return V.NA, "no required marks"
    missing = c.required_marks - c.supplied_marks
    if not missing:
        return V.PASS, "all required marks supplied"
    if _bridge_sufficient(c, EdgeKind.MARK_TRANSFER, g):
        return V.PASS, "unmarked support licensed by a graded mark-transfer node"
    return V.FAIL, "marked obligation on unmarked support: missing " + ", ".join(sorted(missing))


def g_composition(c: Claim, g: Graph):
    if not c.is_combination:
        return V.NA, "not a combination"
    for e in _bridges(c, EdgeKind.COMPOSITION_WITNESS):
        node = g.get(e.target)
        if node and edge_ok(node.grade, c.grade, conditional=False):
            return V.PASS, f"combination carries a graded witness ({e.target})"
    if _bridges(c, EdgeKind.COMPOSITION_WITNESS):
        return V.FAIL, "composition witness cited but graded too weak to support this claim"
    return V.FAIL, "witness-free product/union/assembly"


def g_rung(c: Claim, g: Graph):
    if not c.measured_values:
        return V.NA, "no measured values"
    bad = [n for (n, v, s) in c.measured_values if not s]
    if bad:
        return V.FAIL, "measured value(s) with no scale tag: " + ", ".join(bad)
    return V.PASS, "every measured value carries its scale"


def g_precedence(c: Claim, g: Graph):
    dead = [nid for nid in _reachable(c.id, g)
            if g[nid].status in ("killed", "superseded", "retired")]
    if dead:
        return V.FAIL, "reaches dead node(s): " + ", ".join(dead)
    return V.PASS, "no killed/superseded/retired node reachable"


def g_core_closure(c: Claim, g: Graph):
    if not c.closed_core:
        return V.NA, "not declared closed-core"
    opens = [nid for nid in _reachable(c.id, g) if g[nid].grade == Grade.OPEN]
    if opens:
        return V.FAIL, "closed core rests on Open node(s): " + ", ".join(opens)
    return V.PASS, "closed core has no Open dependency"


def g_coverage(c: Claim, g: Graph):
    if not c.failure_regions and not c.global_suppression_argument:
        return V.NA, "no failure regions / global argument declared"
    if c.global_suppression_argument and not c.failure_regions:
        return V.FAIL, "global suppression argument with no registered detector"
    uncharted = [r for r, dets in c.failure_regions.items() if not dets]
    if uncharted:
        return V.FAIL, "un-instrumented failure region(s): " + ", ".join(uncharted)
    return V.PASS, "every registered failure region maps to a detector"


def g_common_mode(c: Claim, g: Graph):
    if c.verification == "single_exact":
        return V.NA, "single-instrument exact check (exempt)"
    if c.verification != "agreement_based":
        return V.NA, "verdict not agreement-based"
    if c.error_independence_arg or c.external_crosscheck:
        return V.PASS, "agreement carries error-independence / external cross-check"
    return V.PROVISIONAL, "verdict rests on agreement alone (no disjoint channel)"


def g_model(c: Claim, g: Graph):
    if not c.model_tag:
        if any(g.get(e.target) and g[e.target].model_tag for e in _supports(c)):
            return V.FAIL, "model-dependent claim with no model tag"
        return V.NA, "claim not model-dependent"
    mism = [e.target for e in _supports(c)
            if g.get(e.target) and g[e.target].model_tag
            and g[e.target].model_tag != c.model_tag]
    if not mism:
        return V.PASS, f"exact model named ({c.model_tag})"
    if _bridge_sufficient(c, EdgeKind.TRANSFER_CERTIFICATE, g):
        return V.PASS, "model mismatch bridged by an in-scope, graded transfer certificate"
    return V.FAIL, "model transfer without an in-scope graded certificate: " + ", ".join(mism)


def g_heuristic_bridge(c: Claim, g: Graph):
    if not c.heuristic_step:
        return V.NA, "no load-bearing heuristic step"
    if _bridge_sufficient(c, EdgeKind.HEURISTIC_BRIDGE, g):
        return V.PASS, "heuristic step carries a graded bridge (derivation)"
    if c.grade in (Grade.PLAUSIBLE, Grade.CONJECTURE, Grade.OPEN):
        return V.PASS, "unbridged heuristic, but claim already graded <= Plausible"
    return V.PROVISIONAL, "unbridged heuristic feeding a quantitative claim (ceiling = Plausible)"


# assembled ONCE, in checklist order (§9)
GATES = [
    ("DOMAIN", g_domain), ("ENDPOINT", g_endpoint), ("POLARITY", g_polarity),
    ("MEASURE", g_measure), ("MARK", g_mark), ("COMPOSITION", g_composition),
    ("RUNG", g_rung), ("PRECEDENCE", g_precedence), ("CORE CLOSURE", g_core_closure),
    ("MODEL", g_model), ("COVERAGE", g_coverage), ("COMMON-MODE", g_common_mode),
    ("HEURISTIC-BRIDGE", g_heuristic_bridge),
]


def g_constitutional(c: Claim, g: Graph):
    bad = []
    for e in c.edges:
        dep = g.get(e.target)
        if dep is None:
            bad.append(f"{e.target}?missing")
            continue
        if not edge_ok(dep.grade, c.grade, e.conditional):
            tag = "conditional" if e.conditional else "edge"
            bad.append(f"{e.target}({dep.grade.value},{tag})")
    if bad:
        return V.FAIL, "inadmissible dependency: " + ", ".join(bad)
    return V.PASS, "all edges admissible"


def hash_drift(c: Claim, g: Graph):
    stale = []
    for e in c.edges:
        dep = g.get(e.target)
        if dep and e.pinned_hash and e.pinned_hash != dep.content_hash:
            stale.append(f"{e.target}(pinned {e.pinned_hash} != now {dep.content_hash})")
    if stale:
        return V.FAIL, "STALE: dependency edited since last gating: " + "; ".join(stale)
    return V.PASS, "all dependency edges hash-consistent"


# ----------------------------------------------------------------------------
# ceiling arithmetic — "maximize the weakest link"
# ----------------------------------------------------------------------------
_WARRANT_ORDER = [Grade.KILLED, Grade.OPEN, Grade.CONJECTURE, Grade.PLAUSIBLE,
                  Grade.DERIVED, Grade.LITERATURE, Grade.PROVEN_MODULO,
                  Grade.PROVEN, Grade.PRIMITIVE]


def _min_grade(a: Grade, b: Grade) -> Grade:
    return a if _WARRANT_ORDER.index(a) <= _WARRANT_ORDER.index(b) else b


def _effective_ceiling(c: Claim, gate_results: dict, g: Graph, memo: dict) -> Grade:
    if c.id in memo:
        return memo[c.id]
    memo[c.id] = c.grade
    ceil = c.grade
    if gate_results.get("HEURISTIC-BRIDGE", (V.NA,))[0] == V.PROVISIONAL:
        ceil = _min_grade(ceil, Grade.PLAUSIBLE)
    if gate_results.get("COMMON-MODE", (V.NA,))[0] == V.PROVISIONAL:
        ceil = _min_grade(ceil, c.grade)
    for e in c.edges:
        dep = g.get(e.target)
        if dep and not e.conditional:   # conditional debts tracked separately, not inherited
            dep_gr = run_claim(dep, g, memo_ceiling=memo)["ceiling"]
            ceil = _min_grade(ceil, dep_gr)
    memo[c.id] = ceil
    return ceil


# ----------------------------------------------------------------------------
# run one claim
# ----------------------------------------------------------------------------
def run_claim(c: Claim, g: Graph, memo_ceiling: Optional[dict] = None) -> dict:
    results = {name: fn(c, g) for (name, fn) in GATES}
    con = g_constitutional(c, g)
    drift = hash_drift(c, g)

    fails = [n for n, (v, _) in results.items() if v == V.FAIL]
    if con[0] == V.FAIL:
        fails.append("CONSTITUTIONAL")
    if drift[0] == V.FAIL:
        fails.append("HASH-DRIFT")
    provs = [n for n, (v, _) in results.items() if v == V.PROVISIONAL]

    memo = memo_ceiling if memo_ceiling is not None else {}
    ceiling = _effective_ceiling(c, results, g, memo)

    opens = [nid for nid in _reachable(c.id, g) if g[nid].grade == Grade.OPEN]
    outstanding = [e.target for e in c.edges if e.conditional]

    if fails:
        verdict = f"BLOCKED at: {', '.join(fails)}"
    elif provs or ceiling != c.grade:
        verdict = f"PROVISIONAL — usable ceiling {ceiling.value}"
    else:
        verdict = f"PASS at {c.grade.value}"

    return {"results": results, "constitutional": con, "drift": drift,
            "ceiling": ceiling, "verdict": verdict, "closable": (not opens),
            "open_deps": opens, "modulo": outstanding, "fails": fails, "provs": provs}


# ----------------------------------------------------------------------------
# validate_graph — hard structural pre-pass (well-formedness, NOT semantics)
# ----------------------------------------------------------------------------
_STATUSES = {"live", "superseded", "killed", "retired"}


def _find_cycle(g: Graph) -> Optional[list]:
    """Return one dependency cycle (list of ids) if any, else None. DFS colouring."""
    WHITE, GREY, BLACK = 0, 1, 2
    color = {nid: WHITE for nid in g}
    stack_path: list = []

    def dfs(u) -> Optional[list]:
        color[u] = GREY
        stack_path.append(u)
        for e in g[u].edges:
            v = e.target
            if v not in g:
                continue
            if color[v] == GREY:                      # back-edge => cycle
                i = stack_path.index(v)
                return stack_path[i:] + [v]
            if color[v] == WHITE:
                r = dfs(v)
                if r:
                    return r
        stack_path.pop()
        color[u] = BLACK
        return None

    for nid in g:
        if color[nid] == WHITE:
            r = dfs(nid)
            if r:
                return r
    return None


def validate_graph(g: Graph) -> list:
    """Return a list of hard errors. Empty list == well-formed. This checks
    STRUCTURE only; passing it does not mean any claim's gates pass."""
    errors = []

    # 1. missing edge targets
    for c in g.values():
        for e in c.edges:
            if e.target not in g:
                errors.append(f"[missing-target] {c.id} -> {e.target} (not in graph)")

    # 2. duplicate content hashes (a hash must identify one claim's exact content)
    seen = {}
    for c in g.values():
        if c.content_hash in seen and seen[c.content_hash] != c.id:
            errors.append(f"[duplicate-hash] {c.id} and {seen[c.content_hash]} "
                          f"share content_hash '{c.content_hash}'")
        else:
            seen[c.content_hash] = c.id

    # 3. illegal cycle (a proof graph must be a DAG)
    cyc = _find_cycle(g)
    if cyc:
        errors.append("[cycle] circular dependency: " + " -> ".join(cyc))

    # 4. inconsistent status/grade
    for c in g.values():
        if c.status not in _STATUSES:
            errors.append(f"[bad-status] {c.id} has unknown status '{c.status}'")
        if c.grade == Grade.KILLED and c.status == "live":
            errors.append(f"[status/grade] {c.id} is graded Killed but marked live")

    return errors


# ----------------------------------------------------------------------------
# serialization for whole graphs / registry files
# ----------------------------------------------------------------------------
def graph_to_json(g: Graph, indent: int = 2) -> str:
    return json.dumps({"schema": "gate-kernel/1.2",
                       "claims": [c.to_dict() for c in g.values()]}, indent=indent)


def graph_from_json(s: str) -> Graph:
    data = json.loads(s)
    claims = [Claim.from_dict(d) for d in data["claims"]]
    return {c.id: c for c in claims}


def save_graph(g: Graph, path: str) -> None:
    with open(path, "w") as f:
        f.write(graph_to_json(g))


def load_graph(path: str) -> Graph:
    with open(path) as f:
        return graph_from_json(f.read())


# ----------------------------------------------------------------------------
# Registry — minimal, hash-keyed, leaf-up admission
# ----------------------------------------------------------------------------
class Registry:
    """Stores admitted claims by content hash. admit() runs the shell and REFUSES
    any claim whose verdict is BLOCKED; PROVISIONAL is admitted with its ceiling
    recorded. Dependencies must already be admitted (leaf-up construction)."""

    def __init__(self):
        self.by_id: dict = {}
        self.by_hash: dict = {}
        self.ceilings: dict = {}   # id -> recorded ceiling grade (for PROVISIONALs)

    def _graph_with(self, c: Claim) -> Graph:
        gg = dict(self.by_id)
        gg[c.id] = c
        return gg

    def admit(self, c: Claim):
        # integrity: a content hash identifies exactly one claim
        if c.content_hash in self.by_hash and self.by_hash[c.content_hash].id != c.id:
            return False, [f"content_hash '{c.content_hash}' already bound to "
                           f"{self.by_hash[c.content_hash].id}"]
        # leaf-up: every dependency must already be admitted
        unresolved = [e.target for e in c.edges if e.target not in self.by_id]
        if unresolved:
            return False, [f"unresolved dependency (admit it first): {t}" for t in unresolved]

        gg = self._graph_with(c)
        cyc = _find_cycle(gg)
        if cyc:
            return False, ["would create a cycle: " + " -> ".join(cyc)]

        r = run_claim(c, gg)
        if r["fails"]:
            return False, [f"BLOCKED at {', '.join(r['fails'])}"]

        # accept (PASS or PROVISIONAL-with-ceiling)
        self.by_id[c.id] = c
        self.by_hash[c.content_hash] = c
        if r["provs"] or r["ceiling"] != c.grade:
            self.ceilings[c.id] = r["ceiling"]
        return True, [r["verdict"]]

    def graph(self) -> Graph:
        return dict(self.by_id)

    def save(self, path: str) -> None:
        save_graph(self.by_id, path)

    @staticmethod
    def load(path: str) -> "Registry":
        reg = Registry()
        g = load_graph(path)
        # re-admit leaf-up so verdicts/ceilings are recomputed, not trusted from file
        for cid in _topo_order(g):
            reg.admit(g[cid])
        return reg


def _topo_order(g: Graph) -> list:
    """Dependencies before dependents (Kahn). Assumes a DAG; validate first."""
    indeg = {nid: 0 for nid in g}
    for c in g.values():
        for e in c.edges:
            if e.target in g:
                indeg[c.id] += 1
    # nodes with no unmet deps first
    ready = [nid for nid, d in indeg.items() if d == 0]
    order, seen = [], set(ready)
    # simple repeated sweep (graphs here are small)
    changed = True
    while changed:
        changed = False
        for nid in list(g):
            if nid in seen:
                continue
            deps = [e.target for e in g[nid].edges if e.target in g]
            if all(d in seen for d in deps):
                seen.add(nid)
                ready.append(nid)
                changed = True
    return ready


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
_SYM = {V.PASS: "PASS", V.FAIL: "FAIL", V.PROVISIONAL: "PROV", V.NA: " -- "}


def report(c: Claim, g: Graph):
    r = run_claim(c, g)
    print(f"\n{'='*74}\n{c.id}  [{c.grade.value}]  —  {c.statement}")
    print("-" * 74)
    print(f"  {'CONSTITUTIONAL':<16} {_SYM[r['constitutional'][0]]}   {r['constitutional'][1]}")
    print(f"  {'HASH-INTEGRITY':<16} {_SYM[r['drift'][0]]}   {r['drift'][1]}")
    for name, _ in GATES:
        v, why = r["results"][name]
        print(f"  {name:<16} {_SYM[v]}   {why}")
    print("-" * 74)
    if c.closed_core or r["open_deps"]:
        print(f"  closable: {'YES' if r['closable'] else 'NO'}"
              + ("" if r["closable"] else f"  (Open: {', '.join(r['open_deps'])})"))
    if r["modulo"]:
        print(f"  outstanding modulo-conditions: {', '.join(r['modulo'])}")
    print(f"  >> VERDICT: {r['verdict']}")


# ============================================================================
# DEMONSTRATIONS
# ============================================================================
def demo_program() -> Graph:
    """The live q0-rate program with R0 encoded as a graded Open node."""
    g: Graph = {}

    def add(c): g[c.id] = c; return c

    add(Claim(id="R0",
              statement="Remainder of the pairing-failure expansion under typed pair-Palm at rate order O(r^3): currently unbounded.",
              grade=Grade.OPEN, content_hash="R0#a1", measure_tag="pair-Palm", model_tag="kernelK"))

    for rid, txt in [("inner", "inner-region pairing-failure bound"),
                     ("far", "far-region pairing-failure bound"),
                     ("bdry", "boundary-region pairing-failure bound")]:
        add(Claim(id=f"{rid}_bound", statement=txt, grade=Grade.PROVEN,
                  content_hash=f"{rid}#1", measure_tag="pair-Palm", model_tag="kernelK",
                  parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
                  supplied_marks={"b"}, direction="upper", uncertainty_side="plus"))

    add(Claim(id="cm_node",
              statement="Change of measure: Gaussian-pinned weight -> pair-Palm weight (Palm/Slivnyak accounting).",
              grade=Grade.PROVEN, content_hash="cm#1"))
    add(Claim(id="cw_node",
              statement="Composition witness: union bound over the three disjoint regions.",
              grade=Grade.PROVEN, content_hash="cw#1"))

    add(Claim(id="C_const",
              statement="Coefficient C in the C*r^3 ceiling (closed form + numerical value).",
              grade=Grade.DERIVED, content_hash="C#1", measure_tag="pair-Palm", model_tag="kernelK",
              measured_values=[("C", "0.4127", "units of r^3")],
              verification="agreement_based", error_independence_arg=False, external_crosscheck=False))

    add(Claim(id="T_main",
              statement="Pairing-failure probability <= C*r^3 under pair-Palm, for r in (0, r0], b in B.",
              grade=Grade.PROVEN_MODULO, content_hash="T#1", measure_tag="pair-Palm", model_tag="kernelK",
              parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
              required_marks={"b"}, supplied_marks={"b"},
              direction="upper", uncertainty_side="plus",
              measured_values=[("C", "0.4127", "units of r^3")],
              is_combination=True, finite_range_bound=True,
              registered_endpoints=[("r=r0", "C*r0^3", True), ("r->0+", "0", True)],
              closed_core=False,
              edges=[Edge("inner_bound", EdgeKind.SUPPORT, "inner#1"),
                     Edge("far_bound", EdgeKind.SUPPORT, "far#1"),
                     Edge("bdry_bound", EdgeKind.SUPPORT, "bdry#1"),
                     Edge("cw_node", EdgeKind.COMPOSITION_WITNESS, "cw#1"),
                     Edge("cm_node", EdgeKind.CHANGE_OF_MEASURE, "cm#1"),
                     Edge("C_const", EdgeKind.SUPPORT, "C#1", conditional=True),
                     Edge("R0", EdgeKind.SUPPORT, "R0#a1", conditional=True)]))

    # closed-attempt variant, built cleanly via serialization round-trip
    t_closed = Claim.from_dict({**g["T_main"].to_dict(),
                                "id": "T_main_CLOSED_ATTEMPT",
                                "statement": "Same theorem, wrongly declared CLOSED.",
                                "content_hash": "Tc#1", "closed_core": True})
    add(t_closed)

    # injected structural failures (shell-catchable)
    add(Claim(id="pinned_input", statement="A coefficient computed under Gaussian-pinning.",
              grade=Grade.PROVEN, content_hash="pin#1", measure_tag="Gaussian-pinned"))
    add(Claim(id="BAD_composition",
              statement="Corridor probability modeled as a product of per-checkpoint probabilities (no witness).",
              grade=Grade.DERIVED, content_hash="bc#1", measure_tag="pair-Palm", model_tag="kernelK",
              is_combination=True, edges=[Edge("inner_bound", EdgeKind.SUPPORT, "inner#1")]))
    add(Claim(id="BAD_measure",
              statement="Uses a Gaussian-pinned coefficient where the claim needs the pair-Palm weight (no bridge).",
              grade=Grade.DERIVED, content_hash="bm#1", measure_tag="pair-Palm", model_tag="kernelK",
              edges=[Edge("pinned_input", EdgeKind.SUPPORT, "pin#1")]))
    add(Claim(id="BAD_stale",
              statement="Rests on inner_bound but was gated against an OLD version of it.",
              grade=Grade.DERIVED, content_hash="bs#1", measure_tag="pair-Palm", model_tag="kernelK",
              edges=[Edge("inner_bound", EdgeKind.SUPPORT, "inner#OLD")]))
    return g


def demo_malformed() -> Graph:
    """A deliberately ill-formed graph to exercise validate_graph."""
    g: Graph = {}
    g["A"] = Claim(id="A", statement="depends on a nonexistent node and on B",
                   grade=Grade.DERIVED, content_hash="dup#1",
                   edges=[Edge("GHOST"), Edge("B")])
    g["B"] = Claim(id="B", statement="cycles back to A; shares A's hash",
                   grade=Grade.DERIVED, content_hash="dup#1", edges=[Edge("A")])
    g["K"] = Claim(id="K", statement="graded Killed but marked live",
                   grade=Grade.KILLED, content_hash="k#1", status="live")
    return g


if __name__ == "__main__":
    print(__doc__)

    g = demo_program()

    print("\n\n########## [1] GRAPH VALIDATION (live program) ##########")
    errs = validate_graph(g)
    print("  well-formed." if not errs else "\n".join("  " + e for e in errs))

    print("\n\n########## [2] validate_graph ON A MALFORMED GRAPH ##########")
    for e in validate_graph(demo_malformed()):
        print("  " + e)

    print("\n\n########## [3] PER-CLAIM GATE REPORTS ##########")
    for cid in ["T_main", "T_main_CLOSED_ATTEMPT", "C_const",
                "BAD_composition", "BAD_measure", "BAD_stale"]:
        report(g[cid], g)

    print("\n\n########## [4] REGISTRY ADMISSION (leaf-up) ##########")
    reg = Registry()
    for cid in _topo_order(g):
        ok, msg = reg.admit(g[cid])
        flag = "ADMIT " if ok else "REFUSE"
        print(f"  {flag}  {cid:<24} {msg[0]}")

    print("\n\n########## [5] JSON ROUND-TRIP ##########")
    reg.save("/mnt/user-data/outputs/q0_registry.json")
    reg2 = Registry.load("/mnt/user-data/outputs/q0_registry.json")
    same = (set(reg.by_id) == set(reg2.by_id)
            and all(run_claim(reg.by_id[i], reg.graph())["verdict"]
                    == run_claim(reg2.by_id[i], reg2.graph())["verdict"] for i in reg.by_id))
    print(f"  wrote q0_registry.json, reloaded {len(reg2.by_id)} claims; "
          f"verdicts stable across round-trip: {same}")

    print("\n" + "=" * 74)
    print("Reminder (Master File §0.5): every PASS above is a SHELL verdict —")
    print("tag-consistency under the declared tags, NOT tag-to-content fidelity.")
