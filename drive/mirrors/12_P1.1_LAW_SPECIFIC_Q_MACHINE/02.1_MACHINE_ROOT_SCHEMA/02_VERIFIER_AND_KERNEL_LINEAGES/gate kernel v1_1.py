#!/usr/bin/env python3
"""
GATE KERNEL v1.1 — executable enforcement of the Master Gate File (v1.1) SHELL.

================================  SCOPE (read first)  ========================
This kernel checks the mechanical SHELL of every gate:
  - tag presence and tag-compatibility across dependency edges,
  - graph reachability (PRECEDENCE, CORE CLOSURE),
  - dependability of every cited node INCLUDING bridge nodes (constitutional
    rule + the three dependability tiers),
  - hash-drift / stale-PASS across edges.
For each claim it returns the 13-gate vector + constitutional check + a hash
integrity check + an effective verdict (promotable? capped? blocked?).

It DOES NOT and CANNOT check the CORE of any gate — whether a tagged claim is
TRUE, whether a cited change-of-measure is VALID, whether a failure-region
list is COMPLETE. Per Master File §0.5 the kernel certifies tag-CONSISTENCY,
not tag-to-content FIDELITY. Fidelity is produced only by the §7 disciplines
(adversarial reimplementation, separation of duties). A kernel PASS means
"consistent under the declared tags," never "correct."

Corollary the output states on every run: the shell-catchable failures are the
STRUCTURAL ones (missing witness, tag mismatch, domain gap, reachable dead
node, stale edge). The WRONG-BUT-NAMED-witness failure — a false "Independence"
node that is correctly typed and sufficiently graded but mathematically false —
is a CORE failure and passes the shell. §3.8 narrows it (the false node must be
graded, so it enters the §7 verification queue) but does not let the kernel
adjudicate it.
=============================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


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


# Fully dependable: may support a claim of ANY grade.
_FULLY = {Grade.PRIMITIVE, Grade.PROVEN, Grade.PROVEN_MODULO, Grade.LITERATURE}


def dependability_level(g: Grade) -> int:
    """3 = fully dependable, 2 = self-tier (Derived), 1 = Plausible-tier,
    0 = not dependable (Conjecture/Open), -1 = Killed (never)."""
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
    """Minimum dependability a NON-conditional dependency must have to support
    a parent of the given grade."""
    if parent in _FULLY:
        return 3          # nothing below fully-dependable may lift a Proven/-Modulo/Lit/Primitive claim
    if parent == Grade.DERIVED:
        return 2          # Derived-on-Derived is legal; nothing above Derived launders through it
    if parent == Grade.PLAUSIBLE:
        return 1
    return 0


def edge_ok(dep_grade: Grade, parent_grade: Grade, conditional: bool) -> bool:
    """Constitutional admissibility of a single edge."""
    if dep_grade == Grade.KILLED:
        return False  # a Killed node may never be depended on, conditional or not
    if conditional and parent_grade == Grade.PROVEN_MODULO:
        # Proven-Modulo == "proved GIVEN named conditions"; a declared conditional
        # edge may point at an Open/Conjecture/not-yet-fully-dependable node.
        # (CORE CLOSURE separately forbids CLOSING such a claim.)
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
# Data model — the §2 proof object (fields the SHELL needs)
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
    target: str                       # dependency / bridge node id
    kind: EdgeKind = EdgeKind.SUPPORT
    pinned_hash: str = ""             # hash of target AT THE TIME this edge was gated
    conditional: bool = False         # a declared "modulo" condition (Proven-Modulo only)
    scope_covers: bool = True         # for TRANSFER_CERTIFICATE: does its scope cover deployment?


@dataclass
class Claim:
    id: str
    statement: str
    grade: Grade
    content_hash: str
    status: str = "live"              # live | superseded | killed | retired

    # transfer axes
    measure_tag: str = ""
    model_tag: str = ""
    parameter_domain: dict = field(default_factory=dict)   # param -> (lo, hi)
    required_marks: set = field(default_factory=set)
    supplied_marks: set = field(default_factory=set)

    # sidedness
    direction: str = "none"           # upper | lower | two-sided | none
    uncertainty_side: str = "none"    # plus | minus | symmetric | none

    # scale
    measured_values: list = field(default_factory=list)    # (name, value, scale_tag)

    # composition
    is_combination: bool = False

    # endpoints (finite-range bounds)
    finite_range_bound: bool = False
    registered_endpoints: list = field(default_factory=list)   # (point, value, satisfies)

    # coverage
    failure_regions: dict = field(default_factory=dict)    # region -> [detector ids]
    global_suppression_argument: bool = False

    # verification methodology (COMMON-MODE)
    verification: str = "none"        # single_exact | agreement_based | none
    error_independence_arg: bool = False
    external_crosscheck: bool = False

    # rigor mode (HEURISTIC-BRIDGE)
    heuristic_step: bool = False      # a load-bearing analogical/heuristic step feeds this claim

    # closure
    closed_core: bool = False

    edges: list = field(default_factory=list)


Graph = dict  # id -> Claim


# ----------------------------------------------------------------------------
# graph helpers
# ----------------------------------------------------------------------------
def _reachable(start: str, g: Graph) -> set:
    seen, stack = set(), [start]
    while stack:
        cur = stack.pop()
        for e in g.get(cur, Claim("", "", Grade.OPEN, "")).edges:
            if e.target not in seen and e.target in g:
                seen.add(e.target)
                stack.append(e.target)
    return seen


def _bridges(c: Claim, kind: EdgeKind) -> list:
    return [e for e in c.edges if e.kind == kind]


def _bridge_sufficient(c: Claim, kind: EdgeKind, g: Graph) -> bool:
    """A bridge of `kind` is present AND graded high enough to lift this parent."""
    for e in _bridges(c, kind):
        node = g.get(e.target)
        if node and edge_ok(node.grade, c.grade, conditional=False):
            if kind != EdgeKind.TRANSFER_CERTIFICATE or e.scope_covers:
                return True
    return False


def _supports(c: Claim) -> list:
    return [e for e in c.edges if e.kind == EdgeKind.SUPPORT]


# ----------------------------------------------------------------------------
# The gates (SHELL only). Each returns (verdict, reason).
# ----------------------------------------------------------------------------
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


def g_domain(c: Claim, g: Graph):
    if not c.parameter_domain:
        return V.NA, "no parameter domain declared"
    gaps = []
    for e in _supports(c):
        dep = g.get(e.target)
        if not dep or not dep.parameter_domain:
            continue
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
    # band side must match
    if c.direction == "upper" and c.uncertainty_side not in ("plus", "none"):
        return V.FAIL, "upper claim carries a non-(+) band"
    if c.direction == "lower" and c.uncertainty_side not in ("minus", "none"):
        return V.FAIL, "lower claim carries a non-(-) band"
    # input directions must match
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


def g_heuristic_bridge(c: Claim, g: Graph):
    if not c.heuristic_step:
        return V.NA, "no load-bearing heuristic step"
    if _bridge_sufficient(c, EdgeKind.HEURISTIC_BRIDGE, g):
        return V.PASS, "heuristic step carries a graded bridge (derivation)"
    if c.grade in (Grade.PLAUSIBLE, Grade.CONJECTURE, Grade.OPEN):
        return V.PASS, "unbridged heuristic, but claim already graded <= Plausible"
    return V.PROVISIONAL, "unbridged heuristic feeding a quantitative claim (ceiling = Plausible)"


GATES = [
    ("DOMAIN", g_domain), ("ENDPOINT", g_endpoint), ("POLARITY", g_polarity),
    ("MEASURE", g_measure), ("MARK", g_mark), ("COMPOSITION", g_composition),
    ("RUNG", g_rung), ("PRECEDENCE", g_precedence), ("CORE CLOSURE", g_core_closure),
    ("MODEL", None), ("COVERAGE", g_coverage), ("COMMON-MODE", g_common_mode),
    ("HEURISTIC-BRIDGE", g_heuristic_bridge),
]


def g_model(c: Claim, g: Graph):
    if not c.model_tag:
        # only a failure if the claim is model-dependent; we treat a declared
        # covariance/model dependency as signalled by having support edges that
        # themselves carry a model tag.
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


# fill the MODEL slot (defined after GATES for readability)
GATES = [(name, g_model if name == "MODEL" else fn) for (name, fn) in GATES]


# ----------------------------------------------------------------------------
# hash-drift / stale-PASS
# ----------------------------------------------------------------------------
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
    """A claim's ceiling = its stated grade, lowered by any PROVISIONAL gate on
    itself, and by the ceiling of any dependency it rests on (weakest link)."""
    if c.id in memo:
        return memo[c.id]
    memo[c.id] = c.grade  # guard against cycles
    ceil = c.grade
    # own PROVISIONAL gates
    if gate_results.get("HEURISTIC-BRIDGE", (V.NA,))[0] == V.PROVISIONAL:
        ceil = _min_grade(ceil, Grade.PLAUSIBLE)
    if gate_results.get("COMMON-MODE", (V.NA,))[0] == V.PROVISIONAL:
        # COMMON-MODE freezes the claim at its current grade (no upward promotion);
        # for a supported parent this means the dependency cannot rise to fully-
        # dependable, so its usable ceiling is its present grade.
        ceil = _min_grade(ceil, c.grade)
    # inherit from dependencies (support + bridge)
    for e in c.edges:
        dep = g.get(e.target)
        if dep and not e.conditional:  # conditional debts are tracked separately, not inherited as ceiling
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

    return {
        "results": results, "constitutional": con, "drift": drift,
        "ceiling": ceiling, "verdict": verdict,
        "closable": (not opens), "open_deps": opens, "modulo": outstanding,
        "fails": fails, "provs": provs,
    }


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
# DEMONSTRATION — the live q0-rate program, with R0 encoded as a graded object.
# Running the kernel on the actual program should reproduce the review's
# conclusion WITHOUT being told it: the main theorem is legitimately
# Proven-Modulo, is NOT closable while R0 is Open, and its supporting constant
# is COMMON-MODE PROVISIONAL until a disjoint second pipeline exists.
# ============================================================================
def demo() -> Graph:
    g: Graph = {}

    def add(c): g[c.id] = c; return c

    # --- R0: the remainder under typed pair-Palm at the order of the rate. OPEN.
    add(Claim(
        id="R0",
        statement="Remainder of the pairing-failure expansion under typed pair-Palm at rate order O(r^3): currently unbounded.",
        grade=Grade.OPEN, content_hash="R0#a1",
        measure_tag="pair-Palm", model_tag="kernelK",
    ))

    # --- the three regional bounds (referee-grade, cover the full box, marked in b)
    for rid, txt in [
        ("inner", "inner-region pairing-failure bound"),
        ("far", "far-region pairing-failure bound"),
        ("bdry", "boundary-region pairing-failure bound"),
    ]:
        add(Claim(
            id=f"{rid}_bound", statement=txt, grade=Grade.PROVEN,
            content_hash=f"{rid}#1", measure_tag="pair-Palm", model_tag="kernelK",
            parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
            supplied_marks={"b"}, direction="upper", uncertainty_side="plus",
        ))

    # --- bridge nodes, now first-class graded objects (§3.8)
    add(Claim(
        id="cm_node",
        statement="Change of measure: Gaussian-pinned weight -> pair-Palm weight (Palm/Slivnyak accounting).",
        grade=Grade.PROVEN, content_hash="cm#1",
    ))
    add(Claim(
        id="cw_node",
        statement="Composition witness: union bound over the three disjoint regions.",
        grade=Grade.PROVEN, content_hash="cw#1",
    ))

    # --- the constant C: certified by two pipelines that SHARE a stack -> COMMON-MODE
    add(Claim(
        id="C_const",
        statement="Coefficient C in the C*r^3 ceiling (closed form + numerical value).",
        grade=Grade.DERIVED, content_hash="C#1",
        measure_tag="pair-Palm", model_tag="kernelK",
        measured_values=[("C", "0.4127", "units of r^3")],
        verification="agreement_based",           # two instruments agree...
        error_independence_arg=False,             # ...but share the linear-algebra/sampling stack
        external_crosscheck=False,
    ))

    # --- the MAIN THEOREM. Proven-Modulo, resting on the regional bounds + witness,
    #     with R0 and C-certification as DECLARED conditional (modulo) debts.
    add(Claim(
        id="T_main",
        statement="Pairing-failure probability <= C*r^3 under pair-Palm, for r in (0, r0], b in B.",
        grade=Grade.PROVEN_MODULO, content_hash="T#1",
        measure_tag="pair-Palm", model_tag="kernelK",
        parameter_domain={"r": (0.0, 1.0), "b": (0.0, 3.0)},
        required_marks={"b"}, supplied_marks={"b"},
        direction="upper", uncertainty_side="plus",
        measured_values=[("C", "0.4127", "units of r^3")],
        is_combination=True,
        finite_range_bound=True,
        registered_endpoints=[("r=r0", "C*r0^3", True), ("r->0+", "0", True)],
        closed_core=False,
        edges=[
            Edge("inner_bound", EdgeKind.SUPPORT, "inner#1"),
            Edge("far_bound", EdgeKind.SUPPORT, "far#1"),
            Edge("bdry_bound", EdgeKind.SUPPORT, "bdry#1"),
            Edge("cw_node", EdgeKind.COMPOSITION_WITNESS, "cw#1"),
            Edge("cm_node", EdgeKind.CHANGE_OF_MEASURE, "cm#1"),
            Edge("C_const", EdgeKind.SUPPORT, "C#1", conditional=True),   # modulo: C certified by disjoint pipeline
            Edge("R0", EdgeKind.SUPPORT, "R0#a1", conditional=True),      # modulo: R0 discharged
        ],
    ))

    # --- an ATTEMPT to declare the theorem closed while R0 is Open (should FAIL)
    t_closed = Claim(**{**T_main_kwargs(g["T_main"]), "id": "T_main_CLOSED_ATTEMPT",
                        "statement": "Same theorem, wrongly declared CLOSED.",
                        "closed_core": True, "content_hash": "Tc#1"})
    add(t_closed)

    # --- three deliberately-broken claims to exhibit shell-catchable FAILs
    add(Claim(
        id="BAD_composition",
        statement="Corridor probability modeled as a product of per-checkpoint probabilities (no witness).",
        grade=Grade.DERIVED, content_hash="bc#1",
        measure_tag="pair-Palm", model_tag="kernelK", is_combination=True,
        edges=[Edge("inner_bound", EdgeKind.SUPPORT, "inner#1")],   # no composition_witness edge
    ))
    add(Claim(
        id="BAD_measure",
        statement="Uses a Gaussian-pinned coefficient where the claim needs the pair-Palm weight (no bridge).",
        grade=Grade.DERIVED, content_hash="bm#1",
        measure_tag="pair-Palm", model_tag="kernelK",
        edges=[Edge("pinned_input", EdgeKind.SUPPORT, "pin#1")],
    ))
    add(Claim(
        id="pinned_input", statement="A coefficient computed under Gaussian-pinning.",
        grade=Grade.PROVEN, content_hash="pin#1", measure_tag="Gaussian-pinned",
    ))
    add(Claim(
        id="BAD_stale",
        statement="Rests on inner_bound but was gated against an OLD version of it.",
        grade=Grade.DERIVED, content_hash="bs#1", measure_tag="pair-Palm", model_tag="kernelK",
        edges=[Edge("inner_bound", EdgeKind.SUPPORT, "inner#OLD")],   # pinned hash != current inner#1
    ))
    return g


def T_main_kwargs(c: Claim) -> dict:
    """shallow copy of the mutable fields of T_main for the closed-attempt variant."""
    from copy import deepcopy
    d = deepcopy(c.__dict__)
    return d


if __name__ == "__main__":
    print(__doc__)
    g = demo()
    order = ["T_main", "T_main_CLOSED_ATTEMPT", "C_const",
             "BAD_composition", "BAD_measure", "BAD_stale"]
    for cid in order:
        report(g[cid], g)
    print("\n" + "=" * 74)
    print("Reminder (Master File §0.5): every PASS above is a SHELL verdict —")
    print("tag-consistency under the declared tags, NOT tag-to-content fidelity.")
    print("The wrong-but-named-witness failure is a CORE failure and is invisible")
    print("here by design; it enters the §7 verification queue via §3.8 grading.")
