"""Durable bounded research coordination, with exact local arithmetic examples.

Policies are inspectable journal-history heuristics. This module neither trains a
model nor awards Universal Law theorem certificates or executes submitted code.
"""
from __future__ import annotations

import copy
import math
import sys
import time
import uuid
from contextlib import contextmanager, ExitStack
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT / "engine") not in sys.path:
    sys.path.insert(0, str(ROOT / "engine"))
from conductor import Journal
from model import digest, fingerprints, load

STRATEGIES = {
    "rational_sampling": {"context": "rational_quadratic", "base": 20,
        "method": "Evaluate exactly at a finite rational grid; cannot establish a universal inequality."},
    "exact_square_completion": {"context": "rational_quadratic", "base": 10,
        "method": "Complete the square over rationals and recompute the minimum and a witness."},
    "source_scope_audit": {"context": "canonical", "base": 20,
        "method": "Compare the bound statement and exact source scope; missing source bytes block conclusions."},
    "dependency_decomposition": {"context": "canonical", "base": 15,
        "method": "Inspect explicit premises and open side conditions without inferring theorem truth."},
    "contradiction_investigation": {"context": "conflict", "base": 20,
        "method": "Compare both preserved assertions and evidence at the identical proposition and scope."},
    "independent_source_comparison": {"context": "conflict", "base": 15,
        "method": "Seek independent evidence before an explicit owner decision."},
}
DIAGNOSES = {"sample_coverage", "missing_evidence", "scope_mismatch", "dependency_gap",
             "execution_error", "none", "contradictory_evidence"}
EXAMPLES = {"nonnegative": ["1", "-2", "1"], "counterexample": ["1", "-2", "0"]}
MILESTONES = [
    {"milestone_id": "d5-second-factorial-upper", "title": "D5 matching upper second factorial moment",
     "intended_question": "Establish or refute a scope-aligned upper bound E[N_r(N_r-1)] <= C*r^3 for the full nonpin between-pin-height count, including colliding witness configurations, under the original conditioned and endpoint-tilted law in every fixed d>=2.",
     "scope": {"kind": "research_milestone", "milestone": "d5-second-factorial-upper", "dimension": "each fixed d>=2",
        "model": "original fixed periodized torus model", "law": "original conditioned and endpoint-tilted law",
        "marks": "compact birth marks and positive gap marks bounded away from zero",
        "count": "all nonpin critical points N_r in (b-k*r^3,b)", "regime": "r down to zero",
        "required_coverage": "full multipoint and collision strata; separated tuples alone are insufficient"},
     "bound_record_ids": ["D5d-M", "D5d-G", "D4-REMOTE-PAIR", "D4-TUPLE", "D4-KR"],
     "base_priority": 500,
     "priority_reason": "Highest open count milestone: D5d-M gives a lower bound only; no matching upper second factorial moment is admitted. Full collision control remains research work."},
    {"milestone_id": "marked-kac-rice-elder-alignment", "title": "Marked Kac–Rice and global Borel elder alignment",
     "intended_question": "Check a complete scope-aligned chain from the original conditional field kernel and continuous marked Kac–Rice hypotheses to the global Borel elder indicator and diagonal exhaustion, identifying every unsupported applicability step.",
     "scope": {"kind": "research_milestone", "milestone": "marked-kac-rice-elder-alignment",
        "model": "fixed smooth periodized Gaussian field on a compact torus",
        "marks": "global Borel elder event on the Morse distinct-critical-value locus",
        "domain": "compact off-diagonal exhaustion followed by an integrable near-diagonal limit",
        "required_coverage": "conditional kernel, finite measures, continuous marks, monotone extension and exhaustion hypotheses"},
     "bound_record_ids": ["D1-KR-BOREL", "D1-ELDER-MEAS", "D1-KR-KERNEL", "D1-KR-DOM"],
     "base_priority": 450,
     "priority_reason": "A marked formula is useful only after matching the actual global elder event and all conditional-kernel and exhaustion assumptions."},
    {"milestone_id": "normalizer-floor-weighted-transfer", "title": "Original endpoint normalizer floor and weighted transfer",
     "intended_question": "Verify a uniform positive floor for the correctly scaled original endpoint normalizer on the stated compact marks, and audit weighted transfer using exactly that denominator rather than a witness-conditioned replacement.",
     "scope": {"kind": "research_milestone", "milestone": "normalizer-floor-weighted-transfer",
        "model": "original endpoint-conditioned fixed periodized torus law",
        "marks": "compact birth and strictly positive compact gap marks; fixed dimension and torus",
        "normalization": "original full endpoint determinant/index tilt with its stated small-r scaling",
        "required_coverage": "uniform positivity, endpoint-only denominator, numerator moments and transfer applicability"},
     "bound_record_ids": ["D4-Z", "D5-ANN-R5", "D1-02"],
     "base_priority": 400,
     "priority_reason": "Weighted estimates require the original endpoint denominator and a uniform floor; changing to a witness-conditioned denominator changes the intended question."},
]


def text(value, name, limit=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError("invalid " + name)
    return value


def bounded(value, name, size=20000):
    """Reject non-JSON values, excessive nesting, NaN and oversized data."""
    import json
    def visit(item, depth=0):
        if depth > 12:
            raise ValueError(name + " nesting exceeds bound")
        if item is None or type(item) in (bool, int, float, str):
            return
        if isinstance(item, (list, dict)) and len(item) <= 256:
            if isinstance(item, dict):
                if not all(isinstance(k, str) for k in item):
                    raise ValueError(name + " keys must be strings")
                item = item.values()
            for child in item:
                visit(child, depth + 1)
            return
        raise ValueError("invalid " + name)
    visit(value)
    try:
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("invalid " + name) from exc
    if len(raw) > size:
        raise ValueError(name + " exceeds byte bound")
    return copy.deepcopy(value)


def evidence_list(value, required=False):
    if value is None:
        value = []
    if not isinstance(value, list) or len(value) > 32 or (required and not value):
        raise ValueError("evidence must be a bounded nonempty list" if required else "invalid evidence")
    return [text(v, "evidence reference", 2000) for v in value]


class ResearchDirector:
    def __init__(self, workdir, coordinator=None):
        self.coordinator = coordinator
        self.base = load()
        self.base_fingerprints = fingerprints(self.base)
        self.journal = Journal(workdir)
        self.snapshot()  # Corrupt historical state fails closed on construction.

    @contextmanager
    def _transaction(self):
        # Global order: live engine mutex, engine process lock, director lock.
        # Keep canonical inputs stable through result verification and append.
        with ExitStack() as stack:
            if self.coordinator is not None:
                stack.enter_context(self.coordinator._mutex)
                stack.enter_context(self.coordinator.journal.transaction())
            stack.enter_context(self.journal.transaction())
            yield

    def _inputs(self):
        if self.coordinator is not None:
            state = self.coordinator.snapshot()
            return ({r["id"]: r for r in state["records"]},
                    {o["id"]: o for o in state["obligations"]})
        return ({r["id"]: dict(r, fingerprint=self.base_fingerprints[r["id"]]) for r in self.base["records"]},
                {o["conclusion"]: dict(o, external_imports=o["external_theorem_imports"],
                    corrections=o["source_corrections"]) for o in self.base["obligations"]})

    @staticmethod
    def _canonical_binding(record):
        question = "Audit the exact proof obligation for " + record["id"] + ": " + record["statement"]
        scope = {"kind": "canonical_record", "record_id": record["id"], "text": record["scope"]}
        fp = digest({"question": question, "scope": scope, "record_fingerprint": record["fingerprint"]})
        return question, scope, fp

    def _read(self):
        if self.journal.path.exists() and self.journal.path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("director journal exceeds 32 MiB bound; archive explicitly")
        self.journal._load()
        if len(self.journal.events) > 20000:
            raise ValueError("director journal event bound exceeded")
        s = {"tasks": {}, "claims": {}, "attempts": [], "assertions": {}, "conflicts": {},
             "scope_proposals": [], "target_record_ids": []}
        for event in self.journal.events:
            p, kind = event["payload"], event["type"]
            if kind == "PLAN_SEEDED":
                for task in p["tasks"]:
                    if task["task_id"] in s["tasks"] and s["tasks"][task["task_id"]] != task:
                        raise ValueError("intended question or task binding changed in journal")
                    s["tasks"][task["task_id"]] = copy.deepcopy(task)
                s["target_record_ids"] = p["target_record_ids"]
            elif kind == "MILESTONES_SEEDED":
                for task in p["tasks"]:
                    if task["task_id"] in s["tasks"] and s["tasks"][task["task_id"]] != task:
                        raise ValueError("milestone question or scope changed in journal")
                    s["tasks"][task["task_id"]] = copy.deepcopy(task)
            elif kind == "EXAMPLE_CREATED":
                s["tasks"][p["task_id"]] = copy.deepcopy(p)
            elif kind == "TASK_CLAIMED":
                if p["task_id"] not in s["tasks"]:
                    raise ValueError("orphan claim")
                s["claims"][p["claim_id"]] = copy.deepcopy(p)
            elif kind == "ATTEMPT_RECORDED":
                if p["claim_id"] not in s["claims"]:
                    raise ValueError("orphan attempt")
                s["attempts"].append(copy.deepcopy(p))
            elif kind == "ASSERTION_ADDED":
                s["assertions"][p["assertion"]["assertion_id"]] = copy.deepcopy(p["assertion"])
                if p.get("conflict"):
                    s["conflicts"][p["conflict"]["conflict_id"]] = copy.deepcopy(p["conflict"])
                    s["tasks"][p["investigation"]["task_id"]] = copy.deepcopy(p["investigation"])
            elif kind == "CONFLICT_RESOLVED":
                if p["conflict_id"] not in s["conflicts"]:
                    raise ValueError("orphan conflict resolution")
                conflict = s["conflicts"][p["conflict_id"]]
                conflict.update(status="resolved_by_owner", resolution=copy.deepcopy(p))
            elif kind == "SCOPE_PROPOSED":
                s["scope_proposals"].append(copy.deepcopy(p))
            else:
                raise ValueError("unknown director event type: " + kind)
        return s

    def _append(self, kind, payload):
        if len(self.journal.events) >= 20000:
            raise ValueError("director journal event bound exceeded")
        return self.journal.append(kind, payload)

    def seed_plan(self, target_record_ids=None):
        with self._transaction():
            s = self._read()
            records, obligations = self._inputs()
            targets = s["target_record_ids"] if target_record_ids is None else target_record_ids
            if not isinstance(targets, list) or len(targets) > 104 or any(not isinstance(t, str) or t not in records for t in targets):
                raise ValueError("target_record_ids must identify canonical records")
            targets = sorted(set(targets))
            bindings = {rid: self._canonical_binding(r) for rid, r in records.items()}
            ids = {rid: "research:" + rid + ":" + bindings[rid][2][:16] for rid in records}
            tasks = []
            for rid in sorted(records):
                if ids[rid] in s["tasks"]:
                    continue
                question, scope, fp = bindings[rid]
                obligation = obligations[rid]
                task = {"task_id": ids[rid], "kind": "canonical", "context": "canonical",
                    "record_id": rid, "title": records[rid]["title"], "intended_question": question,
                    "scope": scope, "input_fingerprint": fp, "record_fingerprint": records[rid]["fingerprint"],
                    "dependencies": [ids[p] for p in obligation["premises"]],
                    "premise_record_ids": obligation["premises"],
                    "open_side_conditions": obligation["side_conditions"],
                    "external_imports": obligation.get("external_imports", []),
                    "source_corrections": obligation.get("corrections", []),
                    "status": "open", "full_theorem_certificate": False}
                tasks.append(task)
            if tasks or targets != s["target_record_ids"]:
                self._append("PLAN_SEEDED", {"tasks": tasks, "target_record_ids": targets})
            return self.snapshot()

    def seed_milestones(self):
        """Fixed open research questions; never new canonical mathematical records."""
        self.seed_plan()
        with self._transaction():
            s = self._read()
            records, _ = self._inputs()
            tasks = []
            for template in MILESTONES:
                task = copy.deepcopy(template)
                bound = {rid: records[rid]["fingerprint"] for rid in task["bound_record_ids"]}
                fp = digest({"question": task["intended_question"], "scope": task["scope"], "bound_record_fingerprints": bound})
                task_id = "milestone:" + task["milestone_id"] + ":" + fp[:16]
                if task_id in s["tasks"]:
                    continue
                dependencies = ["research:" + rid + ":" + self._canonical_binding(records[rid])[2][:16] for rid in task["bound_record_ids"]]
                task.update(task_id=task_id, kind="project_milestone", context="canonical",
                    input_fingerprint=fp, bound_record_fingerprints=bound, dependencies=dependencies,
                    status="open", established=False, full_theorem_certificate=False)
                tasks.append(task)
            if tasks:
                self._append("MILESTONES_SEEDED", {"tasks": tasks})
            return self.snapshot()

    def create_example_task(self, case="nonnegative"):
        if not isinstance(case, str) or case not in EXAMPLES:
            raise ValueError("unregistered example; allowed: nonnegative, counterexample")
        with self._transaction():
            s = self._read()
            task_id = "example:" + case
            if task_id in s["tasks"]:
                return copy.deepcopy(s["tasks"][task_id])
            scope = {"kind": "rational_quadratic", "domain": "rationals", "quantifier": "forall",
                     "relation": ">=0", "coefficients": EXAMPLES[case]}
            question = ("For every rational x, is x^2 - 2*x + " + EXAMPLES[case][2] + " >= 0?")
            task = {"task_id": task_id, "kind": "worked_example", "context": "rational_quadratic",
                "title": "Exact rational inequality: " + case, "intended_question": question,
                "scope": scope, "input_fingerprint": digest({"question": question, "scope": scope}),
                "dependencies": [], "status": "open", "full_theorem_certificate": False}
            self._append("EXAMPLE_CREATED", task)
            return copy.deepcopy(task)

    def _stale(self, task, records, state=None):
        if task["kind"] == "canonical":
            return self._canonical_binding(records[task["record_id"]])[2] != task["input_fingerprint"]
        if task.get("bound_record_fingerprints"):
            if any(records[rid]["fingerprint"] != fp for rid, fp in task["bound_record_fingerprints"].items()):
                return True
        if task.get("canonical_input_fingerprint"):
            record_id = task["scope"]["record_id"]
            if self._canonical_binding(records[record_id])[2] != task["canonical_input_fingerprint"]:
                return True
        if task["kind"] == "conflict_investigation" and state is not None:
            conflict = state["conflicts"][task["conflict_id"]]
            return conflict["status"] != "contested" or conflict["investigation_task_id"] != task["task_id"]
        return False

    @staticmethod
    def _task(s, task_id):
        text(task_id, "task_id", 160)
        if task_id not in s["tasks"]:
            raise ValueError("unknown task_id")
        return s["tasks"][task_id]

    @staticmethod
    def _rank(s, task):
        attempts = [a for a in s["attempts"] if a["task_id"] == task["task_id"]]
        rejected = {a["strategy"] for a in attempts if a["outcome"] in {"inconclusive", "failed", "blocked"}}
        ranking = []
        for name, strategy in STRATEGIES.items():
            if strategy["context"] != task["context"] or name in rejected:
                continue
            history = [a for a in s["attempts"] if a["context"] == task["context"] and a["strategy"] == name]
            successes = sum(a["outcome"] in {"verified", "counterexample"} and a["verification"].get("recomputed") for a in history)
            failures = sum(a["outcome"] in {"failed", "inconclusive", "blocked"} for a in history)
            score = strategy["base"] + min(successes, 5) * 20 - min(failures, 5) * 2
            reason = "registered method; %d recomputed contextual successes, %d unsuccessful attempts" % (successes, failures)
            if attempts:
                reason += "; different method after " + attempts[-1]["diagnosis"]["code"]
            ranking.append({"strategy": name, "score": score, "reason": reason, "method": strategy["method"]})
        return sorted(ranking, key=lambda r: (-r["score"], r["strategy"]))

    def claim(self, task_id, agent_id, strategy=None, lease_seconds=60):
        text(agent_id, "agent_id", 120)
        if strategy is not None:
            text(strategy, "strategy", 120)
        if type(lease_seconds) not in (int, float) or not math.isfinite(lease_seconds) or not 0 < lease_seconds <= 3600:
            raise ValueError("lease_seconds must be finite and between 0 and 3600")
        with self._transaction():
            s = self._read()
            task = self._task(s, task_id)
            records, _ = self._inputs()
            if self._stale(task, records, s):
                raise ValueError("stale task input; seed a fresh plan binding")
            if any(a["task_id"] == task_id and a["outcome"] in {"verified", "counterexample"} for a in s["attempts"]):
                raise ValueError("task already has a recomputed local result")
            if task["kind"] == "conflict_investigation" and s["conflicts"][task["conflict_id"]]["status"] != "contested":
                raise ValueError("conflict already resolved")
            completed = {a["claim_id"] for a in s["attempts"]}
            active = [c for c in s["claims"].values() if c["task_id"] == task_id and c["claim_id"] not in completed and c["expires_at"] > time.time()]
            if active:
                old = active[-1]
                if old["agent_id"] == agent_id and (strategy is None or strategy == old["strategy"]):
                    return copy.deepcopy(old)
                raise ValueError("task has an active claim lease")
            ranking = self._rank(s, task)
            if not ranking:
                raise ValueError("registered strategies exhausted; owner investigation required")
            strategy = strategy or ranking[0]["strategy"]
            if strategy not in {r["strategy"] for r in ranking}:
                raise ValueError("strategy must be registered, applicable and different from a failed method")
            now = time.time()
            claim = {"claim_id": "claim:" + uuid.uuid4().hex, "task_id": task_id,
                "agent_id": agent_id, "strategy": strategy, "input_fingerprint": task["input_fingerprint"],
                "intended_question": task["intended_question"], "scope": task["scope"],
                "claimed_at": now, "expires_at": now + lease_seconds,
                "selection_reason": next(r["reason"] for r in ranking if r["strategy"] == strategy)}
            self._append("TASK_CLAIMED", claim)
            return copy.deepcopy(claim)

    @staticmethod
    def _execute(task, strategy):
        base = {"recomputed": False, "full_theorem_certificate": False, "scope": task["scope"]}
        if task["context"] != "rational_quadratic":
            return dict(base, limitation="No automated general theorem checker for this scope; submitted notes remain unverified.")
        a, b, c = map(Fraction, task["scope"]["coefficients"])
        if strategy == "rational_sampling":
            points = [Fraction(0), Fraction(2)]
            values = [a*x*x+b*x+c for x in points]
            return dict(base, recomputed=True, actual_outcome="inconclusive", points=list(map(str, points)),
                values=list(map(str, values)), limitation="A finite sample cannot discharge a universal rational quantifier.")
        if strategy == "exact_square_completion":
            vertex = -b / (2*a)
            minimum = c-b*b/(4*a)
            # Re-expand the identity and evaluate the witness independently.
            if (a, -2*a*vertex, a*vertex*vertex+minimum) != (a, b, c):
                raise ValueError("square-completion identity failed")
            witness_value = a*vertex*vertex+b*vertex+c
            if witness_value != minimum:
                raise ValueError("witness recomputation failed")
            return dict(base, recomputed=True, actual_outcome="verified" if a > 0 and minimum >= 0 else "counterexample",
                identity={"a": str(a), "shift": str(vertex), "remainder": str(minimum)},
                vertex=str(vertex), minimum=str(minimum),
                counterexample=None if minimum >= 0 else {"x": str(vertex), "value": str(witness_value)},
                limitation="Exact check of this registered rational quadratic only; no UL theorem certificate.")
        raise ValueError("unregistered executable strategy")

    def record_attempt(self, task_id, agent_id, strategy, outcome, diagnosis,
                       artifacts=None, *, claim_id=None, input_fingerprint=None, scope=None):
        text(agent_id, "agent_id", 120)
        text(strategy, "strategy", 120)
        if not isinstance(outcome, str) or outcome not in {"inconclusive", "failed", "blocked", "verified", "counterexample", "success"}:
            raise ValueError("invalid outcome")
        diagnosis = {"code": diagnosis} if isinstance(diagnosis, str) else bounded(diagnosis, "diagnosis")
        if not isinstance(diagnosis, dict) or diagnosis.get("code") not in DIAGNOSES:
            raise ValueError("diagnosis requires a registered structural code")
        notes = bounded({} if artifacts is None else artifacts, "artifacts")
        with self._transaction():
            s = self._read()
            task = self._task(s, task_id)
            if not isinstance(claim_id, str) or claim_id not in s["claims"]:
                raise ValueError("a valid claim_id is required")
            claim = s["claims"][claim_id]
            if (claim["task_id"], claim["agent_id"], claim["strategy"]) != (task_id, agent_id, strategy):
                raise ValueError("claim owner, task or strategy mismatch")
            if claim["expires_at"] <= time.time() or any(a["claim_id"] == claim_id for a in s["attempts"]):
                raise ValueError("claim lease expired or already consumed")
            if input_fingerprint != task["input_fingerprint"] or input_fingerprint != claim["input_fingerprint"]:
                raise ValueError("stale input fingerprint")
            records, _ = self._inputs()
            if self._stale(task, records, s):
                raise ValueError("stale result: canonical inputs changed")
            if scope is not None and bounded(scope, "scope") != task["scope"]:
                raise ValueError("unapproved scope change; propose_scope instead")
            verification = self._execute(task, strategy)
            actual = verification.get("actual_outcome")
            if actual is not None and outcome != actual:
                raise ValueError("reported outcome differs from recomputed outcome: " + actual)
            if actual is None and outcome in {"verified", "counterexample", "success"}:
                raise ValueError("unverified outcome: no scope-aligned executable checker")
            if actual == "inconclusive":
                diagnosis = dict(diagnosis, code="sample_coverage")
            elif actual in {"verified", "counterexample"}:
                diagnosis = dict(diagnosis, code="none")
            attempt = {"attempt_id": "attempt:" + uuid.uuid4().hex, "task_id": task_id,
                "agent_id": agent_id, "claim_id": claim_id, "strategy": strategy, "context": task["context"],
                "outcome": outcome, "diagnosis": diagnosis, "verification": verification,
                "artifacts": notes, "input_fingerprint": input_fingerprint, "scope": task["scope"],
                "intended_question": task["intended_question"], "recorded_at": time.time(),
                "established": False, "full_theorem_certificate": False}
            self._append("ATTEMPT_RECORDED", attempt)
            return copy.deepcopy(attempt)

    def add_assertion(self, task_id, source_id, proposition, value, scope, evidence=None):
        text(source_id, "source_id", 160)
        text(proposition, "proposition", 12000)
        if type(value) is not bool:
            raise ValueError("assertion value must be an explicit boolean")
        scope, evidence = bounded(scope, "scope"), evidence_list(evidence)
        with self._transaction():
            s = self._read()
            task = self._task(s, task_id)
            if task["kind"] == "conflict_investigation":
                raise ValueError("assertions must bind an original research question, not a conflict investigation task")
            if scope != task["scope"]:
                raise ValueError("assertion scope differs from exact task scope")
            records, _ = self._inputs()
            if self._stale(task, records, s):
                raise ValueError("stale task assertion")
            assertion = {"assertion_id": "assertion:" + uuid.uuid4().hex, "task_id": task_id,
                "source_id": source_id, "proposition": proposition, "value": value, "scope": scope,
                "evidence": evidence, "input_fingerprint": task["input_fingerprint"],
                "established": False, "status": "unverified_source_assertion"}
            # Opposition is meaningful only within one immutable input version.
            # Preserve historical groups instead of pulling stale assertions
            # into a newly restaged conflict with the same proposition/scope.
            members = [a for a in s["assertions"].values() if a["proposition"] == proposition
                       and a["scope"] == scope and a["input_fingerprint"] == task["input_fingerprint"]]
            payload = {"assertion": assertion}
            if any(a["value"] != value for a in members):
                conflict_id = "conflict:" + digest({"proposition": proposition, "scope": scope,
                    "input_fingerprint": task["input_fingerprint"]})[:24]
                member_ids = sorted([a["assertion_id"] for a in members] + [assertion["assertion_id"]])
                previous = s["conflicts"].get(conflict_id)
                conflict = {"conflict_id": conflict_id, "proposition": proposition, "scope": scope,
                    "input_fingerprint": task["input_fingerprint"],
                    "assertion_ids": member_ids, "status": "contested", "established": False,
                    "previous_resolutions": ([] if not previous else previous.get("previous_resolutions", []) +
                        ([previous["resolution"]] if previous.get("resolution") else []))}
                investigation = {"task_id": "investigate:" + digest(member_ids)[:24], "kind": "conflict_investigation",
                    "context": "conflict", "title": "Resolve opposing scoped assertions",
                    "intended_question": "Investigate opposing assertions of the exact proposition: " + proposition,
                    "scope": scope, "input_fingerprint": digest({"assertions": member_ids, "scope": scope}),
                    "dependencies": [], "conflict_id": conflict_id, "status": "open", "full_theorem_certificate": False}
                if scope.get("kind") == "canonical_record":
                    investigation["canonical_input_fingerprint"] = self._canonical_binding(records[scope["record_id"]])[2]
                if task.get("bound_record_fingerprints"):
                    investigation["bound_record_fingerprints"] = copy.deepcopy(task["bound_record_fingerprints"])
                conflict["investigation_task_id"] = investigation["task_id"]
                payload.update(conflict=conflict, investigation=investigation)
            self._append("ASSERTION_ADDED", payload)
            return copy.deepcopy(assertion)

    def resolve_conflict(self, conflict_id, accepted_assertion_id, actor, rationale, evidence=None):
        text(conflict_id, "conflict_id", 160)
        text(accepted_assertion_id, "accepted_assertion_id", 160)
        text(actor, "actor", 120)
        if not actor.startswith(("owner:", "human:")) or not actor.split(":", 1)[1].strip():
            raise ValueError("explicit owner or human authority required")
        text(rationale, "resolution rationale", 12000)
        evidence = evidence_list(evidence, required=True)
        with self._transaction():
            s = self._read()
            conflict = s["conflicts"].get(conflict_id)
            if not conflict or conflict["status"] != "contested":
                raise ValueError("conflict missing or already resolved")
            if accepted_assertion_id not in conflict["assertion_ids"]:
                raise ValueError("accepted assertion must belong to this exact conflict")
            records, _ = self._inputs()
            if any(self._stale(s["tasks"][s["assertions"][aid]["task_id"]], records, s)
                   for aid in conflict["assertion_ids"]):
                raise ValueError("stale conflict input; restage assertions against current canonical bindings")
            decision = {"conflict_id": conflict_id, "accepted_assertion_id": accepted_assertion_id,
                "actor": actor, "rationale": rationale, "evidence": evidence,
                "status": "resolved_by_owner", "established": False, "resolved_at": time.time()}
            self._append("CONFLICT_RESOLVED", decision)
            return copy.deepcopy(decision)

    def propose_scope(self, task_id, proposed_scope, actor, rationale):
        text(actor, "actor", 120)
        text(rationale, "scope rationale", 2000)
        proposed_scope = bounded(proposed_scope, "proposed_scope")
        with self._transaction():
            s = self._read()
            task = self._task(s, task_id)
            if proposed_scope == task["scope"]:
                raise ValueError("scope proposal must change scope")
            proposal = {"proposal_id": "scope:" + uuid.uuid4().hex, "task_id": task_id,
                "proposed_scope": proposed_scope, "original_scope": task["scope"],
                "actor": actor, "rationale": rationale, "status": "awaiting_owner_review"}
            self._append("SCOPE_PROPOSED", proposal)
            return copy.deepcopy(proposal)

    def snapshot(self):
        with self._transaction():
            s = self._read()
            records, _ = self._inputs()
            tasks = list(s["tasks"].values())
            by_record = {t["record_id"]: t for t in tasks if t["kind"] == "canonical" and not self._stale(t, records)}
            target_depth = {}
            def visit(rid, depth):
                if depth <= target_depth.get(rid, -1):
                    return
                target_depth[rid] = depth
                if rid in by_record:
                    for parent in by_record[rid]["premise_record_ids"]:
                        visit(parent, depth + 1)
            for target in s["target_record_ids"]:
                visit(target, 0)
            done_claims = {a["claim_id"] for a in s["attempts"]}
            done_tasks = {a["task_id"] for a in s["attempts"] if a["outcome"] in {"verified", "counterexample"}}
            claims = list(s["claims"].values())
            now = time.time()
            for claim in claims:
                claim["status"] = ("consumed" if claim["claim_id"] in done_claims else
                    "stale" if self._stale(s["tasks"][claim["task_id"]], records, s) else
                    "expired" if claim["expires_at"] <= now else "active")
            active = {c["task_id"] for c in claims if c["status"] == "active"}
            recommendations = []
            for task in tasks:
                stale = self._stale(task, records, s)
                task["stale"] = stale
                task["dependencies_open"] = [d for d in task["dependencies"] if d not in done_tasks]
                task["priority"] = 100 + 10 * target_depth[task["record_id"]] if task.get("record_id") in target_depth else task.get("base_priority", 10)
                if task["kind"] == "conflict_investigation":
                    conflict = s["conflicts"][task["conflict_id"]]
                    task["priority"] = 600
                    resolved = conflict["status"] == "resolved_by_owner"
                    superseded = conflict["investigation_task_id"] != task["task_id"]
                    task["status"] = "resolved_by_owner" if resolved else "superseded" if superseded else "open"
                task["status"] = ("stale" if stale else "diagnostic_completed" if task["task_id"] in done_tasks else
                    "claimed" if task["task_id"] in active else task["status"])
                ranking = self._rank(s, task)
                task["strategy_ranking"] = ranking
                if not ranking and task["status"] == "open":
                    task["status"] = "strategies_exhausted_owner_review"
                if task["status"] == "open" and ranking:
                    recommendations.append({"task_id": task["task_id"], "priority": task["priority"],
                        "strategy": ranking[0]["strategy"], "reason": task.get("priority_reason", "") + (" " if task.get("priority_reason") else "") + ranking[0]["reason"],
                        "dependencies_open": task["dependencies_open"], "intended_question": task["intended_question"]})
            for conflict in s["conflicts"].values():
                for aid in conflict["assertion_ids"]:
                    assertion = s["assertions"][aid]
                    assertion["status"] = ("contested" if conflict["status"] == "contested" else
                        "owner_accepted_unverified" if aid == conflict["resolution"]["accepted_assertion_id"] else "owner_rejected_preserved")
            recommendations.sort(key=lambda r: (-r["priority"], bool(r["dependencies_open"]), r["task_id"]))
            return {"version": "research-director-1", "tasks": sorted(tasks, key=lambda t: (-t["priority"], t["task_id"])),
                "claims": claims, "attempts": s["attempts"], "assertions": list(s["assertions"].values()),
                "conflicts": list(s["conflicts"].values()), "recommendations": recommendations,
                "scope_proposals": s["scope_proposals"], "target_record_ids": s["target_record_ids"],
                "project_targets": [{"task_id": t["task_id"], "milestone_id": t["milestone_id"],
                    "title": t["title"], "intended_question": t["intended_question"], "scope": t["scope"],
                    "priority": t["priority"], "priority_reason": t["priority_reason"],
                    "status": t["status"], "bound_record_ids": t["bound_record_ids"]}
                    for t in tasks if t["kind"] == "project_milestone" and not t["stale"]],
                "events": copy.deepcopy(self.journal.events[-120:]), "last_sequence": len(self.journal.events),
                "journal_head_sha256": self.journal.events[-1]["event_hash"] if self.journal.events else "0"*64,
                "full_theorem_certificates": 0, "policy": {"learning": "journal_history_ranking_not_model_training",
                "strategies": copy.deepcopy(STRATEGIES), "diagnosis_codes": sorted(DIAGNOSES),
                "dependency_semantics": "Necessary research prerequisites; auditing them cannot establish the conclusion.",
                "claim_semantics": "One active lease per task across local POSIX processes; same-owner retries return the existing lease.",
                "authority_semantics": "Local API trusts explicit owner:/human: actors; no multiuser authentication provided."}}
