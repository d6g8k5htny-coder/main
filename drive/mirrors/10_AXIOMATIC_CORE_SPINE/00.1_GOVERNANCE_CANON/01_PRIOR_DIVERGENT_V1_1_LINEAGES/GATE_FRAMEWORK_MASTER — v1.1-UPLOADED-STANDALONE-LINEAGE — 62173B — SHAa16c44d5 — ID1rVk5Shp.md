# THE GATE FRAMEWORK — MASTER GATE FILE (Standalone Edition)

**A typed, graded discipline for rigorous reasoning, and a hardening layer for AI-assisted mathematics.**

**Version:** Master v1.1 (standalone) · **Date:** 2026-07-17 · **Supersedes:** Master v1.0 (2026-07-17), retained and marked per §7 · **Status:** current unified set — **13 established gates + 1 schema field**, plus 2 candidate gates staged pending retrodiction (§11, explicitly *not* counted in the established set).

> **Standalone notice.** This document is self-contained. A reader with no prior context — human or AI model — can adopt and apply the framework from this file alone. Domain-specific examples are drawn from a Gaussian-random-field (GRF) proof program; §8.2 adds a second worked example in a software/empirical domain to demonstrate the abstraction transfers. A glossary (§10) defines every specialist term. Another model may treat this as a complete specification and apply it immediately.

> **What changed from v1.0 (summary; full changelog in Appendix A).** The framework's guarantees are now stated honestly as a **type system for claims** rather than as blanket "mechanical verification": each gate splits into a mechanical *shell* (tag-consistency) and a non-mechanical *core* (a proof obligation), and the gates certify tag-*consistency*, not tag-to-content *fidelity* (§0.5). Bridges, witnesses, and transfer certificates are promoted from free-text fields to **first-class graded nodes** so the constitutional rule bites on them (§3.8) — closing the largest hole in v1.0. The division of labor between the gates and the operating disciplines is made explicit: **gates classify; disciplines detect** (§0.6). Four smaller defects in v1.0 are repaired: the exclusion of *Derived* from the dependency set is resolved into a stated dependability ordering (§3.1); PROVISIONAL is given general grade-ceiling semantics and its dangling reference removed (§3.1); the retrodiction protocol now runs COMMON-MODE on its own evidence (§6); and the content-hash field is made active (§7, §11).

---

## 0. What this is, and how to use it

### 0.1 The problem it solves
AI-assisted reasoning (and human reasoning under time pressure) fails in a recognizable profile: citations that do not exist or do not say what they are claimed to say; silent conflation of different *kinds* of claim; elegant arguments that are load-bearing and wrong; numbers presented as measurements that were never measured. These failures are individually plausible and collectively fatal. The gate framework is a set of checkable rules that catch this failure profile *before* a claim is trusted or promoted.

### 0.2 The unit of operation
The framework operates on **claims** — the atomic assertions (nodes, lemmas, positions, results, estimates) in a reasoning artifact. Every claim is written as a structured **proof object** (§2) carrying explicit metadata. The gates (§4) are predicates over proof objects and the dependency graph they form.

### 0.3 The core discipline, in one sentence
> **Name the restriction, gate the drop.**

Every claim is established *under restrictions* — a fixed regime, a specific measure, a single side, an independence assumption, a finite parameter range, a charted region. A claim fails when a restriction under which it was established is silently dropped when it is reused. Each gate guards one restriction axis and forbids its silent drop. This is the organizing principle of the set (§1) and the rule for extending it (§6).

### 0.4 How to apply it
1. Write every externally visible claim as a proof object with all fields populated (§2).
2. Assign each claim an **epistemic grade** and record its dependencies — including bridge/witness dependencies as nodes (§3.1, §3.8).
3. Run **all** gates over the claim and its dependency cone (§4). A gate returns PASS, FAIL, or PROVISIONAL.
4. A claim may be **promoted** (used as a dependency, or published at its stated grade) **only if every gate passes.** A FAIL blocks promotion until repaired; a PROVISIONAL imposes a grade ceiling (§3.1).
5. Build thresholds, bands, and estimators only as the gate-design standard permits (§5).
6. Preserve every failure (§7). Corrections supersede; they never overwrite.
7. To add a gate, use the retrodiction protocol (§6) — never a constructed or forward-looking score.

### 0.5 What the gates actually guarantee — the shell/core split *(new in v1.1)*
The proof object (§2) is a **type**. The gates (§4) are **typing rules**. Every substantive gate has two layers:

- **Shell (mechanical, decidable):** checks that the required tags are *present* and that the tags on a dependency edge are *compatible* with the tags on the parent. This is decidable by inspection of the graph — a machine can do it.
- **Core (a proof obligation, not mechanical):** the mathematical content the tags *assert*. Whether the stated change-of-measure is *valid*; whether the named composition witness *holds*; whether the failure-region enumeration is *complete*. This is not decidable by inspection; it is discharged by a proof, or by the operating disciplines of §7.

The consequence, stated plainly so a sophisticated reader hits it up front rather than deriving it and losing trust:

> **The gates certify tag-*consistency*, not tag-to-content *fidelity*.**

Mis-tag a lower bound as an upper bound and POLARITY will happily check the wrong tag against other wrong tags and PASS. Tag a dependent product "Independence" when the events are correlated and COMPOSITION's shell PASSes. **Fidelity — that the tags match the actual mathematics — is not produced by any gate.** It is produced only by §7 (adversarial reimplementation, separation of duties, dual-track verification). Therefore every "mechanical" gate is *downstream of a non-mechanical tagging step*, and the framework's guarantees are conditional on that step being honest and competent.

This framing is deliberately more modest than v1.0's "mechanically checkable," and it is stronger for being modest: (a) it survives adversarial reading, because it concedes the exact point an auditor would otherwise use to discredit the whole set; (b) it connects the framework to proof-assistant machinery (a system like Lean or Coq typechecks the shell and discharges the core with tactics/proofs), which has thought carefully about exactly this boundary; and (c) it tells the adopter *where to spend vigilance* — the shell is free, so effort belongs on the tagging step and on the cores.

### 0.6 Gates classify; disciplines detect *(new in v1.1)*
A related honesty. The framework's two machines do different jobs, and conflating them overstates the gates:

- **The §7 operating disciplines *detect* content errors.** A recomputation under separated duties that produces a different number is what *catches* a numerical/content mistake. The two flagship GRF catches — a conditioning coefficient off by nearly a factor of two, and a corridor probability off by up to ~3.35× — were, in mechanism, caught by *recomputation seeing a discrepancy*, not by a gate firing on the claim as written.
- **The §4 gates *classify* every error by restriction axis, and *detect* the structural/transfer errors that live *inside a correctly-tagged step*** — a domain gap, a reachable killed node, a witness-free product, a model invoked outside its transfer scope. These are errors that a single-step recomputation would not reveal, because the arithmetic of the step can be *correct* while the step is being *used* outside where it holds.

Both are essential and they are not redundant. But when reporting efficacy, attribute each catch to the machine that actually made it. Classification is not a lesser role: it is what builds the *frozen, typed failure record* that trains attention and powers the retrodiction protocol (§6). A number caught and then correctly filed as (say) a MEASURE failure is what lets the next MEASURE error be anticipated.

---

## 1. The meta-rule: restriction-drop

The gates are not independent inventions. They are instances of a single shape:

> **A property established under a restriction must not be used after the restriction is silently dropped. Either carry the restriction, or supply an explicit bridge that licenses its removal — and, per §3.8, that bridge is itself a graded node.**

Each gate guards one **restriction axis**:

| gate | restriction axis it guards |
|---|---|
| POLARITY | sidedness (upper / lower / two-sided) |
| MEASURE | base measure / conditioning law |
| MARK | mark / type carried by the estimate |
| MODEL | the exact probabilistic model (e.g. covariance) |
| RUNG | the scale a measured value lives at |
| COMPOSITION | statistical independence (or the witness that licenses combining) |
| DOMAIN | the parameter range of validity |
| ENDPOINT | verification at the boundary of a declared range |
| COVERAGE | the region actually charted by a detector |
| PRECEDENCE | live-vs-dead (killed / superseded) status |
| CORE CLOSURE | finished-vs-open status |
| COMMON-MODE | independence of the *checking instruments'* error channels |
| HEURISTIC-BRIDGE | rigor mode (heuristic/analogy vs. derivation) |

**Two orthogonal lenses.** The restriction-axis table above says *what each gate guards*. The shell/core split (§0.5) says *how much of each gate is mechanical*. They are independent readings of the same set — a gate can guard a clean restriction axis while having a large non-mechanical core (COVERAGE is the extreme: its axis is crisp, but its core — enumerating every failure region — is the undecidable heart of the whole verification problem, which is exactly why COVERAGE exists only because a region was once missed).

**Why this framing matters.** It makes the set memorable, teachable, extensible, and diagnosable: to audit a claim, enumerate the restrictions its derivation actually relied on, and check each drop.

**Honesty on the framing.** The fit is clean for most gates and loose for a few (PRECEDENCE, CORE CLOSURE, and ENDPOINT read as restriction-drop only by stretch). Treat §1 as an **organizing lens**, not a theorem.

---

## 2. The proof object (claim schema)

Every externally visible claim carries these fields. A missing field is itself a gate failure (a claim that cannot state its own metadata cannot be trusted).

| field | meaning |
|---|---|
| **atomic statement** | the single assertion, stated without hedging or bundling |
| **dependency IDs** | the claims this one rests on (edges into the dependency graph). **Each edge is pinned to the content hash of the dependency at the time of the check** (§7 hash-drift discipline). |
| **evidence IDs** | the computations, proofs, or measurements that support it |
| **bridge/witness IDs** *(elevated in v1.1)* | pointers to the graded nodes that license any change-of-measure, composition, mark-transfer, or model-transfer this claim relies on (§3.8). These are **dependency IDs**, not free text. |
| **model tag** | the exact probabilistic/mathematical model assumed (§3.3) |
| **measure tag** | the base measure / conditioning law (§3.2) |
| **parameter domain** | the range of every free parameter over which the claim holds |
| **required marks / supplied marks** | the type/mark variables the claim *needs* vs. the ones its evidence *carries* (§3.4) |
| **quantifier** | ∀ / ∃ / "for typical" — stated, not implied, **including quantifier order** where two quantifiers interact (see §11 uniformity candidate) |
| **direction** | lower bound, upper bound, or equality/two-sided |
| **epistemic grade** | position on the warrant ladder (§3.1) |
| **uncertainty side** | for a numerical band: one-sided (+ or −) or symmetric, with justification |
| **scale / rung tag** | the scale a measured value is expressed in (§3.5) |
| **composition witness** | if the claim combines sub-claims, the **node ID** of the named license for combining (§3.6, §3.8) |
| **source-precedence ID** | which version/supersession lineage this claim belongs to |
| **content hash** | a hash of the claim's exact text/data. **Active in v1.1:** it pins dependency edges so that editing a dependency invalidates ("staleness") every PASS that stood on the old text (§7, §11). |
| **supersession status** | live / superseded / killed / retired |
| **regime-of-validity** *(field F-1)* | the tuple (base measure, spatial region, scale, parameter box, conditioning depth) under which the claim/mechanism/constant was *established* — so transfer gates have an explicit deployment target to check against (§3.7) |

---

## 3. Controlled lexicons

Gates are only as sharp as the vocabularies they check against. These lexicons are mandatory and closed (extend only with an explicit, consistently applied definition).

### 3.1 Epistemic grade ladder (warrant axis), dependability, and PROVISIONAL
From strongest to weakest **warrant**:

1. **Primitive** — a declared assumption (recorded in an assumption ledger). Not proved; explicitly assumed.
2. **Proven** — proved within the artifact (referee-grade), or a self-contained theorem package.
3. **Proven-Modulo** — proved *given* named conditional dependencies; the conditions are explicit and finite.
4. **Derived** — calculation or bound executed, not yet packaged to referee grade.
5. **Literature-Supported** — supported by cited external work; may require adaptation.
6. **Plausible** — coherent and mechanism-supported, but not proof-complete.
7. **Conjecture** — believed, not proved.
8. **Open** — explicitly unresolved.
9. **Killed** — false, invalid, superseded, or unusable.

**Warrant vs. dependability (resolves the v1.0 "Derived gap").** The ladder orders *warrant* (how well-established). Whether a claim may be **depended upon** tracks a different property — *scrutiny survived* — and the two axes do not coincide. Define three dependability tiers:

- **Fully dependable** (may support a claim of *any* grade, including the top): **Primitive** (declares content, so asserts nothing to scrutinize), **Proven** and **Proven-Modulo** (survived internal adversarial reimplementation under separated adjudication), **Literature-Supported** (survived external refereeing).
- **Self-tier dependable** (may support only claims graded ≤ itself): **Derived**. A Derived result has been *executed* but not yet independently scrutinized, so it cannot lift a parent above Derived — but Derived-on-Derived is permitted, because that is the honest state of live calculation mid-program. A Derived step becomes fully dependable by passing adversarial reimplementation (§7), at which point it is regraded Proven / Proven-Modulo.
- **Not dependable**: **Plausible, Conjecture, Open, Killed.**

This is why the permitted-dependency set for a claim graded Proven or above is exactly **{Primitive, Proven, Proven-Modulo, Literature-Supported}** and *excludes* Derived: that set is the *fully-dependable* tier, not the top of the warrant ladder. Derived's absence from it in v1.0 was correct; what was missing — now supplied — is the explicit statement that Derived is *self-tier* dependable, so building Derived-on-Derived is legal while nothing above Derived may launder through it.

**Constitutional dependency rule (binding).** A claim may depend only on objects **at least as dependable as the claim's own grade requires**, per the tiers above. It may **never silently** depend on Plausible, Conjecture, Open, or Killed. If it depends on an Open or Conjecture object, the claim **must itself be graded Proven-Modulo**, making the conditional dependence explicit. A Killed object may never re-enter the dependency graph unless repaired and relabeled. This single rule prevents hidden circularity and silent dependence on dead mathematics — and, via §3.8, it now also governs every bridge and witness.

**PROVISIONAL — general semantics (repairs the v1.0 dangling reference).** PROVISIONAL is a third verdict any gate *may* emit; in the current set only **COMMON-MODE** and **HEURISTIC-BRIDGE** emit it. A PROVISIONAL verdict attaches a **grade ceiling** to the dependent claim — the claim may hold at or below the ceiling but may not promote past it until the provisional condition is discharged:

- **COMMON-MODE PROVISIONAL** → ceiling = *the claim's current grade* (no upward promotion) until an error-independence argument or an external cross-check is supplied.
- **HEURISTIC-BRIDGE PROVISIONAL** → ceiling = **Plausible** until the heuristic step is bridged.

(There is no `§4.12–4.13`; v1.0's pointer to those subsections was dangling and is removed. PROVISIONAL is defined here and nowhere else.)

**Note on axes.** Warrant is distinct from *reasoning type* (empirical / theoretical / analogical / normative). Where the two interact — an analogical step becoming load-bearing — see gate 13 (HEURISTIC-BRIDGE).

### 3.2 Measure / law-type lexicon
The conditioning law under which a probabilistic quantity is computed. These may not be silently interchanged (gate MEASURE):
- **Unconditioned** — the plain law.
- **Gaussian-pinned** — conditioned on field values (and/or derivatives) at specified points.
- **Typed-pinned** — conditioned on critical points of a specified index/type.
- **Pair-Palm** — the Palm law of a *pair* point process (conditioned on the existence of a configured pair at specified locations).

*(A general adopter substitutes the conditioning laws native to their domain; the rule — do not interchange without a bridge, and the bridge is a node — is invariant.)*

### 3.3 Model lexicon
The exact model a claim assumes must be named (gate MODEL): the specific kernel/covariance, distribution family, or generative model — or, if a claim is transported from one model to another, an explicit **transfer certificate** bounding the error of the transport, valid within a declared scope. Per §3.8, the transfer certificate is a **graded node**, not an attestation.

### 3.4 Mark / type lexicon
A **mark** is auxiliary data attached to a point/event (e.g. height/value, critical-point index/type). An estimate is **marked** if it controls the marked quantity and **unmarked** if it controls only the underlying (e.g. spatial) quantity. An unmarked estimate cannot discharge a marked obligation (gate MARK). A mark-transfer — the argument that an unmarked estimate suffices for a marked target — is a **graded node** (§3.8).

### 3.5 Scale / rung lexicon
Every measured value carries the scale it is expressed in (e.g. absolute units, or units of a small parameter such as r, r², ℓ). A measured number with no scale tag is uninterpretable and may not be compared or composed (gate RUNG).

### 3.6 Composition-witness lexicon
When a claim combines sub-claims (a product of probabilities, a union bound, an assembly), the license must be one of these **named** witnesses — and, per §3.8, the witness is a **graded node** whose own grade is subject to the constitutional rule:
- **Independence** — the events are independent.
- **Conditional independence** — independent given a stated conditioning.
- **Markov property** — a stated Markov structure.
- **Negative dependence / negative association** — a stated inequality direction.
- **Comparison theorem** — a named inequality (e.g. Slepian, FKG, Gaussian comparison).
- **Joint certificate** — a direct bound on the joint object.
- **Exact algebra** — the combination is an identity, not an approximation.

A product/union with **no** witness node is forbidden. A product/union whose witness node is graded only Plausible (or blank) cannot support a parent graded above Plausible — this is the constitutional rule (§3.1) doing the work that free text could not.

### 3.7 Regime-of-validity (field F-1)
The tuple recording where a claim/constant/mechanism was **established**: base measure, spatial region/scale, parameter box, conditioning depth. Its purpose is to give the transfer gates (MEASURE, DOMAIN, MODEL, MARK) an explicit target: a deployment site is checked against the establishment regime, and a mismatch is flagged. This is a **schema field, not a gate** — the minimal fix for the most frequent transfer-error shape without blurring the four sharp transfer gates into one vague one.

### 3.8 Bridges are nodes, not attestations *(new in v1.1 — the largest v1.0 hole closed)*
In v1.0, a composition witness, a change-of-measure, a mark-transfer, and a model-transfer certificate were **free-text schema fields**. This was an escape hatch that bypassed the grade ladder entirely: an author could write a product of correlated probabilities, type the word "Independence," and COMPOSITION would PASS on the presence of a listed witness. The recorded ~3.35× corridor error is *exactly what an author who wrongly believed the events independent would write* — the v1.0 gate caught the *honesty* gap (a blank witness) but did nothing about the *competence* gap (a wrong witness), which is the more dangerous of the two and, on the evidence, the actual mechanism of that miss.

**Rule.** Every bridge — every composition witness, change-of-measure, mark-transfer, and model-transfer certificate — is a **first-class node in the dependency graph with its own epistemic grade**, cited by node ID from the parent's `bridge/witness IDs`. It is not discharged by naming; it is discharged by grading, and its grade is then subject to the constitutional dependency rule (§3.1).

**Consequence.** "Independence" stops being a word and becomes a pointer to a sub-claim that must itself be graded Proven / Proven-Modulo / Literature-Supported to lift a Proven-Modulo parent. A change-of-measure that is only Plausible caps the claim it bridges at Plausible. This closes the single place in v1.0 where unproven content could launder past the grade ladder, and it costs nothing already unbuilt: bridges were always *stated*; they are now *graded*.

**Shell/core note.** Promoting a bridge to a node moves the bridge's *core* (is the change-of-measure valid?) into the ordinary machinery of grading and §7 verification, and leaves the parent gate with a clean *shell* check: *is a bridge node cited, and is its grade sufficient?* That shell is mechanical; the bridge's core is where the mathematics lives.

---

## 4. The gates (complete, uncompressed)

Each gate is stated as: **Guards** (restriction axis) · **Rule** · **Shell (mechanical) / Core (obligation)** · **PASS / FAIL[/PROVISIONAL]** · **Canonical failure prevented** · **Notes**. Gates 1–11 are the established set; gates 12–13 are added 2026-07-17 with retrodiction evidence in the appendix.

---

### Gate 1 — DOMAIN
- **Guards:** the parameter range of validity.
- **Rule:** a dependency must cover the complete parameter domain of the claim that invokes it.
- **Shell:** compare declared domain intervals across the edge. **Core:** confirm the dependency's *stated* domain is one on which it was actually established (not merely asserted).
- **PASS / FAIL:** PASS iff, for every free parameter, the dependency's domain ⊇ the parent's domain. FAIL if the parent is asserted on a range the dependency does not cover.
- **Canonical failure prevented:** a lemma proved on a sub-interval used to support a claim asserted on a wider interval.
- **Notes:** domain containment is checked with an explicit tolerance; do not rely on "obviously covers." DOMAIN checks *in-range*, not *uniform-over-range* — the latter is the §11 uniformity candidate.

### Gate 2 — ENDPOINT
- **Guards:** verification at the boundary of a declared range.
- **Rule:** every registered endpoint of a claimed finite-range bound must satisfy that bound.
- **Shell:** evaluate the claim at each registered endpoint. **Core:** confirm the *registered* endpoints are the actual extrema of interest (an interior extremum is out of ENDPOINT's scope — see §11).
- **PASS / FAIL:** PASS iff evaluating the claim at each registered endpoint satisfies the stated inequality. FAIL if any endpoint violates it.
- **Canonical failure prevented:** asserting "≤ 0.97 on this range" whose range secretly contains a point where the value is 0.985.
- **Notes:** endpoints are the cheapest falsifiers of a range claim; check them first.

### Gate 3 — POLARITY
- **Guards:** sidedness.
- **Rule:** a lower-bound claim may consume only lower one-sided inputs; an upper-bound claim only upper one-sided inputs; any uncertainty band attached to a one-sided claim must itself be one-sided in the matching direction.
- **Shell:** match `direction` and `uncertainty side` tags across the edge. **Core:** confirm the tags are *faithful* — this gate is the clearest illustration of §0.5, since a mis-tagged direction passes the shell against other mis-tags.
- **PASS / FAIL:** PASS iff input directions match the claim's direction and every band on a one-sided claim is one-sided (+ for upper, − for lower). FAIL on a symmetric band attached to a one-sided claim, or a lower input feeding an upper claim (or vice versa).
- **Canonical failure prevented:** attaching a symmetric ±ε band to an upper theorem.
- **Notes:** "±" on a one-sided quantity is the most common tell.

### Gate 4 — MEASURE
- **Guards:** base measure / conditioning law.
- **Rule:** unconditioned, Gaussian-pinned, typed-pinned, and pair-Palm claims (or the domain's analogous conditioning laws) may not be silently interchanged. Any interchange must cite a **change-of-measure node** (§3.8).
- **Shell:** match `measure` tags across the edge, or confirm a change-of-measure node is cited and sufficiently graded. **Core:** the validity of the cited change-of-measure (now itself a graded node with its own core).
- **PASS / FAIL:** PASS iff measure tags match, or a sufficiently-graded change-of-measure node bridges them. FAIL on any silent swap of conditioning law, or a bridge node too weakly graded to lift the parent.
- **Canonical failure prevented:** using a coefficient computed under one conditioning where the claim requires another — a real recorded instance had these differ by nearly a factor of two.
- **Notes:** the conditioning law is part of *what the number means*; a bare number hides it. The catch of that factor-of-~2 instance was, mechanistically, a §7 recomputation; MEASURE *classifies* it and, going forward, *prevents* the silent version by demanding the bridge node (§0.6).

### Gate 5 — MARK
- **Guards:** the mark/type carried by the estimate.
- **Rule:** a spatial (unmarked) estimate does not prove a height-marked or type-marked estimate unless the mark variables appear in the dependency cone, or a **mark-transfer node** (§3.8) licenses the step.
- **Shell:** match `required marks` against `supplied marks`, or confirm a mark-transfer node is cited and graded. **Core:** the validity of that transfer.
- **PASS / FAIL:** PASS iff every required mark is supplied by evidence or by a sufficiently-graded mark-transfer node. FAIL if a marked obligation is discharged by unmarked support with no bridge.
- **Canonical failure prevented:** citing spatial repulsion (unmarked) to assert a bound that requires a height-window factor (marked).
- **Notes:** "incomplete source-to-estimand transfer" — the source proves less than the claim needs — is the general disease; MARK is its most common form.

### Gate 6 — COMPOSITION
- **Guards:** statistical independence (or the license to combine).
- **Rule:** a product of probabilities, a union bound, or any assembly of sub-claims requires a **composition-witness node** from the closed list (§3.6, §3.8).
- **Shell:** confirm a witness node is cited for every combination and is sufficiently graded. **Core:** whether the cited witness (independence, comparison theorem, etc.) actually *holds* — the witness's own core.
- **PASS / FAIL:** PASS iff every combination cites a witness node graded high enough to support the parent. FAIL on any witness-free product/union/assembly, or a witness node too weakly graded.
- **Canonical failure prevented:** modeling a multi-checkpoint corridor probability as a product of per-checkpoint probabilities with no dependence structure (a recorded instance was off by up to ~3.35×). Under v1.1 the "Independence" witness would be a node requiring its own grade; a wrong independence claim is now a *weakly-graded or false witness node*, not a passing free-text field.
- **Notes:** independence is a *claim*, not a default; if it cannot be named and graded, it must be bounded.

### Gate 7 — RUNG
- **Guards:** the scale of a measured value.
- **Rule:** every measured value carries an explicit scale/parameter tag.
- **Shell:** confirm a `scale/rung` tag is present on every measured number and matches across any comparison/composition. **Core:** none beyond tag fidelity — RUNG is nearly all shell.
- **PASS / FAIL:** PASS iff each measured number states its scale. FAIL if a value is treated as scale-free or compared across mismatched scales.
- **Canonical failure prevented:** promoting a measured constant to a "truth constant" with no scale, then composing it with quantities at a different scale.
- **Notes:** pairs with F-1 (§3.7): the rung is one component of the full regime-of-validity.

### Gate 8 — PRECEDENCE
- **Guards:** live-vs-dead status.
- **Rule:** a killed or superseded claim may not be reachable from a live root in the dependency graph.
- **Shell:** graph reachability on the `supersession status` field. **Core:** none — PRECEDENCE is fully mechanical.
- **PASS / FAIL:** PASS iff no path from a live theorem reaches a node marked killed/superseded/retired. FAIL on any such reachable path.
- **Canonical failure prevented:** continuing to cite a retired estimand or a superseded intermediate constant.
- **Notes:** enforced by graph reachability; supersede-never-overwrite (§7) keeps the dead nodes present *and* marked. The hash-drift discipline (§7) is PRECEDENCE's natural companion — it catches a dependency that was silently *edited* rather than superseded.

### Gate 9 — CORE CLOSURE
- **Guards:** finished-vs-open status.
- **Rule:** a declared-finished core may not depend on an open extension node.
- **Shell:** check that no node in a closed core has a dependency graded Open (without the constitutional Proven-Modulo relabel). **Core:** none — mechanical given correct grades.
- **PASS / FAIL:** PASS iff no closed-core claim depends on an Open node. FAIL otherwise.
- **Canonical failure prevented:** declaring a result "closed" while it silently rests on an unresolved lemma.
- **Notes:** the mirror of the constitutional rule (§3.1) at the level of whole cores.

### Gate 10 — MODEL
- **Guards:** the exact probabilistic model.
- **Rule:** every model-dependent (e.g. covariance-dependent) claim identifies its exact model, or cites a **transfer-certificate node** (§3.8) valid within a declared scope.
- **Shell:** confirm a `model` tag is present and matches the evidence, or a transfer-certificate node is cited with a scope covering the deployment. **Core:** the certificate's error bound and its scope validity — the certificate's own core.
- **PASS / FAIL:** PASS iff the model tag matches, or a scoped transfer-certificate node bridges them and is used within its scope. FAIL on an unnamed model, or a transfer used outside its certified scope.
- **Canonical failure prevented:** a covariance computation that does not name its kernel; a transfer certified only for a bounded set of orders/dimensions/distances but invoked outside that box (fix: partition the domain and declare any larger use).
- **Notes:** "which model?" must have a single, checkable answer.

### Gate 11 — COVERAGE
- **Guards:** the region actually charted by a detector.
- **Rule:** no aggregate/exponential-rank argument may suppress wrong outputs in a region that activates **no** registered detector (chart). Strength elsewhere does not cover an un-instrumented region.
- **Shell:** confirm each registered failure region maps to ≥1 detector. **Core (large):** *whether the set of registered failure regions is complete* — the undecidable heart of verification, and the reason this gate is mostly core.
- **PASS / FAIL:** PASS iff every region where the claim could fail is covered by at least one detector. FAIL if a strong global argument is used to imply control over a region with no detector.
- **Canonical failure prevented:** a suppression bound exponentially strong on the charted set being read as control over a qualitatively different, un-charted failure region.
- **Notes:** the sibling of gate 12 — COVERAGE asks *"is the region charted at all?"*; COMMON-MODE asks *"are the charts that agree actually independent?"* Because COVERAGE's core is the completeness question itself, treat a COVERAGE PASS as *"no gap found by the current chart set,"* never as *"no gap exists."*

---

### Gate 12 — COMMON-MODE  *(added 2026-07-17)*
- **Guards:** independence of the *checking instruments'* error channels.
- **Rule:** a gate PASS that **rests on agreement between two instruments** (any regression/agreement check comparing two computations) is valid only if the instruments are shown not to share the error channel under test — via either (a) a second construction with a demonstrably disjoint failure mode, or (b) an external contradiction test against an independent artifact. Absent (a) or (b), the PASS is **PROVISIONAL** (§3.1) and the claim it supports may not promote in grade.
- **Shell:** detect that a verdict is agreement-based and check whether an (a)/(b) argument is attached. **Core:** whether the claimed error-channel disjointness *actually holds*.
- **PASS / FAIL / PROVISIONAL:** PASS iff an agreement-based verdict carries an error-independence argument (a) or an external cross-check (b). PROVISIONAL if it rests on agreement alone. Single-instrument exact (arbitrary-precision) checks are **exempt** — they assert no agreement to certify.
- **Canonical failure prevented:** two instruments that share a construction step agree, and the agreement is mistaken for certification, when both carry the same defect — e.g. both include a spurious point inside a search window, or both drop the same coordinate offset and "agree at the wrong station." Recorded instances were caught only by luck or by an external cross-check; this gate makes that cross-check a condition of PASS.
- **Notes:** gates 1–11 constrain the structure of a *claim*; this is the first gate that constrains *verification methodology* — and it is therefore the gate that partially discharges §0.5's worry, by demanding fidelity-checks where a PASS rests on agreement. Scope it strictly to agreement-based verdicts to avoid a blanket "reimplement everything" tax. The dual use is encouraged: when two independent constructions produce the *same wrong* result, the shared error convicts the shared instrument.

### Gate 13 — HEURISTIC-BRIDGE  *(added 2026-07-17)*
- **Guards:** rigor mode (heuristic/analogy vs. derivation).
- **Rule:** any step whose warrant is analogy, structural similarity, or an unexplained symbolic coincidence, and which becomes **load-bearing for a quantitative or asymptotic claim**, must cite an explicit **bridge node** (a derivation of the transition) or accept an explicit **grade downgrade** of the dependent claim. An un-bridged heuristic→quantitative transition may not promote past **Plausible**.
- **Shell:** flag steps tagged analogical/heuristic that feed a claim graded Derived or above, and check for a bridge node. **Core:** whether the bridge node actually derives the transition.
- **PASS / FAIL / PROVISIONAL:** PASS iff each analogical/heuristic step feeding a quantitative claim either cites a sufficient bridge node or the dependent claim is graded ≤ Plausible. PROVISIONAL (ceiling = Plausible) while unbridged. FAIL if an analogical step silently supports a claim graded Derived or higher with no bridge and no downgrade.
- **Canonical failure prevented:** an argument by analogy to a different theory, treated as a derivation and made to carry a quantitative prediction (later falsified by measurement); a symbolic derivation in which required quantities appear "as if by magic" (fix: supply the missing derivation as a bridge node).
- **Notes:** the guard on the axis where *warrant* (§3.1) and *reasoning type* interact. It bites at the generative frontier — where the most expensive errors originate — and its cost (a tag plus a bridge-or-downgrade obligation) is low. With §3.8, its bridge is a graded node like every other.

---

## 5. Gate-design standard (how to build thresholds, bands, and estimators)

Gates and their thresholds are themselves artifacts that can be wrong. These rules (registry-grade) govern their construction:

1. **Every threshold must be DERIVED** — from a physical/structural scale, or from the estimator's sampling distribution. **Never** compare single-instance statistics against arbitrary percent bands, and never use an arbitrary absolute tolerance. *(Scope note, v1.1: this rule governs **measurement/estimator** thresholds at the object level. Discrete **governance** choices — e.g. the "≥2 recorded entries" minimum for admitting a gate, §6 — are not measurement thresholds and are exempted; the exemption is stated and defended there rather than smuggled.)*
2. **Separate exactness from deployment.** Deterministic claims get arbitrary-precision (exact) gates, kept separate from floating-point deployment gates.
3. **Diagnose the instrument before the physics.** When a gate FAILS, first suspect the variable definition and the estimator; interrogate those before concluding a substantive/physical failure. (A recorded case: a first diagnosis of "aliasing" was wrong; the true cause was an under-resolved grid.)
4. **Limit-bands from remainder structure only.** A band on a limiting value must come from the exact remainder/error structure, or be explicitly labeled a *forecast* — never from naive extrapolation of finite-parameter data with competing correction terms.
5. **Independent numerical check before comparison.** A closed-form constant gets an independent numerical evaluation before it is used in any comparison — and, per gate 12, if that check rests on two instruments agreeing, their error-channel disjointness is itself required.

---

## 6. Adding a new gate — the retrodiction protocol

New gates are earned against a **frozen failure record**, never justified by a constructed or forward-looking score. (A forward-looking "improvement percentage" over cases selected to flatter the gate is not evidence; it is arithmetic that follows from the selection.)

**Protocol.** Freeze a record F of past failures (a kill registry / error ledger). A candidate rule R qualifies only if:
1. **R fires on ≥ 2 recorded entries in F that are independent under a COMMON-MODE check** *(strengthened in v1.1)* — it would have flagged them before their actual catch mechanism did, **and** the two entries do not both trace to a single underlying analogy, misconception, or shared instrument. Two kills that are really one mistake counted twice are *not* two witnesses. Run gate 12 on the evidence for a candidate gate exactly as you would on any agreement-based verdict: the candidate's justification *is* an agreement-based verdict (F agrees the rule would have helped), and its witnesses must have disjoint error channels.
2. **R is silent on most of F** — a rule that flags everything is uninformative.
3. **R is not already discharged** by an existing gate or by the gate-design standard.

If any condition fails, **reject R.** Most candidates die on condition 3 — and rejecting them is the point. The number of gates is an **output** of what the failure record demands, not an input.

**On the "≥2" itself (honesty about §5.1).** The threshold in condition 1 is a *governance* minimum — the smallest witness count that is more than anecdote — not a measurement threshold derived from a sampling distribution, and §5.1's derivation requirement is scoped (see §5.1 note) to exclude it. This is a deliberate, stated exemption, not an oversight: the object of condition 1 is admission of a rule into a discrete set, a decision that does not have a sampling distribution to derive against. The COMMON-MODE strengthening in condition 1 is the compensating rigor.

**Binding limitation (survivorship).** Retrodiction tests only *caught* failures. A gate's value against failures that were never detected is unmeasurable this way. Retrodiction can show a rule would have helped on known misses; it cannot prove a rule's full value, nor that the current gap list is complete. State this limitation whenever the framework is presented as validated.

---

## 7. Operating disciplines

The gates presuppose a workflow that preserves state and failures. Per §0.6, **these disciplines are what *detect* content errors; the gates *classify* them.**

- **Freeze-before-execute.** Commit gate specifications and predicted outcomes *before* running the instrument. A prediction written after seeing the result is not a test.
- **Supersede-never-overwrite.** A correction creates a new versioned artifact; the superseded artifact is retained and marked. This is what lets PRECEDENCE (gate 8) enforce reachability on dead nodes. *(This document is itself an instance: v1.1 supersedes v1.0, which is retained and marked, not deleted.)*
- **Hash-drift / stale-PASS discipline** *(new in v1.1, activating the content-hash field).* Every dependency edge is pinned to the content hash of the dependency at the moment the parent's gates were run. If a dependency is later edited — its hash changing from *h* to *h′* without a supersession event — every parent PASS that stood on *h* is marked **stale** and must be re-gated. This catches the "I fixed the lemma but never re-checked what stood on it" failure, which is live for any versioned, multi-document program. (Staging note: promoted to a full gate only once it clears §6 against the kill registry — see §11.)
- **Preserve every failure.** Every kill and every failed-as-written gate is logged permanently in an error ledger (a "kill registry" / E-ledger). Failures are structure, not embarrassment: they are the test set for §6.
- **Ledgers written after adjudication.** Record gate outcomes in a step *separate from and subsequent to* running the gates. A ledger written in the same act as the computation can claim a result before it is inspected.
- **Adversarial reimplementation.** Verify by independent instruments — and, per gate 12, with a demonstrably disjoint error channel where a verdict rests on agreement. **This is the primary detector of numerical/content error; the gates classify what it finds.**
- **Separation of duties.** Distinct roles across generation, verification, and adjudication; dual-track (external citation/source verification in parallel with internal consistency checking). **This is the primary source of tag-to-content fidelity (§0.5), which no gate produces.**
- **Maximize the weakest link.** A claim's grade is bounded by its weakest dependency — now including its weakest *bridge* node (§3.8); a chain is only as exact as its least-exact link. Design for the weakest link, not the average.

---

## 8. Worked examples

### 8.1 GRF (source domain)

**Claim (as written):** "The probability that a configured feature fails to pair with its nearest partner is at most C·r³, for r ∈ (0, r₀] and parameter b ∈ B."

**Proof object (populated):**
- atomic statement: as above.
- direction: upper bound. · quantifier: ∀ r ∈ (0, r₀], ∀ b ∈ B.
- measure tag: pair-Palm. · model tag: [named kernel]. · parameter domain: r ∈ (0, r₀], b ∈ B.
- required marks: height b (marked). · supplied marks: must include b.
- epistemic grade: Proven-Modulo. · dependencies: {inner-region bound, far-region bound, boundary bound}.
- bridge/witness IDs: {change-of-measure node lifting any pinned input to pair-Palm; composition-witness node combining the region bounds}.
- uncertainty side: upper (+) only. · scale/rung: coefficient in units of r³.
- regime-of-validity (F-1): (pair-Palm, near-diagonal region, r-scale, box r≤r₀ & b∈B, full pair conditioning).

**Running the gates (illustrative):**
- **POLARITY** — upper claim, upper inputs, one-sided (+) band → candidate PASS (shell); fidelity of the "upper" tags rests on §7.
- **MEASURE** — every regional input must be pair-Palm. If one input is a *pinned* coefficient, MEASURE **FAILS** unless a change-of-measure *node* is cited and graded high enough to lift a Proven-Modulo parent. *(In v1.0 a free-text note could have passed; v1.1 requires the graded node.)*
- **MARK** — the height b must be carried by the evidence or by a graded mark-transfer node; a spatial-only region bound with no such node → **FAIL**.
- **COMPOSITION** — the three regional bounds combine only via a witness node (union bound with disjoint events, or a comparison theorem), and that node must itself be sufficiently graded. Blank or weakly-graded witness → **FAIL**.
- **DOMAIN** — each regional bound must cover r ∈ (0, r₀] and b ∈ B; a bound proved only for r ≤ r₀/2 → **FAIL**.
- **COMMON-MODE** — if C was certified by two numerical instruments that share a sampling construction, the PASS is **PROVISIONAL** (ceiling = current grade) until an error-independent check or external cross-check is added.
- **HEURISTIC-BRIDGE** — if any regional bound rests on an analogy to another model rather than a derivation, the claim is **PROVISIONAL** (ceiling = Plausible) until the analogy is supplied as a bridge node.

**Verdict.** Promotes to Proven-Modulo only when every gate passes; otherwise capped or blocked at the named gate, and the failure is logged (§7).

### 8.2 Software / empirical (transfer domain) *(new in v1.1)*

This example uses no GRF vocabulary. Its purpose is to show the abstraction is real, not a GRF reskin — directly serving the cold-adoption goal.

**Claim (as written):** "Median request latency is at most 40 ms under load profile L (≤ 2,000 req/s) on service configuration C."

**Mapping the schema to this domain:**
- direction: upper bound. · quantifier: ∀ arrival rates in L up to 2,000 req/s ("for typical" load, *not* worst-case burst — stated, not implied).
- **measure tag** → *conditioning law of the measurement*: warm-cache steady state (analogue of a conditioning law). A cold-start measurement is a **different law** and may not be silently swapped for the warm-steady one — a change-of-conditions bridge node is required.
- **model tag** → the exact deployment: instance type, kernel version, dependency versions, config C. "Which environment" must have one answer, exactly as "which kernel" must.
- **parameter domain** → arrival-rate range and payload-size range over which the bound is asserted.
- **scale/rung** → units and reference (ms; p50 vs p95 — a p50 number silently compared to a p95 target is a RUNG failure).
- **bridge/witness IDs** → the node licensing end-to-end latency as a composition of stage latencies.

**Running the gates:**
- **DOMAIN** — a benchmark run only up to 1,200 req/s cannot support a claim asserted to 2,000 req/s → **FAIL**.
- **ENDPOINT** — evaluate at the boundary (2,000 req/s and max payload); if p50 there is 55 ms, the ceiling is false where it claims to hold → **FAIL**. (And note the interior-extremum gap: latency can be fine at both endpoints and blow up at a mid-range rate where a queue saturates — that is the §11 uniformity candidate, not caught by ENDPOINT.)
- **MEASURE** — a warm-cache number standing in for a cold-start claim (or vice versa) with no bridge node → **FAIL**.
- **COMPOSITION** — end-to-end latency reported as the *sum* of stage medians assumes the stage latencies compose additively and independently; medians are **not** additive and tail-correlated stages violate independence. This needs a witness node (e.g. a joint-distribution bound, or a measured end-to-end figure — "exact algebra" replaced by direct measurement). A witness-free sum → **FAIL**. *(This is the direct analogue of the ~3.35× corridor error: a product/sum asserted without a dependence witness.)*
- **MODEL** — a latency figure with no pinned environment (instance type, versions, config) → **FAIL** on unnamed model; a figure measured on config C′ used for a claim about C needs a transfer-certificate node bounding the difference.
- **COMMON-MODE** — two load-testing harnesses that share a warmup or clock-sampling routine agree on 40 ms; if both mis-handle warmup identically, the agreement certifies nothing. PASS is **PROVISIONAL** until a harness with a disjoint measurement path, or a production trace, cross-checks it.
- **HEURISTIC-BRIDGE** — extrapolating from a single-endpoint microbenchmark to the full-system claim by analogy ("the hot path dominates") is a load-bearing heuristic; without a derivation it caps the claim at **Plausible**.

The gates, lexicons, and disciplines port without modification; only the *contents* of the measure/model/mark lexicons are swapped for the domain's native ones, exactly as §3.2's parenthetical promised.

---

## 9. The gates at a glance (checklist form)

For rapid application, a claim is gated by answering each. **Shell questions are mechanical; the parenthetical core is the obligation §7 must discharge.**

1. **DOMAIN** — do all dependencies cover the full parameter range asserted?
2. **ENDPOINT** — does the bound hold at every *registered* boundary? (Core: are the registered endpoints the true extrema? See §11.)
3. **POLARITY** — do input directions and band sides match a one-sided claim? (Core: are the direction tags faithful?)
4. **MEASURE** — is every input under the required conditioning law, or bridged by a graded change-of-measure node?
5. **MARK** — are all required marks/types supplied, or bridged by a graded mark-transfer node?
6. **COMPOSITION** — does every product/union/assembly cite a sufficiently-graded witness node?
7. **RUNG** — does every measured value carry its scale?
8. **PRECEDENCE** — is any killed/superseded node reachable? Is any dependency *stale* under hash-drift (§7)?
9. **CORE CLOSURE** — does a "finished" claim secretly depend on an open node?
10. **MODEL** — is the exact model named, or a scoped transfer-certificate node cited and used in-scope?
11. **COVERAGE** — is every *registered* possible-failure region covered by a detector? (Core: is the region list complete? Treat PASS as "no gap found," not "no gap.")
12. **COMMON-MODE** — if a PASS rests on instrument agreement, are the instruments' error channels shown disjoint?
13. **HEURISTIC-BRIDGE** — is any load-bearing analogy bridged by a node, or the claim graded ≤ Plausible?

Plus the schema field **F-1** (regime-of-validity) on every constant/mechanism; the **constitutional dependency rule** (§3.1) on every edge, now including **bridge/witness edges** (§3.8); and the **hash-drift discipline** (§7) on every dependency edge.

---

## 10. Glossary (for readers outside the source domain)

- **Palm distribution / pinning** — a conditioned law: the distribution of a random field *given* that specified events occur (e.g. given field values at points, or given a critical point of a given type at a location). Different conditionings give genuinely different laws.
- **Pair-Palm** — the Palm law conditioned on the existence of a *pair* of configured features at specified places.
- **Kac–Rice** — a formula giving the expected number/intensity of level crossings or critical points of a random field; the standard tool for such counts.
- **Mark / marked point process** — points carrying auxiliary data (a "mark"), e.g. the height/value of the field or the type/index of a critical point. Controlling the unmarked points is weaker than controlling the marked ones.
- **Model / covariance / kernel** — the exact stochastic model; for a Gaussian field, its covariance function (kernel) determines everything. "Which model" must be named.
- **Arbitrary-precision (exact) vs. floating-point** — exact symbolic/rational or very-high-precision computation, as opposed to ordinary floating-point; the framework keeps their gates separate.
- **Freeze** — committing a specification/prediction to an immutable record before executing the computation that tests it.
- **Kill registry / E-ledger** — the permanent, append-only record of failed claims, preserved as the test set for validating gates.
- **Retrodiction** — testing a proposed rule against a *frozen* record of past failures (can it return "no"?), as opposed to a constructed or forward-looking score.
- **Supersede-never-overwrite** — corrections create new versions; superseded artifacts are retained and marked, never deleted.
- **Restriction-drop** — the framework's core failure shape: using a result after silently dropping a restriction under which it was established.
- **Shell / core** *(v1.1)* — the mechanical, tag-checking layer of a gate (shell) versus the non-mechanical mathematical obligation the tags assert (core).
- **Bridge node** *(v1.1)* — a change-of-measure, composition witness, mark-transfer, or model-transfer certificate represented as a first-class *graded* node in the dependency graph, subject to the constitutional rule — not a free-text attestation.
- **Hash-drift / stale-PASS** *(v1.1)* — the condition where a dependency's content hash changes without a supersession event, invalidating every parent PASS that stood on the old content.

---

## 11. Candidate gates pending retrodiction *(new section in v1.1)*

The framework's own admission protocol (§6) forbids minting a gate without frozen retrodiction evidence. The author of an update is *not* exempt from this — smuggling in a favored rule under authorial privilege is precisely the failure the protocol exists to prevent. The following are therefore **staged as candidates, explicitly not counted in the established set**, for the maintainer to test against the kill registry. Each is admitted only if it clears all three §6 conditions (including the COMMON-MODE-on-evidence strengthening).

**Candidate A — HASH-DRIFT (stale-PASS on silent dependency edit).**
- *Proposed guard:* integrity of a dependency edge across edits — a PASS is valid only against the exact dependency content it was run on.
- *Why staged, not established:* the mechanism is already live as an operating discipline (§7). To promote it to a **gate**, find ≥2 independent kill-registry entries where a dependency was silently edited and a parent's PASS was wrongly retained. If the registry contains this shape twice with disjoint causes, it clears §6 and becomes a full gate; content-hash is then load-bearing rather than merely active. *(Action for maintainer: grep the registry for edit-without-supersession events.)*
- *Discharge check (condition 3):* not covered by PRECEDENCE, which guards *marked-dead* nodes, not *silently-mutated live* ones.

**Candidate B — UNIFORMITY (interior non-uniformity of a bound over a parameter).**
- *Proposed guard:* uniformity of a bound's constant over its parameter range, and quantifier *order* (∀ε∃δ vs. ∃δ∀ε).
- *Why it is a genuine gap:* DOMAIN checks *in-range*; ENDPOINT checks the *boundary*. Neither checks whether a bound's constant stays finite in the *interior* — a constant can be finite at every registered endpoint and blow up approaching an interior bad point, or a per-parameter δ can be silently promoted to a uniform δ. For an asymptotic-rate program over Gaussian fields, non-uniform convergence masquerading as uniform is a classic load-bearing error; in the §8.2 software example it is the mid-range queue-saturation case that passes both endpoints.
- *Why staged, not established:* it must clear §6. Search the kill registry for this shape — a bound asserted uniform that was secretly pointwise, or a quantifier-order swap. If it appears ≥2 times with disjoint causes, admit it. *(Action for maintainer: grep for uniformity/quantifier-order kills.)*
- *Discharge check (condition 3):* not covered by DOMAIN (range-containment ≠ constant-uniformity) nor ENDPOINT (boundary ≠ interior).

If the registry does not yield the evidence, **these are rejected** — and rejecting them is as much a success of the protocol as admitting a gate would be.

---

## Appendix A — Provenance and change log

**v1.0 → v1.1 (2026-07-17).** v1.1 supersedes v1.0; v1.0 is retained and marked (§7). Changes:
- **Type-system reframing (§0.5).** "Mechanically checkable" replaced by an explicit shell (mechanical, tag-consistency) / core (non-mechanical proof obligation) split, with the stated concession that gates certify tag-*consistency*, not tag-to-content *fidelity*, and that fidelity comes only from §7. Each gate in §4 now names its shell and core; §9 marks shell vs. core per line.
- **Bridges are nodes (§3.8).** Composition witnesses, changes-of-measure, mark-transfers, and model-transfer certificates promoted from free-text fields to first-class graded nodes, bringing them under the constitutional rule (§3.1). Closes v1.0's largest hole: unproven content laundering past the grade ladder via a named-but-ungraded witness. Gates 4, 5, 6, 10 and the schema (§2) restated accordingly.
- **Gates classify; disciplines detect (§0.6, §7).** Explicit division of labor: the §7 disciplines *detect* content/numerical errors (the factor-2 and 3.35× catches were recomputation catches); the §4 gates *classify* all errors and *detect* structural/transfer errors inside correctly-tagged steps. Corrects v1.0's implicit overstatement of the gates' role in the hardest catches.
- **Derived dependability resolved (§3.1).** The v1.0 exclusion of *Derived* from the permitted-dependency set is resolved into a stated three-tier dependability ordering (fully / self-tier / not dependable). Derived is *self-tier* dependable: Derived-on-Derived is legal, nothing above Derived may launder through it. The original set is revealed as the *fully-dependable* tier, not the top of the warrant ladder — the exclusion was correct but underspecified.
- **PROVISIONAL made general (§3.1).** Given explicit grade-ceiling semantics; the dangling `§4.12–4.13` reference removed; the two emitting gates (COMMON-MODE, HEURISTIC-BRIDGE) and their ceilings named.
- **Retrodiction runs COMMON-MODE on its own evidence (§6).** Condition 1 now requires the ≥2 entries to be independent under a COMMON-MODE check; the ≥2 threshold's exemption from §5.1 is stated and defended (governance vs. measurement threshold), with a matching scope note in §5.1.
- **Content-hash activated (§2, §7) and staged as a gate (§11).** Hash-drift/stale-PASS added as an operating discipline now, and as candidate gate A pending kill-registry retrodiction.
- **Second worked example (§8.2).** A software/empirical latency claim demonstrates domain transfer with no GRF vocabulary.
- **Candidate-gate section (§11).** Hash-drift and uniformity staged, not counted, for the maintainer to test against the registry — honoring §6 against authorial privilege.

**Established set (unchanged in count).**
- **Gates 1–11** (DOMAIN, ENDPOINT, POLARITY, MEASURE, MARK, COMPOSITION, RUNG, PRECEDENCE, CORE CLOSURE, MODEL, COVERAGE): developed through successive audit cycles; consolidated and machine-checked in a verifier prototype; COVERAGE the most recent, from a late audit finding an un-charted failure region.
- **Gate 12 (COMMON-MODE)** and **Gate 13 (HEURISTIC-BRIDGE)**: added 2026-07-17, justified by retrodiction against the frozen kill record (§4 and prior appendix). *(v1.1 note: gate 13's retrodiction evidence should itself be re-checked under the strengthened §6 condition 1 — confirm its supporting kills are distinct analogies, not one analogy reused.)*
- **Field F-1 (regime-of-validity)**: added 2026-07-17 as a schema field, the minimal fix for the most frequent transfer-error shape.

**Count note.** The **established** set stands at 13 gates + 1 schema field — unchanged by v1.1, because no new gate cleared §6 in this revision; the two staged candidates (§11) are explicitly *not* counted until they do. v1.1 strengthens *descriptions, schema, and protocol*, not the gate count. The count remains an output of the failure record, not a goal.

## Appendix B — Adoption notes for another model

- Treat this file as a complete specification. To apply it: represent each claim as a §2 proof object (including bridge/witness IDs as graded nodes, §3.8); assign a §3.1 grade and respect the dependability tiers; run the §9 checklist / §4 gates over the claim and its dependency cone; promote only on all-pass; cap grade or block on any FAIL/PROVISIONAL (§3.1); log failures per §7.
- **Understand what you are and are not getting (§0.5).** The gates give you tag-*consistency* mechanically. They do **not** give you tag-to-content *fidelity* — that comes only from the §7 disciplines (adversarial reimplementation, separation of duties). Do not report a gate PASS as a correctness certificate; report it as "consistent under the declared tags, conditional on the tagging being faithful."
- **Attribute catches honestly (§0.6).** A number caught by recomputation was caught by a discipline, not a gate; the gate *classified* it. Report efficacy this way.
- Do **not** weaken the quantitative gates into qualitative metaphors when applying the framework to a soft domain. A gate you cannot check mechanically at the shell is not a gate.
- When reporting the framework's efficacy, report it as a **qualitative finding tied to specific caught cases**, with the survivorship limitation stated (§6). Do not report a numerical "improvement" over cases selected to be favorable — that is the exact failure mode this framework exists to catch.

*End of Master Gate File — v1.1 — 2026-07-17.*
