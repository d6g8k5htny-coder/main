# THE GATE FRAMEWORK — MASTER GATE FILE

## Standalone executable specification for proof-carrying reasoning and AI-assisted mathematics

**Version:** Master v1.1  
**Date:** 2026-07-17  
**Status:** current successor specification  
**Supersedes:** Master v1.0; v1.0 is preserved and not overwritten  
**Set:** one mandatory schema preflight, 16 numbered gates, five schema field groups

> **Standalone notice.** This file is self-contained. A human or model can
> implement the proof-object schema, gate semantics, promotion rules, and
> operating discipline without any other file. Domain examples use
> Gaussian-random-field mathematics, numerical proof contracts, and
> proof-carrying language-model verification, but the framework itself is
> domain independent.

---

# 0. Purpose and operating rule

## 0.1 Failure profile

Rigorous-looking reasoning often fails without looking obviously broken:

- a theorem is asserted over a wider parameter range than its lemmas cover;
- an upper bound is rounded downward;
- a first-moment coefficient is displayed as a final probability coefficient
  after multiplicative losses have been omitted;
- a spatial point-process estimate is used as a height-marked estimate;
- a coefficient computed under Gaussian pinning is used under a pair-Palm law;
- correlated probabilities are multiplied as if independent;
- a value measured at one rung is treated as scale free;
- a uniform claim is inferred from sampled rungs without an interval extremum
  certificate;
- a symmetric uncertainty band is consumed inside a one-sided theorem without
  construction metadata;
- two checking instruments agree because they share the same defect;
- an analogy becomes load bearing without a derivation;
- a finished theorem still reaches an open or killed node.

Each individual step can sound plausible. The framework makes these failures
machine visible before promotion.

## 0.2 Unit of operation

The unit is an **atomic claim** represented by a structured proof object.
Claims form a directed dependency graph. Gates are predicates over:

1. one claim;
2. its evidence and metadata;
3. the edges into its dependency cone;
4. any numerical assembly or domain certificate attached to it;
5. the status and hashes of its sources.

## 0.3 Core discipline

> **Name the restriction; gate the drop.**

A result established under a restriction may not be used after silently
dropping that restriction. Either carry it into the parent claim or provide a
named bridge.

The numbered gates guard restriction axes. The numerical-contract gates also
guard arithmetic and certificate completeness, because several recorded
failures preserved every verbal restriction while still assembling the wrong
number.

## 0.4 Verdicts

Every gate returns one of:

- **PASS** — the gate obligation is discharged;
- **FAIL** — promotion is blocked;
- **PROVISIONAL** — the claim may remain at the gate's declared cap but may not
  promote beyond it;
- **NOT-APPLICABLE** — the claim does not invoke the guarded operation.

A claim can promote only when every applicable gate is PASS. A PROVISIONAL
result is not equivalent to PASS.

## 0.5 Execution order

Run gates in this order:

```text
SCHEMA preflight
graph closure and precedence
restriction-transfer gates
numerical-contract gates
verification-method gates
coverage and deployment gates
final grade propagation
canonical serialization and hashing
```

---

# 1. Proof-object schema

Every externally visible claim carries all fields below. A missing required
field fails the unnumbered SCHEMA preflight.

## 1.1 Identity and graph

| field | meaning |
|---|---|
| `claim_id` | globally unique stable identifier |
| `atomic_statement` | one assertion, not a bundle |
| `dependency_ids` | load-bearing claim IDs |
| `evidence_ids` | proofs, computations, measurements, source records |
| `source_precedence_id` | version/supersession lineage |
| `status` | live, open, killed, superseded, retired |
| `supersedes` | claim IDs replaced by this object |
| `canonical_hash` | hash of canonical serialization plus dependency hashes |

## 1.2 Logical and epistemic fields

| field | meaning |
|---|---|
| `quantifier` | exact identity, ∀, ∃, asymptotic, typical, measured-at-rung |
| `direction` | lower, upper, equality, structural |
| `assumption_role` | none, definition, primitive hypothesis, policy threshold |
| `warrant_grade` | Proven, Proven-Modulo, Certified, Derived, Literature-Supported, Measured, Plausible, Conjecture |
| `reasoning_mode` | deductive, empirical, analogical, heuristic, normative |
| `load_bearing` | whether this claim can change a promoted conclusion |
| `heuristic_bridge_id` | derivation that licenses an analogy/heuristic transition |

`Open`, `Killed`, and `Superseded` are statuses, not warrant grades.
`Primitive` is an assumption role, not the strongest point on a ladder.

## 1.3 Restriction fields

| field | meaning |
|---|---|
| `model_tag` | exact mathematical/probabilistic model |
| `measure_tag` | unconditioned, Gaussian-pinned, typed-pinned, pair-Palm, or domain analogue |
| `parameter_domain` | complete free-parameter set |
| `required_marks` | marks/types needed by the claim |
| `supplied_marks` | marks/types supplied by evidence |
| `scale_tag` | units/rung of every measured value |
| `establishment_regime` | F-1 tuple under which evidence was established |
| `deployment_regime` | F-1 tuple where parent uses the result |
| `transfer_bridge_ids` | explicit bridges for regime mismatches |

## 1.4 Composition and numerical fields

| field | meaning |
|---|---|
| `operation` | atomic, product, union, affine assembly, monotone transform, interval enclosure, other |
| `composition_witness` | named license for combining |
| `composition_evidence_ids` | evidence for that witness |
| `assembly_spec` | F-2 canonical numerical expression and rounding contract |
| `domain_extremum_certificate` | full-domain infimum/supremum certificate |
| `uncertainty_side` | upper, lower, symmetric, none |
| `uncertainty_provenance` | F-3 construction metadata |
| `registered_endpoints` | endpoint values and evidence |
| `measured_value` | numerical measurement, if any |

## 1.5 Verification and coverage fields

| field | meaning |
|---|---|
| `verification_mode` | exact-single, agreement, adversarial reimplementation, external contradiction |
| `instrument_ids` | checking instruments |
| `common_mode_certificate` | F-4 independence/shared-component analysis |
| `coverage_universe_id` | declared possible-failure universe |
| `chart_ids` | detectors/charts covering the universe |
| `coverage_certificate_id` | proof or measured uncovered-mass bound |
| `active_gate_ids` | domain-specific first-class gates consumed by this claim |

---

# 2. Five schema field groups

## F-1 — establishment and deployment regime

A regime is the tuple:

```text
(
  exact model,
  base measure/conditioning law,
  spatial or state-space region,
  parameter box,
  scale/rung,
  conditioning depth,
  mark/type set,
  numerical precision or deployment mode
)
```

Every reusable constant or mechanism records both:

- `establishment_regime`;
- `deployment_regime`.

A mismatch requires a transfer bridge whose scope contains the deployment
regime. A single vague “validity” label is insufficient.

## F-2 — assembly specification

Every displayed numerical bound assembled from inputs records:

```text
expression:
    canonical arithmetic expression

inputs:
    named values or intervals with source IDs

monotonicity:
    sign of the output with respect to each input

claim_direction:
    upper | lower | equality

rounding_rule:
    upward | downward | exact | outward interval

precision:
    decimal/rational/arbitrary-precision context

residual_terms:
    every positive upper residual or negative lower loss

displayed_value:
    the published number
```

The ASSEMBLY gate evaluates this object, not prose.

## F-3 — uncertainty and band provenance

Every load-bearing uncertainty object records:

```text
kind:
    deterministic | interval | statistical | bootstrap |
    Monte Carlo | randomized numerical integration | forecast

side:
    upper | lower | symmetric

construction:
    formula or algorithm ID

confidence_level:
    if statistical

sample_size:
    if sampled

multiplicity_scope:
    stations, rungs, models, comparisons

seed_or_algorithm_state:
    if stochastic

deterministic_tolerance:
    if numerical

coverage_domain:
    parameter and deployment domain

source_hash:
    immutable artifact
```

A forecast is never silently promoted to a confidence or theorem band.

## F-4 — verification independence

An agreement-based verdict records:

```text
instrument_ids
shared_code
shared_data
shared_preprocessing
shared_model_assumptions
shared_numerical_library
error_channel_under_test
disjoint_error_argument
external_contradiction_test
```

The COMMON-MODE gate consumes this field.

## F-5 — reasoning mode and bridge

Every analogical or heuristic step records:

```text
reasoning_mode
source_domain
target_domain
load_bearing
bridge_claim_id
bridge_scope
falsifier
grade_cap_without_bridge
```

The HEURISTIC-BRIDGE gate consumes this field.

---

# 3. Status, warrant, and dependency semantics

## 3.1 Claim status

Closed status lexicon:

- **LIVE**
- **OPEN**
- **KILLED**
- **SUPERSEDED**
- **RETIRED**

Killed, superseded, and retired objects remain in the archive but may not be
reachable from a live promoted root.

## 3.2 Assumption role

- **DEFINITION** — fixes meaning;
- **PRIMITIVE-HYPOTHESIS** — explicit theorem assumption;
- **POLICY-THRESHOLD** — user or governance choice, not a mathematical fact;
- **NONE**.

A primitive hypothesis does not become “proved” by being placed at the top of
a ladder.

## 3.3 Warrant classes

- **Proven** — self-contained referee-grade proof under definitions only;
- **Proven-Modulo** — proved implication under named hypotheses or unresolved
  conditions;
- **Certified** — machine/interval/formal certificate within declared scope;
- **Derived** — executed derivation not yet referee packaged;
- **Literature-Supported** — external source supports the precise claim or a
  weaker claim with an explicit adaptation bridge;
- **Measured** — empirical/numerical result with provenance;
- **Plausible** — mechanism supported but incomplete;
- **Conjecture** — explicit unproved proposition.

These classes are not forced into a false total order. Promotion follows the
dependency matrix below.

## 3.4 Constitutional dependency rule

1. `KILLED`, `SUPERSEDED`, or `RETIRED` dependencies are forbidden.
2. A `Proven` parent may depend only on definitions and Proven claims.
3. A `Proven-Modulo` parent may depend on named primitive hypotheses, Open
   conditions, or Conjectures **only when the implication from those
   hypotheses is itself proved**.
4. A `Certified` parent may depend on Certified/Proven inputs and explicit
   primitive assumptions within the certificate scope.
5. A `Derived` parent may depend on Derived or stronger inputs; if it consumes
   Measured inputs, it is labeled `Derived-with-Measured-Inputs` or capped by
   the measured weakest link.
6. A Literature-Supported input supports only the exact externally supported
   claim. Adaptation to a new model/measure/domain requires a bridge whose
   grade controls the parent.
7. A Measured input cannot become a Proven numerical constant without an
   interval/statistical theorem that supplies the promotion.
8. An unbridged Plausible/Conjecture input caps the parent at Plausible.
9. An Open dependency not converted into an explicit theorem hypothesis fails
   CORE CLOSURE.
10. Every parent records the weakest-link adjudication.

---

# 4. Composition witness lexicon

Closed witness list:

- `INDEPENDENCE`
- `CONDITIONAL-INDEPENDENCE`
- `MARKOV-PROPERTY`
- `NEGATIVE-DEPENDENCE`
- `COMPARISON-THEOREM`
- `JOINT-CERTIFICATE`
- `UNION-BOUND`
- `MONOTONE-ASSEMBLY`
- `INTERVAL-ARITHMETIC`
- `EXACT-ALGEBRA`

Operation-specific rules:

- a probability **product** requires independence, conditional independence,
  Markov structure plus the relevant transition theorem, negative dependence
  in the required direction, a comparison theorem, or a direct joint
  certificate;
- a **union bound** requires explicit event coverage but no independence and
  no disjointness;
- an **affine/monotone assembly** requires sign and polarity metadata;
- an **identity** uses exact algebra;
- interval enclosures use outward interval arithmetic.

---

# 5. Mandatory SCHEMA preflight

Before Gate 1:

1. validate all required fields for the claim type;
2. validate closed lexicon values;
3. validate unique IDs and source IDs;
4. canonicalize the proof object;
5. verify the content hash;
6. verify dependency IDs exist;
7. verify evidence IDs exist or are explicitly BLOCKED-EXTERNAL;
8. reject bundled statements containing multiple independent assertions.

A claim failing SCHEMA is not gated further except for diagnostic reporting.

---

# 6. The 16 gates

## Gate 1 — DOMAIN

**Guards:** parameter range.

**Rule:** every dependency and bridge covers the complete parent domain.

**PASS:** set containment for every parameter with declared tolerance.

**FAIL:** any parent point lies outside a dependency/bridge domain.

**Canonical failures:** subinterval lemma used on a larger interval; transfer
certificate invoked outside its dimension/order/distance box.

---

## Gate 2 — ENDPOINT

**Guards:** declared boundaries.

**Rule:** every registered endpoint satisfies the displayed finite-range
inequality.

**PASS:** all endpoint evaluations satisfy direction.

**FAIL:** any endpoint violates the bound.

**Note:** endpoint PASS is necessary, never sufficient for a uniform claim.

---

## Gate 3 — POLARITY

**Guards:** sidedness.

**Rule:** upper claims consume upper inputs and positive residuals; lower
claims consume lower inputs and every multiplicative/subtractive loss.

**PASS:** all signs and uncertainty sides match the claim direction.

**FAIL:** symmetric band used unchanged in a one-sided theorem; upper bound
rounded down; lower coefficient shown before subunit factors.

---

## Gate 4 — MEASURE

**Guards:** base law and conditioning.

**Rule:** evidence and parent use the same measure, or an explicit
change-of-measure bridge is present.

**FAIL examples:** Gaussian-pinned coefficient used under pair-Palm;
unconditioned count inserted into a selected conditioned law without a
Campbell/change-of-measure theorem.

---

## Gate 5 — MARK

**Guards:** types and auxiliary variables.

**Rule:** every required mark is supplied by evidence or a mark-transfer
theorem.

**FAIL examples:** spatial repulsion used as a two-height window law; aggregate
critical-point result used as a same-index saddle result.

---

## Gate 6 — COMPOSITION

**Guards:** combination license.

**Rule:** every product, union, or assembly has an operation-appropriate
witness from §4.

**FAIL examples:** \(q_{\rm step}^n\) for a correlated corridor; determinant
weight treated as independent of conditioned values.

**Clarification:** a union bound does not require disjointness.

---

## Gate 7 — RUNG

**Guards:** measured scale.

**Rule:** every measured value carries units, parameter values, and rung.

**FAIL example:** `TRUTH_CONST=0.946` without \(r=0.025\), \(b=1.2\), and
estimand.

---

## Gate 8 — PRECEDENCE

**Guards:** live/dead lineage.

**Rule:** no live root reaches a killed, superseded, or retired node.

**PASS:** graph reachability contains only live admissible nodes.

---

## Gate 9 — CORE CLOSURE

**Guards:** finished/open boundary.

**Rule:** a closed core has no Open dependency and no unresolved extension
upstream.

**Exception:** a proved implication may be `Proven-Modulo` under explicitly
named hypotheses; it is not called an unconditional closed core.

---

## Gate 10 — MODEL

**Guards:** exact model.

**Rule:** every model-dependent claim names the exact model or a scoped transfer
certificate.

**FAIL examples:** planar kernel silently treated as a torus covariance;
matrix transfer used beyond certified derivative order or dimension.

---

## Gate 11 — COVERAGE

**Guards:** charted failure region.

**Rule:** every possible-failure region claimed under control is covered by at
least one registered chart/detector or by an explicit uncovered-mass term.

**PASS:** coverage universe, chart map, and uncovered-mass certificate are
present.

**FAIL:** exponential rank bound read as control over an uncharted error
region.

---

## Gate 12 — COMMON-MODE

**Guards:** independence of verification error channels.

**Applies when:** a verdict rests on agreement between instruments.

**PASS:** either:

1. a common-mode certificate demonstrates disjoint error channels for the
   error under test; or
2. an external contradiction/cross-domain test independently validates the
   result.

**PROVISIONAL:** agreement exists but error independence is unknown.

**Cap:** no promotion to Proven or Certified from the agreement alone.

**Exemption:** a single exact proof or arbitrary-precision identity that makes
no agreement claim.

**Framework evidence status:** operationally adopted; the v1.0 appendix did
not provide frozen failure IDs/hashes, so the meta-claim that retrodiction is
complete remains PROVISIONAL until that table is supplied.

---

## Gate 13 — HEURISTIC-BRIDGE

**Guards:** reasoning mode.

**Rule:** a load-bearing analogical or heuristic step feeding a quantitative
or asymptotic claim requires an explicit bridge.

**PASS:** bridge derivation exists and covers the deployment regime.

**FAIL:** unbridged analogy supports a claim graded Derived or stronger.

**Allowed downgrade:** without a bridge, the dependent claim is capped at
Plausible and must carry a falsifier.

**Framework evidence status:** operationally adopted; v1.0 did not identify
its claimed retrodiction entries by frozen ID/hash, so framework-level
validation remains PROVISIONAL.

---

## Gate 14 — ASSEMBLY

**Guards:** arithmetic propagation, monotonicity, and conservative display.

**Rule:** every displayed numerical result is recomputed from its certified
inputs using F-2.

**PASS requires:**

1. canonical expression evaluates;
2. every input hash matches;
3. monotonicity signs are correct;
4. every upper residual is added;
5. every lower loss is applied;
6. rounding is conservative:
   - upper rounds upward;
   - lower rounds downward;
7. displayed value encloses the recomputed value at declared precision.

**FAIL examples:**

- \((0.66+2.82)/0.80=4.35\) displayed as \(4.3\);
- \(C^*=0.8411\) displayed as a final lower coefficient before \(AO<1\) and
  Bonferroni losses.

**Retrodiction status:** PASS; at least two independent frozen failures.

---

## Gate 15 — DOMAIN-INFIMUM

**Guards:** full-domain extremum needed by a uniform claim.

**Rule:** a \(\forall\)-claim based on a numerical coefficient carries a
certificate for the relevant infimum/supremum over the **entire** domain.

Accepted certificate types:

- exact analytic extremum;
- monotonicity plus endpoint;
- outward interval/Taylor model;
- exhaustive finite enumeration;
- validated grid plus proved modulus;
- asymptotic limit plus monotonicity/compact remainder.

A list of sampled rungs is not a certificate.

**FAIL examples:**

- continuous upper theorem inferred from three rungs without between-rung
  control;
- product threshold \(80\) accepted from rung minima although the
  \(r\downarrow0\) infimum is \(79.988920\ldots\).

**Retrodiction status:** PASS.

---

## Gate 16 — BAND-PROVENANCE

**Guards:** construction and interpretation of load-bearing uncertainty.

**Rule:** every uncertainty band or stochastic numerical tolerance carries
F-3 metadata.

**PASS:** kind, side, construction, coverage, and all applicable sampling or
algorithmic metadata are complete.

**FAIL examples:**

- an unexplained \(\pm0.02\) band used in an upper theorem;
- a randomized multivariate-normal diagnostic re-executed with changed
  output but no frozen integration state.

**Special rule:** a symmetric band inside an upper theorem is consumed as its
upper side unless proved already one sided.

**Retrodiction status:** PASS when stochastic numerical provenance is included
in scope; the two frozen entries are the unrecovered \(0.02\) band and the
unfrozen orthant diagnostic.

---

# 7. Promotion algorithm

For claim \(C\):

1. SCHEMA preflight.
2. Evaluate all 16 gates.
3. If any FAIL: block promotion.
4. If any PROVISIONAL:
   - apply the strongest gate-specific cap;
   - mark the claim PROVISIONAL;
   - preserve the unresolved certificate ID.
5. Apply dependency warrant propagation.
6. Apply reasoning-mode cap.
7. Apply measurement/uncertainty cap.
8. Record final grade and weakest-link explanation.
9. Canonically serialize:
   - claim metadata;
   - evidence hashes;
   - dependency hashes;
   - gate verdicts;
   - bridge/certificate hashes.
10. Compute root hash.

---

# 8. Gate-design standard

## 8.1 Threshold classes

A threshold is one of:

- **structural/theorem threshold** — Derived or stronger;
- **sampling threshold** — derived from a declared distribution and error
  target;
- **policy threshold** — explicit Primitive-Normative choice;
- **forecast** — non-promotable prediction.

A policy threshold is legitimate but cannot be cited as a discovered
mathematical constant.

## 8.2 Exact versus deployment

Keep separate:

- exact symbolic/arbitrary-precision claim;
- interval or floating-point deployment certificate;
- measured calibration.

A deployment test does not retroactively change an exact identity.

## 8.3 Failure diagnosis order

On numerical failure:

1. inspect variable definition;
2. inspect model/measure/mark;
3. inspect discretization and conditioning;
4. inspect assembly and rounding;
5. inspect the physical/mathematical mechanism last.

## 8.4 Limit bands

A limit band requires:

- exact remainder structure;
- validated asymptotic enclosure;
- or an explicit `FORECAST` label.

Finite-rung extrapolation alone is not a limit certificate.

## 8.5 Independent numerical evaluation

Every closed-form constant receives an independent numerical evaluation before
comparison. If promotion rests on agreement, Gate 12 also applies.

---

# 9. Retrodiction protocol

A new gate qualifies only against a frozen failure record.

1. It fires on at least two independent recorded failures.
2. It is silent on most of the record.
3. It is not already mechanically discharged by an existing gate or mandatory
   schema check.
4. The evidence table names IDs, artifact hashes, and the earlier catch
   mechanism.
5. The test is frozen before adjudication.

A candidate failing any item is rejected or marked PROVISIONAL.

## Survivorship limitation

Retrodiction measures only recorded/caught failures. It cannot estimate
failures that were never found. It demonstrates usefulness on known misses,
not completeness of the gate set.

---

# 10. Operating discipline

- freeze before execute;
- supersede, never overwrite;
- preserve every failure;
- write ledgers after adjudication;
- use adversarial reimplementation;
- separate generation, verification, and adjudication roles;
- run external source verification in parallel with internal consistency;
- externalize state before long computation;
- persist seeds and classify them as theorem-input or diagnostic;
- hash every release;
- verify in ZIP and clean extracted-directory modes;
- make open extensions downstream of, never upstream of, a finished core;
- maximize the weakest link.

---

# 11. Minimal worked numerical example

Claim:

\[
P(\text{defect})\le Cr^3,\qquad0<r\le r_0.
\]

Metadata:

```text
direction: upper
measure: pair-Palm
model: exact named kernel
required marks: position, height, type
domain: (0,r0]
scale: coefficient of r^3
operation: affine assembly
assembly:
    C = (near + exterior)/typing + gamma + collar
domain certificate:
    full interval supremum
uncertainty:
    upper one-sided, fully sourced
```

Gate outcomes:

- DOMAIN — every term covers \((0,r_0]\);
- ENDPOINT — endpoint values obey the ceiling;
- POLARITY — gamma/collar added, typing denominator used in worst direction;
- MEASURE — every term is pair-Palm or bridged;
- MARK — height/type marks present;
- COMPOSITION — union/expectation witness named;
- RUNG — measured station values tagged;
- PRECEDENCE — no killed route reachable;
- CORE CLOSURE — no open H4 node upstream;
- MODEL — exact kernel or scoped transfer;
- COVERAGE — every failure region represented or residualized;
- COMMON-MODE — agreement-based constants independently checked;
- HEURISTIC-BRIDGE — no analogy carries the coefficient;
- ASSEMBLY — recomputation and upward rounding pass;
- DOMAIN-INFIMUM — interval supremum certificate covers all \(r\);
- BAND-PROVENANCE — uncertainty construction complete.

Only then may \(C\) be displayed.

---

# 12. Rapid checklist

1. SCHEMA — is the proof object complete and canonically hashed?
2. DOMAIN — do dependencies cover the full domain?
3. ENDPOINT — do all boundaries satisfy the claim?
4. POLARITY — do signs, residuals, losses, and band sides match?
5. MEASURE — is the conditioning law correct?
6. MARK — are all required marks supplied?
7. COMPOSITION — is every combination licensed?
8. RUNG — are measured values scale tagged?
9. PRECEDENCE — is any dead node reachable?
10. CORE CLOSURE — does a finished root reach an open node?
11. MODEL — exact model or scoped bridge?
12. COVERAGE — is every failure region charted or residualized?
13. COMMON-MODE — are agreement errors independent?
14. HEURISTIC-BRIDGE — is a load-bearing analogy bridged?
15. ASSEMBLY — does the displayed arithmetic conservatively recompute?
16. DOMAIN-INFIMUM — is the full-domain extremum certified?
17. BAND-PROVENANCE — is uncertainty construction complete?

The numbered gate count is 16; SCHEMA is a mandatory preflight.

---

# 13. Adoption notes

Another model adopting this file should:

1. instantiate the schema as typed data structures;
2. reject incomplete proof objects before reasoning about them;
3. implement operation-specific gates rather than keyword heuristics;
4. encode at least one second domain before claiming generality;
5. preserve v1.0 and record v1.1 as a supersession;
6. report Gate 12 and Gate 13 as operationally useful but
   retrodiction-PROVISIONAL until their frozen evidence table is supplied;
7. never claim numerical efficacy percentages from a hand-selected failure
   set;
8. state the survivorship limitation every time gate validation is discussed.

---

# Appendix A — v1.1 change record

| change | reason |
|---|---|
| split status, assumption role, warrant, reasoning mode | v1.0 total ladder conflated unlike concepts |
| added SCHEMA preflight | v1.0 said missing fields fail but named no mechanism |
| added UNION_BOUND and monotone/interval witnesses | composition list was incomplete |
| expanded F-1 to establishment and deployment regimes | transfer comparison requires both sides |
| added F-2 assembly | UB-G downward rounding and lower-loss promotion |
| added F-3 uncertainty provenance | unexplained band and stochastic diagnostic |
| added F-4 common-mode certificate | Gate 12 needed mechanical metadata |
| added F-5 reasoning bridge | Gate 13 needed mechanical metadata |
| added Gate 14 ASSEMBLY | two independent recorded arithmetic propagation failures |
| added Gate 15 DOMAIN-INFIMUM | rung-only uniform claims |
| added Gate 16 BAND-PROVENANCE | uncertainty construction failures |
| clarified provisional caps | v1.0 left promotion semantics underspecified |
| corrected union-bound language | union bounds do not require independence/disjointness |
| separated policy thresholds | not every legitimate threshold is a derived fact |

**End of Master Gate File v1.1.**
