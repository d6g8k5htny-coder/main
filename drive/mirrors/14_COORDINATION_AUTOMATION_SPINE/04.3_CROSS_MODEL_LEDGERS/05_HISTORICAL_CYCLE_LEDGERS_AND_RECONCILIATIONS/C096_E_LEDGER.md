# C096 E-LEDGER

**Cycle:** C096  
**Policy:** append-only; supersede-never-overwrite.  
**Scope:** uploaded Gate Framework files, executable kernels, legacy registry,
PDF, UB-G residual source, and BR-MARK continuation.

---

## E-C096-1 — Two different artifacts were both named Master v1.1

**Finding.** The uploaded `gate-framework-master-v1_1.md` and the C095
`GATE_FRAMEWORK_MASTER_v1_1.md` were independently current, but they specified
different gate counts and different machine contracts.

**Cause.** Parallel development lines reused the same version identifier.

**Impact.** “Use v1.1” was not a well-defined instruction.

**Disposition.** `CORRECTION`. Both files are preserved. Their compatible
content is reconciled in `GATE_FRAMEWORK_MASTER_v1_2_RECONCILED.md`.

---

## E-C096-2 — Metadata consistency had been overstated as mathematical verification

**Finding.** A gate can verify that tags agree while all tags are wrong.

**Source repair imported.** The uploaded v1.1 explicitly separates a
mechanical shell from a mathematical core and states that gates certify
tag-consistency, not tag-to-content fidelity.

**Disposition.** `CORRECTION-RETAINED`. Master v1.2 makes the shell/core
distinction constitutional.

---

## E-C096-3 — Conditions could disappear through “fully dependable” dependencies

**Finding.** Primitive and Proven-Modulo objects were called fully dependable,
but no graph rule propagated their unresolved condition sets.

**Failure shape.**

```text
H -> lemma graded Proven-Modulo -> parent graded Proven
```

could erase `H` if the parent read only the lemma's grade.

**Disposition.** `CORRECTION`. Gate Kernel 2.0 propagates complete condition
sets. A nonconditional parent inherits every unresolved condition or fails.

---

## E-C096-4 — COMMON-MODE's old provisional ceiling was a no-op

**Finding.** “Ceiling = current grade” preserves an already inflated
`Proven` label.

**Disposition.** `CORRECTION`. The default agreement-only ceiling is
`Derived`. A stricter project policy may lower it.

---

## E-C096-5 — Witness nodes were graded but not operation typed

**Finding.** The 1.x schema could use a union-bound witness for a probability
product.

**Cause.** `is_combination=True` did not identify the operation.

**Disposition.** `CORRECTION`. Kernel 2.0 records both operation and witness
type. A witness must be compatible with the exact operation.

---

## E-C096-6 — UNION_BOUND was absent from the closed witness lexicon

**Finding.** The framework invoked a union bound while requiring
“disjoint events” and omitting `UNION_BOUND` from the witness list.

**Disposition.** `CORRECTION`. A union bound requires neither disjointness nor
independence. It is an explicit witness type.

---

## E-C096-7 — Regime-of-validity recorded only establishment, not deployment

**Finding.** A transfer gate needs both regimes to decide whether a
restriction was dropped.

**Disposition.** `CORRECTION`. F-1 is now an establishment/deployment pair.

---

## E-C096-8 — Content hashes were labels rather than hashes

**Finding.** Values such as `C#1` and `T#1` were accepted as content hashes.
Changing the statement while retaining the label did not stale a parent.

**Disposition.** `CORRECTION`. Kernel 2.0 recomputes canonical SHA-256 over
every load-bearing field and verifies every pinned edge.

---

## E-C096-9 — DOMAIN skipped missing restrictions

**Finding.** Gate Kernel v1.2 passed a quantified parent when a dependency had
no domain or omitted one parent parameter.

**Evidence.** `V12-DOMAIN-EMPTY-DEPENDENCY` and
`V12-DOMAIN-MISSING-PARAMETER`.

**Disposition.** `CORRECTION`. Missing domain metadata fails a quantified
parent.

---

## E-C096-10 — MARK could be self-attested by the parent

**Finding.** A parent could copy its required marks into its supplied-marks
field even when every support node was unmarked.

**Disposition.** `CORRECTION`. Marks for a non-leaf claim are collected from
the evidence cone or a typed mark-transfer node.

---

## E-C096-11 — High-grade but untyped nodes bridged arbitrary regimes

**Finding.** In v1.2, any sufficiently graded change-of-measure or transfer
edge could bridge any source and target.

**Disposition.** `CORRECTION`. Every bridge declares kind, source regime,
target regime, scope domain, transferred marks, evidence, and falsifier.

---

## E-C096-12 — COMMON-MODE was a Boolean attestation

**Finding.** `error_independence_arg=true` passed without instrument IDs,
shared components, tested error channel, or source hash.

**Disposition.** `CORRECTION`. A first-class common-mode certificate is now
required.

---

## E-C096-13 — Provisional ceilings did not propagate

**Finding.** The registry recorded `C_const` as provisional/Derived but
admitted a Proven parent because it read the dependency's declared grade.

**Disposition.** `CORRECTION`. Effective warrant ceilings propagate through
every nonconditional edge.

---

## E-C096-14 — PRECEDENCE ignored a dead root

**Finding.** A killed root with no descendants passed because only
descendants were checked.

**Disposition.** `CORRECTION`. The root is included in its own reachable cone.

---

## E-C096-15 — CORE CLOSURE checked Open but not Conjecture or debt

**Finding.** A closed core could retain a conditional Conjecture.

**Disposition.** `CORRECTION`. Closed cores require empty condition sets, no
Open/Plausible/Conjecture nodes, no provisional gates, and no open active
gates.

---

## E-C096-16 — Registry admission bypassed graph validation

**Finding.** Gate Kernel v1.2 admitted a live claim graded Killed as:

```text
PASS at Killed
```

**Disposition.** `CORRECTION`. Kernel 2.0 separates archive, candidate, and
promotion admission after strict graph validation.

---

## E-C096-17 — Registry loading silently returned a partial graph

**Finding.** `Registry.load` ignored admission refusals. A malformed claim
could disappear while the loader returned the remaining registry.

**Disposition.** `CORRECTION`. Loading is all-or-nothing and raises with the
complete structural error set.

---

## E-C096-18 — Schema version was ignored

**Finding.** `graph_from_json` accepted arbitrary schema labels.

**Disposition.** `CORRECTION`. Kernel 2.0 rejects every schema except
`gate-kernel/2.0`.

---

## E-C096-19 — Topological order silently omitted cycles

**Finding.** A cyclic graph produced an incomplete order rather than an error.

**Disposition.** `CORRECTION`. Cycle detection is explicit and topological
ordering raises.

---

## E-C096-20 — Gate Kernel v1.2 used a hard-coded output directory

**Finding.**

```text
/mnt/user-data/outputs/q0_registry.json
```

caused the shipped demo to fail outside its original environment.

**Disposition.** `CORRECTION`. Kernel 2.0 uses CLI paths or paths supplied by
the caller.

---

## E-C096-21 — Primary verdict conflated storage and theorem promotion

**Finding.** The demo printed `PASS at Proven-Modulo` while separately listing
an Open dependency and a provisional coefficient.

**Disposition.** `CORRECTION`. Root reports now distinguish:

```text
archive validity
shell validity
conditional promotability
unconditional promotability
closability
condition set
effective warrant
```

---

## E-C096-22 — The uploaded registry was not the Q0 theorem registry

**Finding.** It uses \(C=0.4127\), \(r\le1\), \(b\le3\), and generic
inner/far/boundary placeholders, which do not match the frozen Q0 program.

**Disposition.** `AUTHORITY-CORRECTION`. It is retained as a legacy schema
demonstration. It is never cited as theorem authority.

---

## E-C096-23 — `R0` had a dangerous cross-domain name collision

**Finding.** In the uploaded registry, `R0` means an unbounded remainder.
In the Q0 program, `R0` means the Gaussian-Sard/Morse-Smale condition.

**Disposition.** `CORRECTION`. The migrated demo uses
`PAIRING_REMAINDER_CONDITION`. Bare `R0` is prohibited in that registry.

---

## E-C096-24 — Source grades in the legacy registry were unsupported

**Finding.** `inner_bound`, `far_bound`, and `bdry_bound` were labeled Proven
without proof artifacts. `C_const` claimed a closed form but supplied only
0.4127.

**Disposition.** `DOWNGRADE`. They are explicit Open/Plausible or
Open/Measured conditions in the migrated archive.

---

## E-C096-25 — Endpoint Booleans were mistaken for endpoint certificates

**Finding.** `true` was stored beside endpoint expressions, with no executed
calculation or immutable evidence object.

**Disposition.** `CORRECTION`. `ENDPOINT-FIDELITY` remains open in the
migration.

---

## E-C096-26 — The PDF was an obsolete v1.0 artifact

**Finding.** The 15-page PDF retains the old “mechanically checkable” wording
and old schema, while the accompanying Markdown is v1.1.

**Disposition.** `SUPERSEDED`. The PDF is provenance-only. A clean Master v1.2
PDF has been generated.

---

## E-C096-27 — The old PDF had weak navigation and object warnings

**Finding.** It had no bookmarks or page numbers and emitted wrong-pointing
object warnings for objects 7 0, 16 0, and 46 0.

**Disposition.** `CORRECTION`. The new PDF has a table of contents, bookmarks,
page numbers, and a normalized object structure.

---

## E-C096-28 — The absolute Gamma ceiling cannot prove a cubic rate at zero

**Finding.** The source bound is

\[
D(r)\le C_{\rm base}r^3+e^{-92}+\operatorname{Collar}(r).
\]

A positive constant cannot be absorbed into \(Cr^3\) on
\(0<r\le r_0\).

**Impact.** The printed UB-G source does not, by itself, prove \(q(r)\to1\).

**Disposition.** `CORRECTION`. The q₀ implication is
`PROVEN-MODULO GAMMA-DENSITY` and an explicit collar coefficient.

---

## E-C096-29 — The collar had an \(O(r^3)\) shape but no usable coefficient

**Finding.** The near-diagonal backstop gives, per radius-\(2r\) collar,

\[
\frac{16\pi}{3}C_{\rm nd}r^3.
\]

The retrieved record does not give a \(C_{\rm nd}\) small enough to fit the
4.35 budget.

**Disposition.** `OPEN-COEFFICIENT`. Structural cubic order remains Derived.

---

## E-C096-30 — Uniform six-pin mark density fails in the pair-scaled chart

**Finding.** The exact limiting Gaussian mark law was computed for
\((A,Z)\). On tested \(O(r)\)-scaled midpoints,

\[
\det\Sigma_{A,Z}\asymp r^8\text{ to }r^{10},
\]

and the Gaussian density supremum grows approximately \(r^{-4}\) to
\(r^{-5}\).

**Impact.** A global factorization into “spatial repulsion constant × uniformly
bounded mark-density constant” is not valid in the near-pair chart.

**Disposition.** `HYPOTHESIS-KILLED-AS-UNIFORM`. BR-MARK is reshaped around a
joint typed spatial-marked Kac–Rice bound with regional charts and Palm
determinant depletion.

---

## E-C096-31 — The framework had two candidate notions of uniformity

**Finding.** Uploaded v1.1 staged a general quantifier-order UNIFORMITY gate;
C095 established a numerical DOMAIN-INFIMUM gate.

**Disposition.** `RECONCILIATION`. Gate 15 establishes numerical
UNIFORMITY/DOMAIN-EXTREMUM. General logical quantifier-order uniformity remains
a staged extension.

---

## E-C096-32 — The 1.x kernel line cannot enforce its own conceptual master

**Evidence.** Seventeen adversarial tests reproduced seventeen shell or
registry escapes.

**Disposition.** `SUPERSEDED`. Gate Kernel 2.0 is a breaking schema revision,
not a patch-level edit.

---

## E-C096-33 — Gate Kernel 2.0 passed the independent replacement battery

**Evidence.** Thirty-six preregistered positive and negative cases passed.

**Disposition.** `CERTIFIED-SHELL-PROTOTYPE`. This grade applies to executable
contract behavior, not to the mathematical truth of arbitrary supplied
certificates.

---

**Entries:** 33  
**Homeless entries:** 0
