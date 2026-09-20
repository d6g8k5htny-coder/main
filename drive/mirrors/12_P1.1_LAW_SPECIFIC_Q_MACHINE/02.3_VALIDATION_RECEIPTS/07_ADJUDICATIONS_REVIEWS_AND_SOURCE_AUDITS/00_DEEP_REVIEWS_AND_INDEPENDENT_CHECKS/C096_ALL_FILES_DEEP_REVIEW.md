# C096 Deep Review of the Uploaded Gate Framework File Set

## 0. Scope and result

This review covers every newly supplied artifact:

1. `gate-framework-master-v1_1.md`;
2. `# THE GATE FRAMEWORK — MASTER….pdf`;
3. `gate kernel v1_1.py`;
4. `gate kernel v1_2.py`;
5. `q0 registry.json`.

It also reconciles them against the already frozen C095 verifier line.

The file set contains two distinct kinds of value:

- a strong conceptual framework, especially the uploaded Markdown v1.1;
- an executable shell prototype whose implementation does not yet enforce
  several of the framework's own strongest rules.

The conceptual framework is retained and strengthened. The 1.x kernel line is
superseded by a breaking 2.0 schema rather than patched in place.

The resulting authority stack is:

```text
conceptual specification:
    GATE_FRAMEWORK_MASTER_v1_2_RECONCILED.md

executable shell:
    gate_kernel_v2_0.py

legacy demonstration registry:
    q0 registry.json
    preserved, not theorem authority

typed migration:
    q0_registry_v2_0.json
    archive-valid, not candidate- or promotion-valid

v1.0 PDF:
    immutable provenance only
```

---

# 1. Uploaded Master v1.1 Markdown

## 1.1 What it gets exactly right

### Shell/core honesty

The file's most important move is the explicit statement that the proof object
is a type and the gates are typing rules. The machine can verify that declared
tags are mutually compatible; it cannot infer that the tags faithfully state
the mathematics.

This avoids the central category error in many automated-verification systems:
metadata consistency is not theorem truth.

The corresponding division of labor is also correct:

```text
gates:
    classify and block structural misuse

operating disciplines:
    detect numerical/content error and establish tag fidelity
```

That should remain the public explanatory center of the framework.

### Bridges are nodes

The file correctly identifies the largest v1.0 escape hatch: writing the word
“Independence,” “change of measure,” or “transfer certificate” in a metadata
field does not establish the witness.

Making every bridge a first-class graded node is the right abstraction. It
turns a bridge from an attestation into a dependency whose own warrant, source,
scope, and failure modes are visible.

### Retrodiction discipline

The v1.1 rule that a candidate gate must survive COMMON-MODE analysis of its
own evidence is an excellent self-application. Two examples derived from the
same misconception are not independent witnesses.

### Domain transfer

The software-latency example is useful because it shows that the framework is
not merely Q0 terminology with renamed nouns. Warm-cache versus cold-start law,
p50 versus p95 rung, and configuration-specific model tags are real transfer
axes.

## 1.2 Remaining conceptual defects

### Warrant and conditionality are still partially conflated

The uploaded v1.1 calls Primitive and Proven-Modulo “fully dependable.” That is
safe only when conditional debts propagate.

A primitive hypothesis can support a proved implication, but not an
unconditional theorem. Likewise, a Proven-Modulo dependency with conditions
\(H_1,\ldots,H_m\) cannot support a Proven parent unless those conditions have
been discharged. Otherwise conditions disappear through the graph.

Master v1.2 therefore treats condition sets as first-class graph data.

### COMMON-MODE's old ceiling can preserve an inflated grade

The uploaded rule says COMMON-MODE PROVISIONAL has ceiling “the claim's current
grade.” If the claim has already been labeled Proven, this ceiling does
nothing. The reconciled default cap is Derived.

### Witness operations need types

The uploaded composition lexicon does not list `UNION_BOUND`, even while its
worked example uses a union bound. It also says “union bound with disjoint
events,” although disjointness is unnecessary.

A witness now specifies both:

```text
operation:
    PRODUCT | UNION | AFFINE-ASSEMBLY | ...

witness type:
    INDEPENDENCE | UNION-BOUND | EXACT-ALGEBRA | ...
```

A union witness cannot license a product.

### F-1 needs two regimes

A reusable result needs an establishment regime and a deployment regime.
Without both, a transfer gate has nothing concrete to compare.

### Canonical hashing was underspecified

The uploaded file activates content hashes but does not define canonical
serialization. A user-supplied label such as `C#1` cannot detect mutation.

Kernel 2.0 hashes deterministic JSON containing every load-bearing field,
including pinned dependency edges.

### Three numerical gates are now earned

The uploaded v1.1 correctly staged new gates conservatively. Later frozen
failures now establish:

- ASSEMBLY;
- numerical UNIFORMITY / DOMAIN-EXTREMUM;
- BAND-PROVENANCE.

These are not convenience additions. They are required by concrete failures
that the first 13 gates did not mechanically catch.

---

# 2. PDF review

## 2.1 Identity

The PDF is not the uploaded Markdown v1.1. It is the earlier v1.0 text:

```text
13 gates + 1 schema field
mechanically checkable
free-text bridge-era schema
old grade rule
old PROVISIONAL reference
```

It must therefore be cited as v1.0 provenance, never as the current master.

## 2.2 Visual inspection

The document rendered successfully into 15 page images. It is visually
legible, text-based, and uses embedded monospaced fonts.

The layout is dense but stable:

- no clipped paragraphs;
- no black squares;
- no overlapping blocks;
- tables remain readable;
- page numbering is absent;
- no diagrams or figures require separate extraction.

The contact sheet shows a consistent single-column technical-manuscript layout.
Page 15 contains only the final adoption note and end marker, leaving large
unused white space.

## 2.3 Structural inspection

```text
pages:       15
size:        612 × 792 pt
encrypted:   no
forms:       none
attachments: none
annotations: none
outline:     none
scanned:     no
```

The lack of bookmarks is a usability defect in a long specification.

Parsers emit warnings for wrong pointing objects:

```text
7 0
16 0
46 0
```

The PDF remains openable, but the next generated version should be normalized
rather than copied from this object structure.

---

# 3. Gate Kernel v1.1

## 3.1 Strengths

v1.1 is a compact and readable prototype. It already contains:

- the 13-gate vector;
- constitutional dependency checks;
- stale-edge labels;
- a useful demo graph;
- explicit scope language saying the core cannot be checked;
- clear regression examples for missing composition, measure mismatch, and a
  stale pinned hash.

It compiles and executes successfully.

## 3.2 Structural limitations

It is not a registry implementation:

- no graph-wide well-formedness validation;
- no cycle rejection;
- no strict serialization;
- no schema version;
- no canonical content hash;
- no all-or-nothing load;
- no root-level admission modes.

Its gate shells also inherit the deeper 1.x issues documented below for v1.2.

v1.1 is best retained as the pedagogical prototype from which the registry
line developed.

---

# 4. Gate Kernel v1.2

## 4.1 Improvements over v1.1

v1.2 adds genuine engineering value:

- `validate_graph`;
- JSON serialization;
- a hash-indexed registry;
- leaf-up admission;
- cycle checks;
- malformed-graph demonstration;
- round-trip intent.

The code compiles.

## 4.2 Portability failure

Running the file as shipped fails at:

```text
/mnt/user-data/outputs/q0_registry.json
```

because the directory is hard-coded and absent in the current environment.

A library-quality executable must accept a path argument or use a path relative
to the working directory/script.

## 4.3 Red-team result

Seventeen frozen adversarial tests were executed. Every one reproduced the
target implementation gap:

```text
tests:                       17
vulnerabilities reproduced: 17
secure rejections:           0
harness errors:              0
```

This does not mean the kernel has no useful checks. It means its current shell
is materially weaker than the uploaded v1.1 contract.

## 4.4 Detailed implementation findings

### DOMAIN silently skips missing information

The gate deliberately treats absent dependency domains as nonviolations. It
also checks only parameters that happen to exist in the dependency.

Therefore a parent quantified over \(r,b\) can pass using:

- a dependency with no domain;
- a dependency that declares only \(r\).

This contradicts the framework's “name the restriction” discipline.

### MARK can be satisfied by the parent itself

The gate computes

```text
required_marks - parent.supplied_marks
```

rather than collecting marks from evidence dependencies.

A parent can therefore write both required and supplied marks even when every
support node is unmarked.

### Bridge nodes are graded but untyped

Any sufficiently graded `CHANGE_OF_MEASURE` edge can bridge any measure pair.
The node does not declare:

- source measure;
- target measure;
- source conditioning depth;
- target conditioning depth;
- domain scope.

The same problem affects model and mark transfers.

Grading a bridge is necessary, but not sufficient. It must also be typed.

### COMPOSITION does not know the operation

The parent stores only `is_combination=True`. A union-bound node can therefore
license a product, and an exact-algebra witness can license an unrelated
statistical assembly.

### COMMON-MODE is a Boolean attestation

The two fields

```text
error_independence_arg
external_crosscheck
```

carry no instrument IDs, shared components, tested error channel, or evidence
hash. Setting a Boolean to true passes the gate.

### Provisional ceilings do not propagate

The registry records a provisional ceiling for `C_const`, but parent
admissibility reads the dependency's declared grade rather than the recorded
ceiling.

A Proven parent can therefore depend on a Proven-but-provisional node and
remain Proven.

### Hash drift is label drift, not content drift

`content_hash` is arbitrary input. The kernel never recomputes it from the
claim.

A statement can be materially changed while preserving the same label and all
parents remain hash-consistent.

### PRECEDENCE ignores the root itself

Reachability starts from descendants. A killed root with no children passes.

### CORE CLOSURE checks only `Grade.OPEN`

A closed core can retain a conditional Conjecture and pass because Conjecture
is not Open.

### Registry admission skips graph validation

`Registry.admit` checks duplicate hashes, unresolved dependencies, and cycles,
but does not invoke the status/grade checks in `validate_graph`.

A live claim graded Killed was admitted as:

```text
PASS at Killed
```

### Registry loading is silently partial

`Registry.load` calls `admit` and ignores its return value. A malformed claim
can be dropped while the function returns an apparently valid smaller
registry.

### Schema version is ignored

`graph_from_json` never checks the `schema` value.

### Topological ordering silently drops cycle nodes

The helper returns an incomplete order rather than raising.

### Verdict language conflates storage and promotion

The demonstration prints:

```text
T_main PASS at Proven-Modulo
```

while also reporting:

```text
Open dependency: R0
conditional debts: C_const, R0
C_const ceiling: Derived
```

A registry must distinguish archival storage, conditional theorem status, and
unconditional promotion.

---

# 5. `q0 registry.json`

## 5.1 It is a demonstration, not the Q0 theorem registry

The registry contains:

```text
C = 0.4127
r ∈ [0,1]
b ∈ [0,3]
generic inner/far/boundary claims
```

These are not the constants or domains of the frozen Q0 Rate Program.

The file must not be used as a mathematical source of truth.

## 5.2 Critical R0 collision

The registry uses `R0` for:

> an unbounded remainder in a pairing-failure expansion.

The Q0 mathematical program uses `R0` for the Gaussian-Sard/Morse-Smale
condition.

This collision is severe enough that the migrated registry renames the demo
object:

```text
PAIRING_REMAINDER_CONDITION
```

## 5.3 Unsupported source grades

The registry labels inner, far, and boundary bounds Proven but supplies no
proof objects, constants, or source IDs.

`C_const` claims “closed form + numerical value” but supplies only 0.4127.

The migration therefore preserves the claims as explicit open conditions rather
than silently endorsing their grades.

## 5.4 Orphaned fixtures

`pinned_input` is disconnected. `cm_node` is attached to the root even though
none of the root's support nodes has the Gaussian-pinned measure.

The migrated bridge and pinned input remain as disconnected historical
fixtures.

## 5.5 Missing theorem machinery

The root lacks:

- a coefficient assembly;
- a full-domain coefficient certificate;
- recomputed endpoint evidence;
- a common-mode certificate for C;
- evidence hashes for the region bounds.

The migrated graph is therefore:

```text
archive-valid:   yes
candidate-valid: no
promotable:      no
```

That is an honest terminal result for the file as supplied.

---

# 6. Reconciled specification and kernel

## 6.1 Master v1.2

The new master combines:

- the uploaded v1.1 shell/core honesty;
- graded bridge nodes;
- condition propagation;
- canonical hashes;
- strict registry semantics;
- the three retrodiction-qualified numerical gates.

## 6.2 Gate Kernel 2.0

The schema revision is intentionally major because the changes are breaking.

Kernel 2.0 implements:

- separate status, warrant, assumption role, and reasoning mode;
- strict parameter domains;
- establishment and deployment regimes;
- evidence-cone mark checking;
- typed bridges;
- operation-specific witness nodes;
- canonical SHA-256;
- pinned dependency hashes;
- all-or-nothing registry loading;
- explicit archive/candidate/promotion admission;
- provisional ceiling propagation;
- condition-set propagation;
- ASSEMBLY;
- DOMAIN-INFIMUM/UNIFORMITY;
- BAND-PROVENANCE;
- no hard-coded output path.

## 6.3 Independent validation

The 2.0 battery contains 36 cases. All 36 returned the preregistered outcome.

The battery includes every reproduced v1.2 escape and positive controls for:

- typed measure/model bridges;
- operation-compatible product witnesses;
- independent common-mode certificates;
- explicit conditional theorem use;
- conservative assemblies;
- full-domain extrema;
- complete band provenance;
- second-domain matrix perturbation;
- clean registry round-trip.

---

# 7. Final dispositions

| Artifact | Final disposition |
|---|---|
| PDF v1.0 | provenance-only; regenerate cleanly |
| uploaded Master v1.1 | conceptually authoritative input; superseded by merged v1.2 |
| kernel v1.1 | pedagogical prototype |
| kernel v1.2 | superseded executable; preserve red-team record |
| q0 registry.json | legacy demo; not theorem authority |
| Master v1.2 | current conceptual specification |
| Gate Kernel 2.0 | current executable shell |
| q0_registry_v2_0.json | strict archival migration; not promotable |

---

# 8. Next research actions

The file-set review changes the order of successor work.

First, all Q0-REFEREE and Q0-SHARP contract objects should be migrated into
Gate Kernel 2.0 so that conditions, bridges, assemblies, and residuals cannot
disappear.

Second, `UB_G_RESIDUAL_UNIFORM` remains the immediate theorem gate. No decimal
upper coefficient should be republished until Gamma and collar residuals occur
inside the actual assembly expression.

Third, BR-MARK should consume the exact six-pin covariance factorization
already derived in C095 and produce typed source/target mark-transfer
certificates rather than prose bridges.

Fourth, the framework should receive a clean v1.2 PDF with:

- normalized object structure;
- bookmarks;
- page numbers;
- a table of contents;
- no stale v1.0 wording.

Fifth, COMMON-MODE and HEURISTIC-BRIDGE should retain operational enforcement
while their framework-level retrodiction evidence remains provisional.

**End of C096 deep review.**
