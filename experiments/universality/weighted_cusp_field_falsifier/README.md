# Weighted-cusp field comparison: source-only development snapshot

This directory implements a deterministic finite-window comparison between independently computed analytic probability bounds and certificates obtained from an actual integer field grid. The intended campaign has four laws and eighteen events, giving seventy-two comparisons. **No completed seventy-two-row prediction or production comparison is reported in this snapshot.** An unresolved result is an allowed outcome; development tests do not establish a scientific falsification or universality result.

Publication status: proposed source-only draft stacked on the unchanged source commit `b79ca1f6ea5f3cfab95a29401ebd16d9b5ee565b` associated with PR606. The proposal does not retarget or merge that existing work. Evidence cut: 26 October 2026, 18:05 UTC. Later campaign outcomes require their own dated, source-bound evidence.

## Model and observable

The model is a smooth perturbation of a two-dimensional periodic field, with a separately documented whole-field definition and exterior argument. Its local coordinates use `A = 2^-16`, `b = 4/9`, and controls

```
u = 3 A^2 Q^2
v = epsilon 2 A^3 Q^3 (1 - Z^8)
```

Here `Q,Z` lie in `[0,1]`, the sign is fair, and the densities are `5 Q^4` and `k Z^(k-1)` for `k = 8,12,15,16`. A rectangular control cell with one sign has exact mass

```
(1/2) (Qb^5 - Qa^5) (Zb^k - Za^k).
```

The declared events concern the positive finite ordinary-H0 bar: lifetime `0 < ell/A^4 <= tau`, optional shortest-torus birth-to-killing-saddle distance `r/A <= r0`, and optional centered birth height `(B-b)/A^4 <= 0`. The lifetime thresholds are `1/4,1/16,1/64`; separation is unrestricted or bounded by `1/16,1/8`; the birth condition is absent or present. The essential class is excluded. The same cells serve all four probability weights, so these are not four independent samples.

The conventional mathematical source gives the complete-field reduction and analytic probability formula. It is manually derived, source-exposed work, not a formal-kernel theorem or an independently organized scientific replication. The numerical observations themselves must select the bar and merger through the field-grid route.

## Separate analytic and observation routes

`prediction.py` computes exact outward probability intervals by a scalar analytic root calculation, lifetime and physical-distance caps, and weighted endpoint integration. It retains interval witnesses and cutoff/mesh/budget uncertainty. It does not consume field observations, uniform cell root certificates, UF/BFS output, or a fitted coupling tolerance. Canonical replay verifies reproducibility of the same algorithm; it is not an independent numerical implementation.

`field.py` defines the field and physical coordinate calculations. `field_certificate.py` constructs the fixed64-node integer grid from uniform whole-cell root brackets, evaluates its rational core values, and uses the existing unchanged `exact_h0` engine for union-find selection and separate connectivity verification. Retained certificates include nodal residuals and integer divisions, the barcode, actual path/cut merger witnesses, root brackets, physical marks, and derived coupling constants. The exterior argument is required to relate this graph carrier to the complete field; the grid is not claimed to uniformly tile the physical torus.

`artifact.py` preserves exact integers, Fractions, tuples, lists, string/integer dictionary keys, booleans, and null in a bounded canonical encoding. `run_falsifier.py` joins the routes only after validating the prediction artifact, frozen input identities, expected source closure, external receipt fields, and freeze order. A real production worker must bind the actual first field-attempt sequence to the completed prediction freeze. Receipt fields are checked evidence supplied by that worker; booleans and hashes alone are not proof that an external review or execution occurred.

## Uncertainty, costs, and retained output

Every durable live partition retains the full probability mass of both signs. A cell that cannot establish roots, coupling, a physical mark, or an event boundary contributes the conservative interval `[0,1]` for that event. A positive-area boundary cell does not become negligible merely because a null boundary passes through it. There is no conditioning or renormalization over successful cells.

The journal retains the initial partition, attempt starts, complete attempted certificates or failure records, split commits, and final state. A parent remains live until both child attempts and the split commit are durable. Interrupted child work remains retained and charged. Live summaries use references to raw records; a missing complete record or an invalid checkpoint reference withdraws raw-output qualification.

Costs distinguish attempted fields, created cells, sampled geometry vertices, root work, and helper/replay work. A sampled-vertex counter is not a total CPU-work counter. An exception after partial work leaves affected costs unknown; it does not manufacture zero cost. Resource, deadline, and record-size limits can stop refinement with full uncertain mass. The selected ceilings do not promise a precision level or runtime fit.

For an admitted positive prediction lower bound `PL`, an overlapping observed interval receives a provisional finite-window consistency label only when its width is at most `min(1/64, PL/4)`. A broad overlap is insufficient. Exact-zero consistency requires both intervals to be exactly `[0,0]`. A qualified disjoint interval is a candidate discrepancy requiring independent examination of arithmetic, implementation and coupling. Numerical comparison status, full table coverage, raw retention, external runtime qualification, and scientific acceptance are distinct fields.

## Development evidence and reproducibility limits

The recorded normal and optimized development suites each ran92 test methods, with105 declared subcases, zero reported failures/errors and no reported source drift. The normal run reported6.74 seconds for the whole parent and the optimized run6.90 seconds. These observations belong to exact immutable execution capsules held privately; they do not automatically apply to modified source or a different launch method.

A separate tiny development control retained18 field attempts, including two resolved cells, and four INCONCLUSIVE comparisons. Its owned child and native parent exited successfully; reported whole-parent time was0.59 seconds. At this evidence cut, the separate full-journal result review remained pending. This control is neither the eighteen-event campaign nor a completed production comparison. Its synthetic external receipt mode cannot qualify production.

The tests require captured `CAMPAIGN_SPEC.json` bytes supplied through the controlled `__captured_data__` interface, and imports also require the existing pinned `exact_h0` dependency. Strict production likewise checks the original captured campaign identity. No ordinary import, package-install, test-discovery, or command-line reproduction recipe has been verified for this publication snapshot. No untested command is presented as a reproduction instruction. A public portable runner and its independent launch verification remain separate work.

The original `CAMPAIGN_SPEC.json` is retained byte-for-byte because its hash participates in the captured-input contract. Its old date, claim number, `source_preparation_only`, `NOT_RUN` values and hard deadline describe the historical source selection; they are not current operational instructions or current result fields. The scientific event table is unchanged.

## Source documents and publication boundaries

The proposed explanatory companions are sanitized derivatives named `DESIGN_PROOF_PUBLIC.md`, `REQUIREMENTS_INTERFACES_PUBLIC.md`, and `PREDICTION_SPEC_PUBLIC.md`. They must preserve the original mathematical statements while removing private workflow identifiers and labeling historical execution language. They are **not** the exact pinned source bytes consumed by the implementation. The original private-source identities are:

| Original source | SHA256 |
| --- | --- |
| Mathematical field/coupling design | `5a20d81b57a216212f88940b79317f1e3593456855914f206a031e50dfb7767d` |
| Required interfaces | `4393d967cabb3ff67852e4b2b3888478030cabad88bb6a37d59e338b481fcb1e` |
| Analytic prediction specification | `e1dd749117f141a0f05e9c6c36bcac154efa137164177a0304f741632796a3c5` |
| Original captured campaign | `be65c90e312a264d48e58ece142c6d2d93b1c73e564e08ada02ad656f9fefad3` |

Historical docstrings in the unchanged source may describe the authoring stage as unexecuted. The dated evidence paragraph above supplies the later bounded development status without changing those tested bytes. Publication of source and explanations does not publish the private execution capsules or establish their public reproducibility.

No Gaussian, higher-homology, PRM, blind-replication, formal-kernel, universality, asymptotic-remainder, or scientific-acceptance claim follows from this draft. The full research objective remains unmet.
