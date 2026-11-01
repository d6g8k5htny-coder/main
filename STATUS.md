# CBZ-0 execution status

**State: Partial Baseline; CBZ-0 is not established.**

This entry adopts the owner's expanded [Master Directive](governance/MASTER_DIRECTIVE_20261029_SWARM.txt)
and [team protocol](GOVERNANCE.md#team-structure-and-coordination).
The inherited scientific material below is explicitly the 29 September
snapshot, not a fresh CLOSED decision or an empirical-validation receipt.
Its mathematical statements, source cuts and review limits are preserved.

## Current verification boundary

The full federation inventory, proof dispositions, required tooling admission,
cross-repository synchronization and final common G0-G11 pass remain unfinished.
Source-only checks and individual successful jobs do not complete the twelve
baseline criteria. The [foundation candidate #695](https://github.com/d6g8k5htny-coder/main/pull/695)
is a candidate handoff, not the current default-branch baseline.
The pre-integration Main source was `11e28815163e15223e8827fe6ff47a1faaf36ebd`.
[PR #713](https://github.com/d6g8k5htny-coder/main/pull/713) merged on 2026-10-31 at 10:18:44 UTC
as `9abd2c100ae105103945073a7fa6c47b15679269`; its qualification evidence
remains bound to the sources actually checked, not automatically to later compositions. The proof catalogue and other
foundation material in #695 are pending, not an admitted default-branch baseline.
This intermediate entry is not a completed cold-start acceptance test.

## How to verify

The existing formal component can be run with
`python -B tools/formal_gate_check.py --run-lean` using the toolchain pinned
in `formal/lean-toolchain`; consult [the formal guide](docs/FORMAL_VERIFICATION.md)
for setup and exact scope. Retain its actual build, kernel, axiom and negative
control outputs. This command alone is not the full G0-G11 federation run.
For the restored bounded replay controls in this PR, use Python 3.11.16 and
run `python3.11 -B -S run_mutations.py --output /new/output` from
`cbz/evidence/kernel/additional/replay-tool/`; see its [README](cbz/evidence/kernel/additional/replay-tool/README.md)
and [governance scope](GOVERNANCE.md#restored-bounded-replay-controls).
This checks the documented Python validator and its three mutants per mode;
it neither authenticates a historical execution nor performs a Lean kernel replay.
The retained Python 3.9 mutation-grader failure is not a passing qualification.

The empirical fuzzing harness and contextual federation crawler are required
but not admitted by this entry. Applicable source-bound empirical validation is
necessary for a new CLOSED decision, alongside the formal gates; unsupported,
empty or missing coverage cannot silently pass. #712's original green controls
do not resolve its later graph and Lean-context findings. A bounded repair is
not full AST/type/theorem synchronization. New gating tools require documented
commands and self-verification before admission; unavailable or unsupported
coverage remains explicit. No documented command here runs all G0-G11 yet.

The empirical statement check for core-Lean arithmetic skeletons is a registered
bounded component on `main` ([PR #726](https://github.com/d6g8k5htny-coder/main/pull/726),
merged `d0d384b18a409ed51061458287dde22816782210`). From a checkout, run:

```
python3 -B -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -O -S -m unittest tests.test_cbz_statement_fuzz -v
python3 -B -S tools/cbz/statement_fuzz.py --package formal --selftest --json report.json
```

It evaluates every registered Lean statement of a formal package independently in exact
arithmetic with Lean-core semantics and grades itself by a mutation contract; the
[registration](GOVERNANCE.md#empirical-statement-check-for-core-lean-arithmetic-skeletons-bounded-registration)
records commands, hosted self-verification and limits. Observed: main `formal/` 31/31
PASS; the Math- core's 63 real-analysis statements are UNSUPPORTED, which is the honest
coverage of the empirical layer today, not a pass. A PASS is statement-level evidence
only: not kernel evidence, not scientific acceptance, not the common G0-G11 run.

The finite MomentTail component is a registered bounded candidate in
[Math PR #545](https://github.com/d6g8k5htny-coder/Math-/pull/545), exact source
`e1cf617b6bc979b75d420f28df17ddad59426522`. In an isolated checkout of that Math
source, run all six commands:

```
python3 -B cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -O cbz/finite_moment_tail_20261030/test_adapter.py -v
python3 -B -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B -O -m unittest discover -s formal/tests -p test_moment_tail.py -v
python3 -B cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
python3 -B -O cbz/finite_moment_tail_20261030/adapter.py --source-root . campaign --seed 20261030 --count 10019
```

The [contract](https://github.com/d6g8k5htny-coder/Math-/blob/e1cf617b6bc979b75d420f28df17ddad59426522/cbz/finite_moment_tail_20261030/CONTRACT.json)
and [bounded governance registration](GOVERNANCE.md#finite-momenttail-empirical-component-bounded-registration)
record the six-command hosted job, exact source/test evidence and limits.
Its CPython 3.11.16 run actually passed; it covers one finite specialization,
not a federation-wide empirical gate or a new CLOSED decision. Existing pins
and the global unsupported inventory are unchanged by this registration.

## Team and reconstitution

Use the discovery, proof-closure, tooling, verification, integration and
documentation roles in [GOVERNANCE.md](GOVERNANCE.md). Roles are fluid;
ownership is not a lock. Start from the current pins, dependency manifests
and registry graph, then assign the highest-value closable work to available
agents. Record actual actors, source cuts, conflicts, decisions and handoffs
in [DECISIONS.md](DECISIONS.md) and [CONFLICTS.md](CONFLICTS.md).
The following is a bounded execution observation at 2026-10-30 00:13:19 UTC,
not an exhaustive list of all agents or a claim of independent scientific review:

| Actual actor | Current role and delivered scope | Source or execution boundary |
|---|---|---|
| OpenAI/Codex chat `01a11d37-16a7-7d32-a8b1-8131336243bf` | Documentation composition; R3 delivery coordination | This additive documentation contribution; [task record](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6511442656) |
| `/root/cbz712_r3_registry_repair_20261029`, delegated in that chat | Tool repair | Two-file staged R3 tree `d6f8eb15af4216eb93352b96382264e5d39525cf` on `15e8f689`; nine focused and original 43 controls pass in both modes; original six retain five separate failures |
| `/root/cbz712_r3_source_review_20261030`, delegated in that chat | Fresh nonauthor engineering verification | Exact R3 tree only; independent five real-Git/CLI controls per mode; same provider, no organizational independence or scientific acceptance |
| OpenAI/Codex chat `01a0bbb5-2fcb-77f0-b78b-4d220ddd7ab2` | Integration and replay-source recovery | [PR #713](https://github.com/d6g8k5htny-coder/main/pull/713); source-qualified handoff and final composed hosted checks remain distinct |
| OpenAI/Codex chat `01a11c48-e9e4-7113-a03f-24b2155da25e` | Tool repair and compatible composition | [#712 source repair pickup](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511525318); separate [PR #715](https://github.com/d6g8k5htny-coder/main/pull/715) trigger contribution |
| OpenAI/Codex chat `01a103c8-7a8e-7461-8e9f-36e3b02757c5` | Completed bounded adversarial qualification | [#712 original six findings](https://github.com/d6g8k5htny-coder/main/pull/712#issuecomment-6511043716); [bounded qualification completed and only that pickup released](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6511605192); this does not assert that the actor is globally idle |

Re-read native records and exact refs when reconstituting these roles; names
and a shared account are not proof of current capacity or source qualification.
At phase boundaries reconcile evidence and re-rank work. A final baseline
must record its observed team composition and runnable final gate commands;
this role description does not claim that acceptance has occurred.

---

## Historical scientific status snapshot

**Snapshot date:** 2026-09-29  
**Purpose:** one human-readable status page for the public front door.  
**Rule:** this page summarizes source-bound review outcomes; it is not an independent promotion register.

The D5, SARD-G successor and C6 rows below use the [29 September source-bound reconciliation](docs/STATUS_RECONCILIATION_20260929.md), read at Math- `4d062e2a976abc8800a1ae63a49a943526b6bc0d`. They do not execute Math- GRAPH transitions or refresh the museum/shop's explicitly pinned historical snapshots.

## ACCEPT — scoped

| Object | Accepted scope | Source and review | Explicit limits |
|---|---|---|---|
| **D1 — parent lifetime theorem (Theorems A, B, C)** | Theorem A `0 <= 1-p_r <= C r^3`, uniform over compact births, compact positive gaps and all frames; Theorem B compact-mark candidate/elder densities `~ c_{B,K} ell^(-1/3)` with difference `O(ell^(2/3))`, for `B` and `K` of positive length; Theorem C unrestricted leading term `c_{d,L} ell^(-1/3)`. Fixed `d>=2` and `L`, existential constants, read with the congruence erratum, the Section 9 replacement v1.1, wording W1 and `r < L/(4 sqrt 2)` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md) · [reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md) · [Math-#126](https://github.com/d6g8k5htny-coder/Math-/pull/126) | No numerical `C`, `r_*`, `z_*` or coefficient; no unrestricted difference rate; this row is confined to the Math- `8224783` reconciliation, whose lower composition is planar (`d=2`); the separately reviewed every-fixed-`d` lower bound `1-p_r >= c r^3` was integrated afterwards by [Math-#129](https://github.com/d6g8k5htny-coder/Math-/pull/129) at `e7f8aca41c308f8ad8614b587f75916943c86f27` and is not part of this row's reconciliation scope; no RN/24-jet certificate |
| **D2 — unrestricted lifetime remainder** | Theorem R at its stated existential `O(1)` remainder scope: candidate and elder densities have leading `c ell^(-1/3)` term with bounded remainder for sufficiently small lifetime | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/67#issuecomment-5841782206) | No numerical remainder constant or numerical lifetime cutoff; no RN/JETMOD closure |
| **D3 — SIDE24 coefficient calculation** | The coefficient expression for SIDE24 in dimensions 2 and 3, including cone moments, all-direction periodization comparison, covariance/Schur transfer, and outward special-function arithmetic | [Proof/package](https://github.com/d6g8k5htny-coder/Math-/tree/main/coefficients/side24_v1) · [review](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841269490) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/65#issuecomment-5841779222) | Coefficient arithmetic acceptance is separate from any imported parent theorem and from a finite-radius error band |
| **D4 — fixed-remote RN count theorem** | Fixed positive spatial separation `rho`, fixed witness separation `eta`, between-pin height window; conditioned covariance, three determinant factors, endpoint normalizer, contact kernel, and fixed-separation factorial moments | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/remote_window_20260924/PROOF.md) · [review](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841783172) · [reconciliation](https://github.com/d6g8k5htny-coder/main/issues/76#issuecomment-5841861362) | Does not cover shrinking `rho` or `eta`, pin neighborhoods, intermediate scales, all remote heights, or global RN/24-jet closure |
| **D6 — P15 full-price theorem** | Realized disjoint capacity/clutter family with `d_i >= 2`, all independent probabilities, `c_v <= phi(p_v)`, same palette `K >= K_H(d)`, sharp `rho*=1/(3-log(3e-2))` | [Proof](https://github.com/d6g8k5htny-coder/Math-/blob/main/frontiers/full_price_20260924/PROOF.md) · [review reconciliation](https://github.com/d6g8k5htny-coder/main/issues/74#issuecomment-5842112010) | Demand one remains an obstruction; no arbitrary-downset or unrestricted prize-problem extension |
| **D5 — planar pin neighborhoods and global window first moment** | Original weighted six-observation pin law, fixed torus and compact positive-gap marks, all planar frames and indices, counting only critical points other than the conditioned pins M and S: all-height punctured pin disks and fixed scaled-ball/collar first moments are `O(r^3)`; nested `kappa*r^2` microdisks are `O(r^5)`. Intermediate shells and the torus-wide first moment are `O(r^3)` **for the stated between-pin height window**. Existential constants only | [Source-bound reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/79f9a73596b1ce925034bd426c005a3251693c1b/reviews/d5_reconciliation_20260929/RECONCILIATION.md) · [planar P2 continuum review](https://github.com/d6g8k5htny-coder/Math-/pull/141#issuecomment-5894512272) · [source ledger](docs/STATUS_RECONCILIATION_20260929.json) | Planar `d=2`, fixed torus, gap bounded away from zero; no all-height global count, higher-dimensional transfer, numerical RN/24-jet certificate, elder-pairing conclusion or sharp C6 factorial order |
| **SARD-G — qualitative robust-chart successor** | The unconditioned planar periodized field and each fixed admissible original two-pin law with its original determinant tilt are almost surely Morse with distinct critical values and no saddle-saddle gradient connection, hence planar Morse-Smale in the completed-measure sense. Uses first-closed-contact charts, fixed atlas axes, corrected v2 A2, R3/R4 and the pinned-law transfer with its genericity input | [Integrated reading rule](https://github.com/d6g8k5htny-coder/Math-/blob/e947ade51defcec1fb75444efbe91ac7fcf43f60/reviews/sard_closure_integration_20260929/READING_RULE.md) · [transfer review](https://github.com/d6g8k5htny-coder/Math-/blob/e947ade51defcec1fb75444efbe91ac7fcf43f60/reviews/sard_closure_integration_20260929/TRANSFER_REVIEW.md) | Per fixed admissible planar law; historical A1/A6 remains AMEND as written. No common probability-one set over uncountably many targets, C103 regional estimate, R0 Boolean transition, near-connection rate, RN/JETMOD closure or quantitative pin-coalescence uniformity |
| **C6 — Fourier count tail and planar factorial upper bound** | At fixed `d>=2`, torus and compact positive-gap marks, Theorem F gives count tail `C exp(-c x^(2/d))` under the original pin law and determinant tilt. Theorem W composes with the dimension's global window mean. In `d=2`, the reviewed planar mean yields `E[N(N-1)] <= C r^3 log(1/r)` and fixed higher factorial orders `C_p r^3 log(1/r)^(p-1)` for the stated window count N, excluding M and S; the tail cap includes all critical points | [Exact reviewed proof](https://github.com/d6g8k5htny-coder/Math-/blob/4d59319b7fbe0790ebd5a1482bc91c4d9cc2d4f3/frontiers/c6_fourier_cutoff_20260929/PROOF.md) · [nonauthor review 5356335148](https://github.com/d6g8k5htny-coder/Math-/pull/148#pullrequestreview-5356335148) · [reading note](https://github.com/d6g8k5htny-coder/Math-/blob/4d062e2a976abc8800a1ae63a49a943526b6bc0d/reviews/c6_fourier_completion_20260929/READING_NOTE.md) | Higher-dimensional factorial bounds remain conditional on `G_d`; sharp Gaussian `O(r^3)`/`Theta(r^3)` and the marked-Palm route are not accepted here. No numerical cutoff or parameter-uniform constants; the abstract barrier is not a Gaussian counterexample |

## AMEND / open

| Object | Current reason | Source |
|---|---|---|
| **D5 — remaining quantitative obligations** | The scoped planar first moments above do not complete the historical numerical RN annulus partition/area-weighted budget, all-radius RN or RN/24-jet certificates. Separately, Math-#141's P_d/C_d/I_d/G_d continuum counts were accepted and merged for fixed `d>=2` and torus, compact birth and positive-gap marks with `k>=k_->0`, sufficiently small `r`, the original `W_r/Z_r`, and counts excluding M and S. Pin/collar estimates are all-height; intermediate shells and the global first moment retain the stated between-pin height window. Constants remain existential. These later results are separate from the planar reconciliation above. Proposed GRAPH transitions have not been executed by this summary. | [D5 reconciliation and exclusions](https://github.com/d6g8k5htny-coder/Math-/blob/79f9a73596b1ce925034bd426c005a3251693c1b/reviews/d5_reconciliation_20260929/RECONCILIATION.md) · [merged #141 proof](https://github.com/d6g8k5htny-coder/Math-/blob/e1ca400b3414e5a7bec13b91bc4f1e5293df32be/frontiers/d5_dimension_lift_20260929/PROOF.md) · [A/B review5357858391](https://github.com/d6g8k5htny-coder/Math-/pull/141#pullrequestreview-5357858391) · [C review5357882570](https://github.com/d6g8k5htny-coder/Math-/pull/141#pullrequestreview-5357882570) (OpenAI, source-exposed) |
| **SARD-G A1/A6 — original formulation** | Remains AMEND as written. The separately reviewed robust-chart/fixed-frame successor establishes its own qualitative fixed-law theorem; it does not validate the defective original A1 premise or supply a regional C103 estimate. | [Original-formulation review](reviews/sard_g_successor_a1_a6_20260926/REVIEW.md) · [historical PR](https://github.com/d6g8k5htny-coder/main/pull/122) |

## Engineering only

These are useful infrastructure, but they carry **no theorem-acceptance meaning**.

| Surface | What it provides |
|---|---|
| **[query-](https://github.com/d6g8k5htny-coder/query-)** | Public read-only package and exact-source lookup; default commit `76e1ca09` includes the package and the canonical no-dependency unittest command in CI |
| **[Universal-Law-Workspace](https://github.com/d6g8k5htny-coder/Universal-Law-Workspace)** | Supporting federation/navigation map with `scientific_status_authority: false` |
| **main CI/navigation** | Link, structure, and engineering checks; green runs are not mathematical review |

## Where to read — public source custody

The two human maps are this file and the [Math- proof index at the pinned public Math commit](https://github.com/d6g8k5htny-coder/Math-/blob/82247833b5dd58e04291d55353b52938d73ec614/PROOF_INDEX.md). Neither file replaces the underlying proof or review.

For exact source identity, use [`docs/public-math/sources.json`](docs/public-math/sources.json). It indexes **2,138 intended public text artifacts** and points to 14 JSON shards. Each artifact row carries:

`repository + path + 40-char commit + Git blob + bytes + sha256`.

The frozen inventory includes the public `Math-` and `main` mathematics selected for publication and explicitly excludes sandbox, quarantine, and personal paths. It is source visibility/custody only.

The machine catalog at [`meta-framework/registry.json`](https://github.com/d6g8k5htny-coder/meta-framework/blob/main/registry.json) currently contains only public GitHub sources (`Math-` plus one declared public `google-drive` replica). The query portable stubs likewise point only to public `Math-` commits. Candidate query stubs are not silently promoted into the catalog.

### Default-branch boundary

- Pinned Math reading checkout: `82247833b5dd58e04291d55353b52938d73ec614`.
- Pinned query checkout: `89456c5033173c69a304c5fffa781888185675cc`.
- [Math- PR #70](https://github.com/d6g8k5htny-coder/Math-/pull/70) landed nine byte-custody copies at `9d7b6802424fb4715b31999066aafca8ee2f3cca`; [PR #75](https://github.com/d6g8k5htny-coder/Math-/pull/75) added their proof-index pointer. These are public default-branch imports, **not newly accepted mathematics**. Source self-labels were not adopted.
- The `main#59` issue-body file in PR #70 is explicitly a transcription, not original source bytes, so it is not counted as byte custody.
- Open draft AMEND/review PRs remain visible but are not represented here as landed mathematics.

## Reading rule

When a reviewed child result imports a parent that remains AMEND or open, keep those facts separate. “ACCEPT — scoped” never means “everything upstream and downstream is accepted.”
