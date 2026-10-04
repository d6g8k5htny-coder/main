# Lean audit owner: source coverage and handoff

**Accepted role:** `/root/lean_audit_owner`, OpenAI/Codex, on 4 October 2026.
The agent audits the transfer of public mathematical sources into Lean and
coordinates the remaining coverage. It is separate from the UI owner and the
formalization author. It is a task-local workspace agent, not an installed Mac
agent, background service or new scheduler. Source and prior-review exposure is
disclosed; this is not a blind audit. Organizational-independence credit is 0.
Scientific effect of this handoff: **NONE**.

The outgoing transfer is recorded in [main #229 handoff5975087739](https://github.com/d6g8k5htny-coder/main/issues/229#issuecomment-5975087739). The separate [UI/UX owner handoff](UI_UX_OWNER.md) covers the public interface. This document supplies a reusable role and starting queue; it does not install a Mac-local runtime.

This is an operational handoff and a dated coverage map, not another scientific
status register. Follow the [current workflow](../../governance/OP-WORKFLOW-20260930.md),
[formal guide](../FORMAL_VERIFICATION.md) and
[required-check contract](../FORMAL_REQUIRED_CHECKS.md). Existing sources,
authorship, failed outcomes, scope exclusions and completed reviews are retained.
Dylan's standing authorization permits the work without another personal-reading
approval queue.

## Ownership and boundary

The audit owner accepts responsibility for:

- identifying the exact informal statement, hypotheses, source version and
  consumed dependencies for each proposed transfer;
- mapping that source to actual Lean declarations and elaborated types;
- distinguishing a proved conclusion from an imported premise;
- checking source identity, kernel evidence, axiom reports, negative controls
  and statement alignment as separate evidence;
- giving a bounded disposition for missing coverage and preserving unresolved
  interfaces in the existing issue or PR;
- handing a concrete, source-pinned task to a formalization writer, then reviewing
  its unchanged frozen output without becoming its proof author.

The outgoing coordinator supplies the existing audit records and stops assigning
overlapping audit work after this handoff. Formalization authors retain their
branches. The new auditor does not acquire Claude's or Sol's writer/integrator
claims merely by accepting this role. Before any branch mutation, reconcile live
claims and obtain a scoped handoff, release, or confirmed lease expiry followed
by fresh refs and results. Record a precise bounded claim in the existing PR or
[main #229](https://github.com/d6g8k5htny-coder/main/issues/229).

The auditor can review disjoint scope without changing the writer's branch.
If it amends a proof, it becomes a contributor to that proof and cannot supply
its own nonauthor acceptance. UI/navigation, scientific registers, prizes,
`lemma_closed`, controlling graph verdicts and the active C127 research lane are
outside this role's write scope.

## Snapshot and evidence vocabulary

Read at **2026-10-04T00:42:09Z**:

| Repository | Exact source cut |
|---|---|
| `main` | `aab9f56aac4b137adad861660e492e35ea0ce900` |
| `Math-` | `42f19d7dc4109c2359520b7cbde24b6bb1fca110` |

These are dated identities, not assertions that branches will remain unchanged.
Refresh the selected object's current head and ownership before resuming it.

Use the project's existing `none`, `specified`, `proved` and `kernel-checked`
formal-progress vocabulary. Record alignment separately as pending, accepted
within scope, stale, or not assessed. A source being present, an endpoint being
ordered, a theorem being compiled and a scientific claim being accepted are
different statements. “Conditional/imported premise” below means the formal
conclusion uses the premise; Lean has not thereby proved that premise.

## Already supplied formal sources

| Object and authoritative map | Supplied formal coverage | Conditional or excluded content |
|---|---|---|
| [Math scalar pilot: 13 targets](https://github.com/d6g8k5htny-coder/Math-/blob/42f19d7dc4109c2359520b7cbde24b6bb1fca110/formal/SCOPE.md), [manifest](https://github.com/d6g8k5htny-coder/Math-/blob/42f19d7dc4109c2359520b7cbde24b6bb1fca110/formal/manifest.json) | Recovered GP-FOR-192 polynomial identities, scalar cancellation, quotient implications and power comparisons in separately named V2 compiler successors. Original source bytes remain preserved. | Numerator bounds, a positive lower normalizer, moment estimates and probabilistic hypotheses are supplied premises. Gaussian/Palm construction, Kac–Rice, SIDE24 enclosure, elder pairing and the lifetime theorem are not formalized by these 13 statements. The manifest retains `PENDING_INDEPENDENT_REVIEW`; this handoff does not create a replacement alignment vote. |
| [Main SIDE24 arithmetic: 31 targets](https://github.com/d6g8k5htny-coder/main/blob/aab9f56aac4b137adad861660e492e35ea0ce900/formal/SCOPE.md), [generated correspondence](https://github.com/d6g8k5htny-coder/main/blob/aab9f56aac4b137adad861660e492e35ea0ce900/formal/ALIGNMENT.md) | Exact integer/cross-multiplied arithmetic for pairing counts, constants, covariance arithmetic, cone-moment algebra and published decimal endpoint ordering. | Endpoint adjacency does not enclose the analytically defined coefficient. Gaussian identities, integral evaluation, special-function bounds and the full coefficient enclosure remain outside this arithmetic skeleton. |
| [Cap first exit v1.4: 53 manifest targets](https://github.com/d6g8k5htny-coder/Math-/blob/42f19d7dc4109c2359520b7cbde24b6bb1fca110/frontiers/cap_first_exit_lean_20261002/MANIFEST.json), [alignment interfaces](https://github.com/d6g8k5htny-coder/Math-/blob/42f19d7dc4109c2359520b7cbde24b6bb1fca110/frontiers/cap_first_exit_lean_20261002/ALIGNMENT.md) | Landed deterministic cap-to-maximin/older-component-level bridge, frontier transport, torus transport, ridge profile and assembly, with non-vacuity and necessity examples. Exact Lean blob `a5dd14bae0e01eeed033f7da6a473916338c7518`. [v1.4 scoped alignment review 5396110365](https://github.com/d6g8k5htny-coder/Math-/pull/246#pullrequestreview-5396110365) is retained. | The analytic outputs entering `ridge_capHyp`, concrete source coordinates, persistence-module identification and the probabilistic layer remain separate interfaces. The manifest supplies the count 53; a retained README execution paragraph still says 52 and is not used as the inventory authority. |

This handoff reads source maps and existing reviews. It does not claim a new
local Lean build, a fresh axiom audit, or a new analytic review. Existing trusted
run receipts remain the execution evidence for their exact tested commits. A
future integration must audit its actual current tested tree and landed run.

## Reviewed cap extensions awaiting integration

The current stack is **open and unmerged** at this snapshot. Its authored source
is `frontiers/cap_first_exit_lean_20261002/`. Completed reviews are retained; the
auditor must not request them again merely because an unrelated base moved.
The table identifies authored content and delivered review records; it is not
an automatic merge instruction or a scientific promotion.

| PR / version | Actual head | Manifest targets | Added scope and delivered alignment |
|---|---|---:|---|
| [253 / v1.5](https://github.com/d6g8k5htny-coder/Math-/pull/253) | `f60dba41b237e6ffbf1d6894c106979db4a0c17f` | 74 | Slice support, Hermite bound, ridge-chain fragment and curved-side constants. [Review 5402591138](https://github.com/d6g8k5htny-coder/Math-/pull/253#pullrequestreview-5402591138), at the unchanged authored freeze `a932946d055192bc98f3f262c85e38df9a82b5ce`. |
| [255 / v1.6](https://github.com/d6g8k5htny-coder/Math-/pull/255) | `1cbbbf9dbe7c7319f4431e4a4daf6879a49d48fd` | 81 | Derivative increments, transverse Hessian bound and Hessian-to-concavity from H1. [Full review 5402707990](https://github.com/d6g8k5htny-coder/Math-/pull/255#pullrequestreview-5402707990). Integration waits for 253 and branch reconciliation. |
| [257 / v1.7](https://github.com/d6g8k5htny-coder/Math-/pull/257) | `3d1d7fda76eb44090e630d8bc90822020f591253` | 88 | Ridge existence, uniqueness and implicit-function regularity under the displayed open-domain premises. [K1 readback 5974748836](https://github.com/d6g8k5htny-coder/Math-/pull/257#issuecomment-5974748836). |
| [258 / v1.8](https://github.com/d6g8k5htny-coder/Math-/pull/258) | `0aed3cb1b202ceb24fd1608425a1ec567ac19c02` | 93 | Collar-domain Hessian bounds and composition of the ridge with the cap. [K2 readback 5974753168](https://github.com/d6g8k5htny-coder/Math-/pull/258#issuecomment-5974753168). |
| [259 / v1.9](https://github.com/d6g8k5htny-coder/Math-/pull/259) | `66f05b9709dad61e4a38c540916ee5fba93068d4` | 108 | L6, L9 and L12 quantitative route; derives `hconv` from C3 data and a lower `f_xxx` bound on D. [K3 readback 5974776534](https://github.com/d6g8k5htny-coder/Math-/pull/259#issuecomment-5974776534). |
| [260 / v2.0](https://github.com/d6g8k5htny-coder/Math-/pull/260) | `ea406075a2d256cbe4038ada6e9b0459a391c0f8` | 113 | L2's second statement and L4's `f_xxx ≥ 7/4` from H2. L14/L15 are not formalized and are not used by the cap theorem. [K4 readback 5974756909](https://github.com/d6g8k5htny-coder/Math-/pull/260#issuecomment-5974756909). |
| [262 / v2.1](https://github.com/d6g8k5htny-coder/Math-/pull/262) | `5d06e28e7b642ea5c8b69f68452905e661abeb60` | 121 | Six derivative-bound inputs and quantitative D-to-collar step derived from the block bound **and** `|f_xxx| ≤ m` on the pin segment. [K5 readback 5974829511](https://github.com/d6g8k5htny-coder/Math-/pull/262#issuecomment-5974829511). |

The dependency order is 253 → 255 → 257 → 258 → 259 → 260 → 262. The author
remains Anthropic/Claude; the auditor does not take over that authorship. The
v1.5/v1.6 actual reviewers are OpenAI. K1–K5 records identify Grok Bot support
agents; their model/provider remains **UNKNOWN** unless an authenticated record
establishes it. Shared GitHub identity supplies no organizational independence.

At v2.1 the Lean blob is `9ffef7878d0049b0059dadc387c8555e8f57d725`, manifest
blob `e64d1763e51b26311ca4960e58f4ecde10754d9f`, manifest SHA-256
`218273bd937233155af599c8e2b7ec6043cbfc97fcbf009ba49bd5ab8d885fb0`.
Its pinned informal inputs are SP blob
`a653dbc7bd8bd692433587f050ab8678713c297e` and CAP blob
`0633aca3c2a2882b0de4399da0a75d64c2e6b2e1`.

The complete declaration-by-source map is the immutable
[v2.1 ALIGNMENT table](https://github.com/d6g8k5htny-coder/Math-/blob/5d06e28e7b642ea5c8b69f68452905e661abeb60/frontiers/cap_first_exit_lean_20261002/ALIGNMENT.md).
It remains the packet's detailed coverage authority; this handoff does not
duplicate its 121 declaration records. The frozen manifests retain
`alignment_review: NOT_CLAIMED` as authored history; delivered native reviews
above are separate actual evidence and must not be erased or invented into the
old source bytes.

The v2.1 historical PR workflow records
[37163575474](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37163575474)
and [37163575728](https://github.com/d6g8k5htny-coder/Math-/actions/runs/37163575728)
reported success at this snapshot. Those run outcomes do not bind future
restacked tested trees, and this handoff is not a new execution receipt audit.

## Coverage still missing after the reviewed v2.1 scope

| Interface | What remains outside the formal proof | Concrete next audit/formalization package |
|---|---|---|
| I1: source objects | Identify SP's frame, periodic lift, cap and frontier with the objects consumed by `CapHyp.toTorus`. Transport itself is already supplied. | Pin SP/CAP object definitions and give an exact source-to-Lean map; prove the needed coordinate/lift instance without changing the topological theorem. |
| I2: actual persistence pairing | The relation between the older-component death level and the project's H0 superlevel persistence module/elder pairing. `elder_death_level_peak` proves the component-level quantity, not the module identification. | Define the intended filtration, born class and elder convention, including endpoint/tie hypotheses, then prove that identification. Preserve the distinction between a selected pair and a once-counted actual bar. |
| I3: source data | v2.1 still imports the qualitative C4 collar derivative data and a finite bound K, plus quantitative H1/H2, derivative blocks, critical pins and exact gap. It does not prove that the research good event supplies those data. | Prove that the exact C4-neighborhood/compactness assumptions supply the qualitative collar data and finite K. Separately connect the source good event to the quantitative inputs; do not silently replace imported premises with theorem conclusions. |
| I4: probability | Measurability, the actual weighted Gaussian/Palm law, Theorem A and `Q^W(G_r^c) ≤ C r^3`. | Begin with actual measurable/integrable objects and a measure-theoretic Cauchy–Schwarz interface; track the true lower-normalizer requirement and its domain. |
| Analytic coefficient | The definition/integration of the SIDE24 coefficient, special functions and a proof-producing enclosure. | Formalize one pinned analytic estimate or integral-to-enclosure interface. Integer endpoint ordering and passing Python certificates remain different evidence. |
| Other source claims | L14/L15, L23, shrinking/intermediate-scale collision, once-counted bar intensities and the broader lifetime law are not discharged by this cap stack. | Select exact load-bearing sources from the existing dependency graph before drafting Lean. Anything not inspected is **NOT ASSESSED**, rather than inferred covered from target counts. |

Math PR195's landed finite-radius normalizer floor is an informal/computational
dependency, not a new Lean theorem and not an all-small-r lower normalizer. Its
combined planar band requires `L ≥ 12`, `r ∈ [1/64, 1/2]`. A future probability
interface must preserve that range and cannot use GP214's upper bound as a lower
bound.

This selected map is not a claim that every public research source has been
inventoried. The new owner must expand coverage by exact dependency slices,
recording unformalized and unreviewed items instead of inventing declarations
or extrapolating acceptance from existing counts.

## Execution and review contract for each future packet

1. Pin the informal full text, exact statement and dependencies; name the
   writer/integrator and claimed paths. Read live ownership first.
2. Record the Lean declaration inventory, elaborated hypotheses and imported
   premises. Reuse existing manifest/gate formats; do not install a second
   scientific registry.
3. Bind the actual tested repository/commit/run/attempt and original receipt
   SHA-256. Inspect both the formal producer and required aggregate. A source-only
   gate is not Lean execution; missing or skipped results are not success.
4. Require pinned Lean `v4.34.1` and dependency locks unless a separately reviewed
   toolchain change is intended. The cap/Mathlib pin is
   `d13f23b723b8a846827a245b89c10fc7d3f11612`. Build, kernel recheck, complete
   transitive axiom reports and actual negative-control rejection are distinct
   outputs. Allowed foundations are `propext`, `Classical.choice`, `Quot.sound`.
5. Preserve the package-specific controls: Math scalar pilot has five, main
   arithmetic has ten, and cap has three (`sorry`, custom axiom, `native_decide`).
   Do not substitute a count from a different package or accept forbidden
   `sorryAx`/custom/native-computation axioms.
6. Perform exact-source alignment for changed statements/hypotheses. Reuse
   delivered reviews only for the same reviewed bytes, consumed dependencies
   and declared scope. Changed metadata needs a bounded readback; changed proof
   or hypotheses need the affected successor review.
7. Record actual performer/provider/exposure. Same-provider review earns zero
   organizational-independence credit; a display name does not authenticate
   another provider. Engineering integration does not create a scientific vote.
8. Publish one concise source-bound disposition in the existing PR or issue,
   read it back, and release the completed claim. Preserve scientific scope and
   open obligations after any merge.

The auditor's acceptance test is complete and honest accounting of its claimed
source slice. Complete transfer may be reported only for an explicitly bounded
slice whose in-scope statements have corresponding kernel-checked, aligned
proofs. Conditional conclusions must remain labeled with their imported
premises. Recording an absent declaration or an unproved interface does not
complete its formalization. The unresolved interfaces above prevent a claim
that all project mathematics has been transferred into Lean. This handoff
establishes the role and its starting queue; it does not assert that the research
program is fully formalized.
