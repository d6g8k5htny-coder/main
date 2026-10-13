# Round 7, first slice (Side24 successor interface): design-review packet, under HOLD

Status: **DRAFT UNDER HOLD.** Coordination correction main#307
[6136949323](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136949323) (02:35Z, 2026-10-13):
no repository, REGISTER, gate or docs mutation before a bounded nonauthor design/source-scope read and explicit
ownership reconciliation. Acknowledged in [6137008125](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6137008125).
Nothing in this directory is a packet change: Math- `8acc4856` is unchanged, the Math- branch
`claude/lean-statements-round7-side24b-20261013` is an unpushed local worktree with no edits, Math-#445 is untouched.
This packet is a recovery artifact (preservation only, which the hold allows) that gives the routed reviewer the exact
object to read.

Author: Anthropic Claude, session `session_01Cz7WZybv8znP64SpPj6sWY` (author of the statement packet and the proof
packet). Organizational-independence credit 0. Scientific effect NONE. Every declaration below would enter as
`specified`; no theorem, no label move, no inhabitant constructed, no acceptance implied.

## The reviewer's question, as posed

"Whether `condHessianCov`/`coneMomentOf` and the proposed interfaces encode the actual conditional Gaussian object with
the claimed source anchors, rather than merely adding assumptions to a formally accepted structure."

Short answer offered for review (the reviewer decides): the draft encodes the parent's (15.1) object *for a centred
Gaussian vector whose covariance is the derivative-rule matrix of `K_24`*, by two standard identities, each of which is
used as a definition and not proved here: (a) the conditional covariance of a jointly Gaussian vector given a
nondegenerate block equals the Schur complement, and (b) the conditional law is the centred Gaussian with that
covariance. Two of the new interface fields (`pinCov_posDef`, `condHessianCov_posDef`) are the parent's own
nondegeneracy clause; the third (`coneMoment_def`) is the (15.1) definition. What remains an *assumption added to the
structure* rather than the source's object is listed under "Caveats": the sphere measure is still free beyond support
and area, and frame invariance of the formal cone expectation is a consistency burden, not a field.

## Files in this directory

| file | what it is |
|---|---|
| `Side24_append.lean` | the draft block to be appended to `Specifications/Side24.lean` before its `end` line (12 declarations: 9 auxiliary, 1 structure, 2 `def … : Prop`) |
| `Side24_draft.lean` | the full draft module = Side24.lean at `8acc4856` minus its last line + the block + `end UniversalLaw.Spec.Side24`; the prefix is byte-identical to the committed module (asserted when assembled) |
| `elab_draft.log` | `lake env lean Side24_draft.lean` at the pin (Lean v4.34.1 `5045d005`, Mathlib `d13f23b7`), run from the built statement-packet checkout; see "Verification performed" |
| `register_rows_proposed.json` | the three statement rows that would be added under claim `side24-coefficient`, plus the proposed sentences for that row's `not_established` / `next_step`; NOT applied |
| `scan_draft.txt` | output of the statement gate's own `scan_module` on the draft (declaration inventory, kinds, docstring labels and verbatim anchors) |
| `ack_hold.md` | the posted acknowledgement |

## Declaration-by-declaration mapping to the pinned sources

Sources are the packet's registered pins: `side24-proof` = Math- `760340e9…` `coefficients/side24_v1/PROOF.md`
(sha256 `c06daccc…`), `lifetime-parent` = Math- `c3916425…` `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`
(sha256 `9350ad6e…`). Line numbers refer to the pinned bytes.

| declaration | kind | source clause transcribed | formal shape | source's object, or added assumption? |
|---|---|---|---|---|
| `TransRow d` | aux | side24-proof:46 "Writing m=d-1 and A for the transverse Hessian" | `{i : Fin d // 0 < i.val}` (the `u`-perp directions in the frame `R`, `u = R 0`) | bookkeeping |
| `TransIndex d` | aux | side24-proof:121-122 (svec, diagonal and sqrt(2) off-diagonal) | `{p : SymIndex d // 0 < p.1.1.val}`: svec indices with both entries transverse | bookkeeping |
| `transScale d a` | aux | side24-proof:122 | `1` diagonal, `sqrt 2` off-diagonal (restriction of `svecScale`) | the source's svec convention |
| `transCov d K R` | aux | side24-proof:39 (`Cov(H_ij,H_kl)` display, stated for `K_infty`) and the stationary rule already used by `jointCov` | `Matrix.of fun a b => scale a * scale b * D^4 K(0)[R a1, R a2, R b1, R b2]` | the Hessian block of the existing `jointCov`, restricted; not shown to be a covariance (same status as `jointCov`) |
| `transPinCov d K u R` | aux | side24-proof:42 (`V = Hu`) | `Matrix.of fun a i => scale a * D^4 K(0)[R a1, R a2, u, R i]` | cross block of `jointCov` between Hessian and pin coordinates |
| `condHessianCov d K u R` | aux | side24-proof:139-143 ("Schur-complement quadratic form … the entire conditional transverse covariance A\|V=0") | `transCov - transPinCov * (pinCov)⁻¹ * transPinCovᵀ` | identity (a): Gaussian conditional covariance = Schur complement **when `pinCov` is invertible**; Mathlib's `⁻¹` is `0` on a singular matrix, so without nondegeneracy the expression degenerates to `transCov` (hence the field `pinCov_posDef`) |
| `symOf d v` | aux | side24-proof:147 ("n=m(m+1)/2 independent symmetric-matrix coordinates") | the symmetric `TransRow × TransRow` matrix with `v_(i,j)` on the diagonal and `v_(min,max)/sqrt 2` off it | inverse of the svec scaling; bookkeeping |
| `coneIntegrandTrans d v` | aux | side24-proof:155 ("h(A)=det(A)^2 1{A<0}") | `Set.indicator {v \| (-(symOf d v)).PosDef} (fun v => (symOf d v).det ^ 2) v` | the source's `h`, with `A < 0` read as `-A` positive definite (as `NegDefCone` already does) |
| `coneMomentOf d K u R` | aux | lifetime-parent:454 "(15.1) D_u=E[(det A_u)^2 1{A_u<0} \| V_u=0]" | `∫ v, coneIntegrandTrans d v ∂(multivariateGaussian 0 (condHessianCov d K u R))` | identity (b): the conditional law of a centred Gaussian vector given `V = 0` is the centred Gaussian with the conditional covariance. Mathlib's `multivariateGaussian μ S` is the Dirac mass at `μ` when `S` is not positive semidefinite (`multivariateGaussian_of_not_posSemidef` at the pin), hence the field `condHessianCov_posDef` |
| `PeriodicCoefficientInterface2` | structure | lifetime-parent:456 "Here A_u conditional on V_u=0 is an actual centered nondegenerate Gaussian symmetric matrix" | `extends PeriodicCoefficientInterface` + `pinCov_posDef`, `condHessianCov_posDef`, `coneMoment_def` (all quantified over `d ∈ {2,3}`, orthonormal frames `R`, `u = R 0`, as `pV0_def` and `tau_sq` already are) | the parent's clause as two positivity hypotheses and its (15.1) as a definition field; the predecessor is unchanged and stays registered |
| `CoefficientEnclosure2 I` | def_prop | side24-proof:25-26 (headline) | `CoefficientEnclosure I.toPeriodicCoefficientInterface` | same display, over the successor |
| `ReferenceComparison2 I` | def_prop | side24-proof:173 (display (4), quantitative, with its tag; anchor distinct from the predecessor's by the tag) | `ReferenceComparison I.toPeriodicCoefficientInterface` | same display, over the successor |

The 26 existing Side24 Props (24 theorem-shaped, 2 vocabulary), the two existing structures and all 38 existing auxiliaries are byte-identical; the
successor Props reuse the predecessors' bodies through the structure projection, so no display is re-typed.

Why the successor removes the known blocker: the round-4 probe showed `(∀ I, CoefficientEnclosure I) → IsEmpty
PeriodicCoefficientInterface` by doubling `coneMoment` and `c24` (both free up to `eq_15_2`). Under
`PeriodicCoefficientInterface2`, `coneMoment d (R 0)` is fixed by `coneMoment_def` for every orthonormal frame, so
that inhabitant no longer exists. This is a statement about the formal interface only; it says nothing about the
truth of either display.

## Caveats found while preparing (the honest part; each is in the draft docstrings verbatim)

1. **`sphereMeasure` is still a given measure** constrained only by `sphereMeasure_support` (carried by the unit
   sphere) and `sphereMeasure_area` (total `2π`, `4π`). So the successor interface determines `c24` from its fields
   only *up to that choice*. My proposal [6136729408](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6136729408)
   said "determined"; that was overstated and is corrected here. For the source's argument the choice is immaterial
   for the two successor *inequalities* because its integrand bounds are uniform in `u` (Section 4, "Angular
   integration preserves these multiplicative bounds because every reference integrand is positive"); it is not
   immaterial to the exact value of `c24`, and the uniformity is a theorem about the formal objects, not a given. Design option for the reviewer: a further field pinning `sphereMeasure d` to Mathlib's surface measure.
   At the pin, `MeasureTheory.Measure.toSphere` (`Mathlib/MeasureTheory/Constructions/HaarToSphere.lean:51`) gives a
   measure on the subtype `sphere (0:E) 1` with `toSphere_apply_univ : μ.toSphere univ = dim E * μ (ball 0 1)`, so
   `(volume.toSphere).map Subtype.val` on `E d` has total mass `d · vol(ball)` = `2π` (d=2), `4π` (d=3), consistent with
   `sphereMeasure_area`. Not included in the draft: it widens the slice beyond what was proposed, and it is exactly the
   kind of choice the design read should make.
2. **Frame invariance is a consistency burden, not a field.** `coneMoment_def` is required in every orthonormal frame
   with `u = R 0`; an inhabitant exists only if the formal `coneMomentOf d (K24 d) u R` does not depend on the
   transverse part of `R`. The parent asserts this for the actual object ("invariant under transverse-frame changes",
   lifetime-parent:456); for the formal objects it is a theorem nobody has stated. The existing fields `pV0_def` and
   `tau_sq` already have the same shape, so this is not new to the successor, but it should be seen.
3. **Nondegeneracy** is carried as `pinCov_posDef` and `condHessianCov_posDef` (the parent's "nondegenerate"). Without
   the first, the Schur complement is not the conditional covariance (Mathlib's inverse of a singular matrix is `0`);
   without the second, a non-positive-semidefinite formal covariance would make `multivariateGaussian` the Dirac
   mass at `0` and the "cone moment" `0` (the indicator at `v = 0` is `0` since `-0` is not positive definite), while a
   singular positive-semidefinite covariance gives a degenerate Gaussian, not a Dirac mass; the absence of a PosDef
   field would not by itself make the covariance singular, and the inherited `coneMoment_pos` together with
   `coneMoment_def` already excludes the Dirac (zero-moment) case. Both are hypotheses on the formal `K24` matrices; nothing here
   proves them (the source derives them from (3) and the reference values; `ReferenceCovarianceLowerBound` and
   `CovarianceOrdering` in the proof packet are the ingredients a proof would use).
4. **Integrability** of the cone integrand is not stated; Mathlib's Bochner integral is `0` for a non-integrable
   function. Under `condHessianCov_posDef` the integrand is a polynomial times an indicator against a nondegenerate
   Gaussian, so it is integrable, but that is again a theorem, not a field.
5. **svec convention.** `transCov` carries the `sqrt 2` scaling on off-diagonal coordinates (as `jointCov` does), and
   `symOf` divides it back out, so `symOf` applied to a sample of the scaled law is the unscaled symmetric matrix whose
   determinant the source means. The reviewer should check this inverse pairing; a sign or scaling slip here would be a
   mis-transcription that no gate detects.
6. **Anchors.** Each new registered-kind declaration quotes a verbatim line (≥ 16 chars) of its named pin; the two
   successor Props use the quantitative headline displays (the `d = 3` enclosure line; display (4) with its tag `(4)`),
   distinct as strings from their predecessors' anchors (the docs renderer matches proof-packet theorems to Props by
   anchor equality, so shared anchors would be ambiguous later). The structure's anchor is the
   parent's Section 15 sentence, i.e. the successor is anchored to the parent, not to the SIDE24 note.

## Verification performed (local, author, not a receipt)

- Statement gate `check.py` `scan_module` on the full draft module, with the packet's pinned sources loaded: 78
  declarations = the committed 66 (26 `def_prop`, 2 structures, 38 auxiliary; identical names, kinds and result types
  in the same order) + 12 new (9 auxiliary, 1 structure, 2 `def_prop`); every new registered-kind declaration carries
  the four labels, names a registered pin and quotes a verbatim anchor (`scan_draft.txt`). No refused token, no
  string literal, no `#`, no attribute, no off-column command in the block.
- The proposed register rows: `does_not_claim` equals the docstring's `Does not claim:` line verbatim; each anchor
  is verbatim in its source and distinct from every existing Side24 register anchor (asserted when the JSON was written).
- Elaboration at the pin: `lake env lean Side24_draft.lean` from the built statement-packet checkout (Lean 4.34.1 `5045d005`, Mathlib `d13f23b7`, `LEAN_NUM_THREADS=2`): exit 0, no output (no error, no warning, no `sorry`), real 1m07.5s (`elab_draft.log`). The hold forbids a `lake build` of the packet, so this is a single-file elaboration of the draft module, not a build receipt.
- Not run, because the hold forbids packet mutation: `check.py` source mode on a worktree with the appended module
  (needs the register rows applied), `test_check.py`, `--execute`, the hosted workflows.

## What this packet does not establish

Nothing mathematical. It does not show that either successor Prop is true or provable; it does not construct an
inhabitant of either interface; it does not show the formal matrices are covariances of any field; it does not show
frame invariance, nondegeneracy or integrability for `K24`; it does not change, promote or reclassify any claim,
premise, obligation or label; it is not a review and it is not evidence of acceptance. Alignment of every statement
remains PENDING_INDEPENDENT_REVIEW; same-account author, credit 0.

## Corrections applied (coordinator design read, main#307 6137276515, 03:04Z)

1. Caveat 3 reworded: only a non-positive-semidefinite covariance triggers Mathlib's Dirac fallback; a singular
   positive-semidefinite one gives a degenerate Gaussian; the absence of a PosDef field does not establish singularity;
   `coneMoment_def` with the inherited `coneMoment_pos` already excludes the zero-moment Dirac case. The field
   docstring of `condHessianCov_posDef` says the same.
2. `ReferenceComparison2` is anchored to the quantitative display (4) line including its tag `(4)` (verbatim,
   side24-proof:173), in the docstring and in the proposed register row; the "NOT claimed equal" sentence is no longer
   its anchor.
3. The angular-measure remark is restricted: uniform directional bounds would make the choice of `sphereMeasure`
   immaterial to the two successor inequalities, not to the exact value of `c24` (module header, `ReferenceComparison2`
   docstring, caveat 1).
4. The count of the pre-existing Side24 inventory reads 26 Props (24 theorem-shaped, 2 vocabulary), 2 structures,
   38 auxiliaries, as the register and the scan have it.
5. For the stacked docs successor: the frontier row of `Side24.CoefficientEnclosure` in Math-#445 says "the conditional
   law of the Hessian block given the gradient under `K24`"; the round-7 docs will say "the transverse Hessian block
   conditioned on the pin `V_u = H_f u = 0`". Math-#445's own branch is frozen and is not edited.

Placement to be confirmed by the author on main#307: round 7 as an explicitly stacked draft on Math-#445 (head
`9fb0a4a0`, base branch `claude/lean-statements-frontier-20261013`), #445 integrated first, no simultaneous edit to
#445's frozen branch, exact four-file scope (`Specifications/Side24.lean` append only, `REGISTER.json` rows,
`README.md`, `SCOPE.md` re-rendered), no gate, import, proof or toolchain change. Activation is the coordinator's.
