# Challenge note — RN mesoscopic annulus reduction (Math- PR #7)

**Date:** 2026-09-25  
**Disposition:** same-author technical challenge of an author-side reduction.  
**Not** an R17 review record. **Not** a gate movement. Organizational independence
credit is **0**. `lemma_closed` stays false. Scientific effect: NONE.

## Exact object

| Field | Value |
|---|---|
| Object | `frontiers/rn_mesoscopic_20260925/PROOF.md` |
| Claim name in object | `RN-MESOSCOPIC-ANNULUS-REDUCTION-20260925-v1` |
| Math- surface | [PR #7](https://github.com/d6g8k5htny-coder/Math-/pull/7) (draft) |
| Bytes / sha256 | **7316** / `b7ef84cd1e5946c766e125e1ccff6bf611214254d35dc2e0778f019b8d48dcb9` |
| Companion recon | `RECONNAISSANCE.md` **1708** / `13b60b25326dc2bd857e54339e9ca2c75b34185d619ea24f32a60ae39fd90ec9` |
| Gate node | `math.rn-mesoscopic-reduction` (`AUTHOR_SIDE_REDUCTION`, layer D5) |

Hash taken from the PR head branch `chatgpt/rn-mesoscopic-annulus-20260925` in this
session. A changed legitimate source needs a new declared identity.

## What the object itself claims to establish

1. Fixed-remote control with `dist ≥ ρ` does not cover `x = r y` with fixed
   `1 < A ≤ |y| ≤ B < ∞`.
2. Raw remote gradients are the wrong witness variables as `r → 0` because residual
   covariance collapses under the pin constraints.
3. After a centered divided-difference / Hermite residual map `J_r(y)`, Lemma A
   asserts uniform positive-definiteness of the contact covariance on the compact
   annulus away from scaled pin sites `±e₁/2`, plus uniform finite-`r` covariance /
   Schur convergence.
4. After `x = r R y`, the expected count on the scaled annulus is an explicit
   `r^d` spatial factor times a height window of length `k r³` times an integrand
   `γ_AB`; an `O(r³)` annulus budget needs `γ_AB ≤ C r^{-d}` (or better).
5. The remaining load-bearing work is finite-dimensional: enumerate independent
   `J_0(y)` rows, compute `det S_r(y)` powers, combine with the conditioned witness
   Hessian determinant, and prove chart-boundary integrability.

The object explicitly does **not** claim the exact `J_0` ledger, pin-neighborhood
limits, the `r ≪ |x| ≪ ρ` transition, witness-collision charts, or any 24-jet /
`lemma_closed` discharge.

## Reconstruction (condensed)

The reduction direction is coherent with the fixed-remote refusal to shrink `ρ` by
compactness alone: the parent/fixed-remote contact transform `U_r` already absorbs
the leading jet; the residual remote gradient must be re-expressed by divided
differences that remain finite at contact. Positive Fourier spectrum ⇒ vanishing
variance of a linear differential functional forces the symbol to vanish, which is
the right mechanism for Lemma A's contact positive-definiteness **once** a fixed
rank chart of independent rows is chosen. The spatial power identity
`dx = r^d dy` and height length `k r³` are elementary and correctly isolate
`γ_AB` as the remaining analytic object.

## Challenges (do not treat as PASS_TECHNICAL)

1. **Row list is an open hypothesis, not a lemma hypothesis.** Lemma A quantifies
   over “the independent rows of `(U_0, J_0(y))`” after removing algebraic
   dependence, but the object never exhibits those rows. A compactness argument
   over an unspecified basis cannot be replayed. Next exact action remains §5
   item 1: enumerate gradient-plus-height contact rows after the six/eight pins.
2. **Rank-chart cover is deferred.** The “Important qualification” after Lemma A
   admits that a single chart need not reach algebraic rank-change loci and that
   overlap Jacobians must be controlled. Until that finite cover exists, Lemma A
   is a mechanism sketch, not an annulus theorem.
3. **Pin excision must be quantitative.** Excluding `±e₁/2` from `K_AB` is
   necessary, but the note gives no modulus for how close `|y ± e₁/2|` may
   approach before the least eigenvalue collapses. That modulus is exactly the
   pin-collision complement recorded as `math.rn-region.pin-collision`
   (`OPEN_ACTIVE`) in the Math- hard gate / [D0 crosswalk](DOWNSTREAM_RN_CROSSWALK_20260925.md).
4. **`γ_AB ≤ C r^{-d}` is not evidenced.** Section 4 correctly states the target
   inequality and that naive fixed-remote covariance cannot prove it. No lower or
   upper contact determinant power is computed. An `O(r³)` claim for this annulus
   would be an illegal promotion under [#90](https://github.com/d6g8k5htny-coder/main/issues/90)
   while `math.rn-mesoscopic-reduction` stays non-terminal.
5. **Same-author / navigation non-discharge.** Green CI on Math- #7, this note,
   or the hard-gate wiring cannot promote the reduction to CONTROLLING.

## Negative controls attempted here

| Control | Result |
|---|---|
| Re-hash PR7 `PROOF.md` / `RECONNAISSANCE.md` | Digests above; mismatch would refuse citation |
| Check object’s own “still open” list against gate D5 regions | Aligns with mesoscopic / pin-collision / intermediate / witness-collision `OPEN_ACTIVE` nodes |
| Search object text for `lemma_closed` flip or 24-jet discharge | Absent; object refuses legacy Boolean promotion |

No symbolic `J_0` enumeration was executed in this session. That omission is why
this note is a challenge, not a technical pass.

## Relation to downstream-first work

| Surface | Role |
|---|---|
| [Math- #7](https://github.com/d6g8k5htny-coder/Math-/pull/7) | Object under challenge |
| [Math- #8](https://github.com/d6g8k5htny-coder/Math-/pull/8) | Fail-closed integrity gate; maps this object as `AUTHOR_SIDE_REDUCTION` |
| [main #92](https://github.com/d6g8k5htny-coder/main/pull/92) / [crosswalk](DOWNSTREAM_RN_CROSSWALK_20260925.md) | Human D0/D4/D5 classification companion |
| [#86](https://github.com/d6g8k5htny-coder/main/issues/86) D5 | Owns shrinking RN geometry; this note does not close D5 |

## Verdict (informal; not an R17 `technical_verdict`)

**AMEND-shaped challenge:** accept the reduction *framing* (wrong variables named,
compactness mechanism identified, remaining ledger isolated). Refuse any reading
that treats Lemma A or the `γ_AB` bound as established. Required amendment is the
explicit `J_0` / `det S_r` ledger on finitely many rank charts with a quantitative
pin-separation modulus.

Gate status after this note: UNCHANGED. Independence credit: 0.
