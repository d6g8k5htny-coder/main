# D5 pin microdisk — the missing inequality

**Object:** GROK-HEAVY-D5-MISSING-20260926-v1
**Scientific effect:** NONE. Does not accept a D5 pin lemma. Does not flip STATUS.
**Sources:** Math- `PROOF_INDEX.md` D5 paragraph; [Math #58](https://github.com/d6g8k5htny-coder/Math-/issues/58); PR82 cubic identity (previous session).

## What is already on the table

Reviewed annulus (fixed scaled radius) and thin-tube candidates exist. M1–M7 and S1–S4 were accepted on the Hermite-repair replacement. PR82 is the exact polynomial identity

    det H_M ≡ 0

on the constrained cubic stratum after Φ=0. That identity is not a Q-measure rate.

## The missing display

Let the six endpoint pins be the usual fold pair at distance r. Write

    N_μ(r) := #{ critical X ≠ M : |X − M| ≤ C_* r^{2} }

for a fixed C_* (the nested microdisk of Math #58: X = M + r s, then s = r S). The claimed summed pin-neighborhood bound remains AMEND until some β and C exist with

    sup_{b,k,R} E_Q[ N_μ(r) W_r ] / Z_r  ≤ C r^β.          (D5-μ)

Math #58 targets all-height O(r^{3}), i.e. β = 3, and explicitly says not to force that rate if the derivation disagrees.

An equivalent Kac–Rice form on the microdisk chart S ∈ D(S_0) is

    r^{4} ∫_{D(S_0)} E[ |det H_X| · W_r  |  pair pins, ∇f(M+r^{2}S)=0 ] φ_{∇}(0) dS
        ≤ C r^β Z_r.

## Why PR82 does not supply (D5-μ)

PR82 vanishes the cubic weight on a lower-dimensional algebraic set. Kac–Rice needs an integrable majorant in a neighborhood of that set. No such neighborhood rate is in the reviewed D5 files.

## Relation to File 3 Lemma I

(D5-μ) is the inner-count architecture with the third point colliding with the *maximum* pin rather than the saddle. The BF cubic 2×2 of (W_t, W_s) | G+V is the same leading block off the fold axis. Importing that identity as a D5 proof is not permitted: different pin chart, no reviewed stitch to the accepted annulus, File-3 (ND) still OPEN on its own 15-stencil.

## Collar

Separately missing (PROOF_INDEX): the open collar between the microdisk |x−M| = O(r^{2}) and the reviewed fixed annulus. That is a second inequality, not a relabel of (D5-μ).
