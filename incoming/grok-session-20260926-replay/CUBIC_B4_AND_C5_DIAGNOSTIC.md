# Cubic B4 identity and C5 diagnostic (verified)

**Scientific effect:** NONE. Does not discharge D1 A3 type-convergence. Does not accept File-3 (ND).

## B4 (exact cubic, computer-algebra identity)

Every degree-≤3 polynomial with the six endpoint pins at `(±1/2,0)` (heights `0,-k`, both gradients zero) is the four-parameter family (B1) of Math- `reviews/collision_mechanism_20260925/NOTE.md`.

After imposing a third critical point `X=(u,v)`, `v≠0`, of intermediate height `-k θ` with `0<θ<1`, the three Hessians satisfy, identically,

    det B_M = (9k²/v²) [ 4θ - (w+1)² ]
    det B_S = (9k²/v²) [ 4(1-θ) - (w-1)² ]
    det B_X = -(9k²/v²) [ (w+1-2θ)² + 4θ(1-θ) ] < 0

where `w=(qv+12ku)/(6k)`. Difference between expanded Hessians and these formulae: `0`.

So every intermediate-height noncollinear cubic witness is a nondegenerate saddle. Endpoint types (M max, S saddle) hold on the bounded interval `-1-2√θ < w < 1-2√(1-θ)`.

This is an algebraic theorem on the contact model. It is not finite-`r` type-convergence of a smooth Gaussian field, so it does not close A3.

## C5 diagnostic (Bargmann–Fock, exact)

Conditioner `U0={f,f_t,f_tt,f_ttt,f_s,f_ts}` (the parent endpoint pin, same set as G+V).
Free jet `(a,q,c,d)=(f_ss,f_tts,f_tss,f_sss)`:

    E[ · | U0=(b,0,0,12k,0,0) ] = (-b, 0, 0, 0)
    Cov = diag(2, 2, 2, 6)

Matches the collision-note C5 sentence. That note already says the diagnostic is not a substitute for finite-`L` covariance.

## Still AMEND / OPEN

A3: `Z_r/r² → 36k² E[a² 1{a<0}|U0]` needs type-convergence + UI on the smooth field, not just the cubic model.
D5: microdisk rate still missing.
File-3 (ND): 15-stencil still unshipped.
