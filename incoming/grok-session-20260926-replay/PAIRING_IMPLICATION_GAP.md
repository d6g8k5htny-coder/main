# Cap pairing implication chain — what is used, what is not

**Object:** GROK-HEAVY-PAIRING-GAP-20260926-v1
**Scientific effect:** NONE. Does not accept Theorem A.
**Sources:** cap `0633aca3c2a2882b0de4399da0a75d64c2e6b2e1`; parent SHA256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`; File 2 Morse-Smale lit pass.

## Pathwise implication the cap file actually writes

On an already-embedded cylinder D, under the estimates

    λ_min(-D_y^{2} f(M)) > (4/(3κ)) r M_3^{2},
    r M_4 ≤ 3κ/10,

there is a closed cap C through S separating M from every older point, and a ridge path through S with min = s and an endpoint of height > b. Therefore the maximin connection level of M is exactly s.

Conversion sentence (verbatim cap):

> On a compact manifold, for a global Morse extension with distinct critical values, the ordinary superlevel elder death partner of M is S.
> Exactly one ascending unstable branch of S ends at M, for a smooth Riemannian gradient.
> **No Morse-Smale assumption and no identification of the ridge with a flow trajectory are used.**

One-branch reason, written out: at S the transverse Hessian is negative definite, so the unique positive Hessian eigenvalue (ascent-unstable) cannot lie in the transverse hyperplane. Its eigenvector has nonzero axial component. The half-branch that enters C cannot exit (∂C ≤ s, height increasing) and its ω-limit in the compact cap is M.

## What parent §8 supplies for that conversion

At each fixed r, Q is a.s. Morse with distinct critical values (overdetermined jets + pin-Hessian density + k,r > 0). The maximin event {d_f(M)=f(S)} is Borel. Transfer to Q^W by absolute continuity and 0 < Z_r < ∞.

That is exactly the Morse + distinct-values input the cap conversion names. It is **not** Morse-Smale.

## What File 2 says is missing, and why it is the wrong gap

File 2: no published a.s. no-saddle-connection theorem for Gaussian fields. Stable/unstable manifold transversality is folklore. Symmetric spectra can force saddle connections.

Cap pairing does not quantify saddle-saddle connections. Elder death of a maximum at an index-(d-1) saddle is a maximin-of-paths statement. A saddle-saddle orbit would be a different cell in the Morse complex. Do not hold D1 for R0.

## Displayed remaining gap (the real one)

    (embedded D) ∧ G_r ∧ Morse ∧ distinct values
        ⇒  ordinary elder partner of M is S.          (pathwise)

    Q^W(G_r^c) ≤ C r^{3}                               (probability)

The first line is the cap + §8 chain. The second line is parent §7 / A7. A7 is still AMEND because it consumes A2/(3.5) and the A3 floor as reviewed inputs. Embedding of D on T_L^d is absorbed into existential r_* (reader comparison: r < L/(4√2) suffices).

File-1 Theorem A is a different object: q(r,b)→1 on T_L^{2} for gradient-adjacent fold pairs, PROVEN-MODULO Lemma I / ND / R0. Do not identify the two Theorems A.
