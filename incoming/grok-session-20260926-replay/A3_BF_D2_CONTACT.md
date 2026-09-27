# A3 contact moment, Bargmann–Fock, d=2

**Object:** GROK-HEAVY-A3-BF-D2-20260926-v1
**Scientific effect:** NONE. Does not prove parent (5.4)–(5.5). Identifies the candidate limit for one kernel in dimension two.

## Parent target

    Z_r / r² → z_0(b,k,R) := (6k)² E[ det(A_0)² 1{A_0 < 0} ]     (5.4)
    0 < z_* ≤ Z_r/r² ≤ z^* < ∞                                 (5.5)

For d=2 the transverse Hessian is the scalar A_0 = f_{ss}. det A_0 = f_{ss}.

## Infinite-plane BF, one-point even block

C(u)=exp(-|u|²/2).

    E[f_{ss} | f, f_{tt}] = -f,
    Var(f_{ss} | f, f_{tt}) = 2.

Odd coordinates {f_t, f_s, f_{ts}, f_{ttt}} drop out. Adding f_{tts} does not change the residual variance.

Contact pins: f → b, f_{tt} → 0. Therefore the candidate law is

    A_0 ~ N(-b, 2).

Let φ, Φ be the standard normal density and cdf. Then

    E[A_0² 1{A_0<0}] = (b² + 2) Φ(b/√2) + √2 b φ(b/√2).

Check: b=0 gives 1. So

    z_0(b,k) = 36 k² [ (b² + 2) Φ(b/√2) + √2 b φ(b/√2) ].

On any compact birth window B bounded below, inf_B z_0(b,k) > 0. That is a candidate z_*, not a reviewed floor for Z_r.

## What is still missing (the actual A3 obstruction)

(5.4) is type-convergence of the weighted intensity Z_r, not evaluation of the contact integrand. Needed: uniform integrability of det(A_r)² 1{A_r<0} under Q, from the Gramian gap in (3.5) plus continuity of the pins. That estimate is not supplied here.

## Torus caveat

File 1 is on T_L². Periodized BF at u=0 has image correction 4 e^{-L²/2}+⋯. At L=2 this is O(10^{-1}); at L=2π it is O(10^{-9}). The identities above are the L=∞ kernel. They do not automatically transfer to small L.

## Axis 4-jet (replay)

Cov(f_{tttt}, f_{ttts} | G+V) = diag(24, 6).
After the separate blow-up /r^8, /r^6 on {z_s=0}:

    det Gram(W_t^{ax}, W_s^{ax}) = z_t^{12}/9 > 0 for z_t ≠ 0.

Independent match of Lucas. Not Appendix B.
