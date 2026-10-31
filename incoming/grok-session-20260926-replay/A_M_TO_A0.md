# A_M → A_0 in the Q-law — scoped A3 slice (not the floor)

**Scientific effect:** NONE.
**Does not accept Theorem A or the z_* floor.** Does not close main #63.
**Sources:** parent §3–§5, SHA256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`, plus landed Math- main `ERRATUM_CONGRUENCE.md` (merge `d8f5505`).

## What the parent asserts

Let `A_M = D_y² f(M)` with `M = -(r/2)u`, and let `A_0` be `D_y² f(0)` under regression on the contact jet `U0* = v_0`. Parent §5: finite-dimensional Gaussian regression gives uniform convergence of means and covariances of `A_M` to those of `A_0`. Together with (5.1)–(5.2) and the corrected congruence this is used for

    D_r H_M D_r → diag(-6k, A_0),
    D_r H_S D_r → diag( 6k, A_0)

in probability, uniformly on compact `B × K × O(d)`.

Also (5.1): `||A_S − A_M||_op ≤ r M3`, so the two endpoint transverse Hessians share the same limit `A_0`.

## What Gaussian regression actually gives

`A_M` and the contact coordinates `U_r` are jointly Gaussian on a finite-dimensional space (`Sym_m` plus the `2(d+1)` pins, rewritten as `U_r`). Conditional laws are therefore Gaussian, and convergence in law is equivalent to convergence of means and covariances.

Write

    E[A | U = v] = μ_A + Σ_{AU} Σ_U^{-1} (v − μ_U),
    Cov(A | U)   = Σ_{AA} − Σ_{AU} Σ_U^{-1} Σ_{UA}.

Parent §3 already claims: `Σ_r(R)` and `Σ_r(R)^{-1}` converge uniformly on `O(d)` for small `r`; the enlarged covariance that includes the independent entries of `A_r` is uniformly positive definite at contact; the Schur complement of `A` given `U` is uniformly bounded above and away from zero.

Two extra identifications are required, and they are standard once A1–A2 are granted:

1. **Site shift.** `A_M` is the Hessian at `M`, `A_0` is the Hessian at `0`. Mean-square continuity of second derivatives follows from the rapidly summable Fourier expansion in §2 and the integral-remainder argument already used for `U_r → U0*` in §3. No new covariance identity is needed beyond C² continuity of the kernel.

2. **Conditioning σ-algebra.** `A_0` is defined under regression on `U0* = v_0`, not under regression on `U_r = v_r`. The two conditionings agree in the limit because `U_r → U0*` in L² uniformly in frames (§3) and `v_r → v_0` uniformly on compact `B × K`. That is the content of the contact-frame construction, not a separate lemma.

Uniformity in the frame `R` is compactness of `O(d)` plus continuity of covariance entries under rotating derivative multi-indices. It is **not** torus rotational invariance. Lattice anisotropy is already ruled out at the jet-rank step: a polynomial `P(R^T n)` vanishing on `ℤ^d` is identically zero.

The endpoint comparison `||A_S − A_M||_op ≤ r M3` is an average / Lipschitz bound from the two gradient pins, the same style as (5.1) for `α` and `β`. On compact marks, `M3` has uniformly bounded moments by §4, so `A_S − A_M → 0` in probability uniformly. Both endpoints share `A_0`.

## Verdict this slice

`A_M → A_0` in the Q-law is **accepted as a Gaussian-regression lemma conditional on A1–A2** (jet rank + uniform `Σ_r^{-1}` + `U_r → U0*` + residual moments that bound `M3`).

It is **not** acceptance of:

- the type indicator → `1{A_0 < 0}` (needs the landed erratum `D_r` plus `√r β → 0`);
- uniform integrability of `W_r/r²`;
- the floor `z_* > 0`;
- Theorem A / `1-p ≤ C r^3`.

## What a counterexample would have to be

| Candidate | Why it does not currently work |
|---|---|
| Some frame `R` with singular `Σ_0(R)` | Contradicts A1 jet rank: `P(R^T n)=0` on `ℤ^d` ⇒ `P ≡ 0`. |
| `U3` cancellation destroying `v_r → v_0` | Parent writes `U3` as an averaged third derivative, target `12k`. |
| `A_M` and `A_0` living at different sites | Controlled by C² continuity of the kernel; `M → 0`. |
| Anisotropy of the torus | Never used. Uniformity is compactness, not isotropy. |

No counterexample to `A_M → A_0` was constructed this session. Absence of a counterexample is not the z_* floor.

## Downstream

A3 remaining hole is **UI of `W_r/r²` plus type-indicator convergence under the landed `D_r`**. This slice removes `A_M → A_0` as an independent blocker, provided A1–A2 stay granted.
