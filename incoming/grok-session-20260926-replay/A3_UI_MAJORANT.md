# A3 UI majorant — parent §5 read with landed erratum

**Scientific effect:** NONE.
**Does not accept Theorem A.**
**Does not close main #63.**

Parent SHA256: `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`
Erratum now on Math- default: `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` (merge `d8f5505`, blob `213594d6`).

## Displayed comparison the parent names but does not write

From (5.3) and the matrix-determinant lemma,

```
|det H_i / r| = |alpha_i det A_i - r beta_i^T adj(A_i) beta_i|
             <= |alpha_i| |det A_i| + r ||beta_i||^2 ||adj(A_i)||.
```

From (5.1), `|alpha_i| <= M_3/2` and `||beta_i|| <= M_3/2`.
From §4, `M_3 <= K_0 (J_r + ||A_r||_F)` with `J_r` independent of `A_r` and in every `L^p(Q)` uniformly in the compact marks.
Elementary: `|det A| <= C_m ||A||^m` and `||adj(A)|| <= C_m ||A||^{m-1}`.

Hence, for `r <= 1`,

```
|det H_i / r|  <=  P_m(J_r, ||A_i||_F)
```

where `P_m` is a polynomial of degree `m+1`.

### m = 1 (scalar transverse block)

```
|det H / r| = |alpha A - r beta^2|
           <= (M_3/2) |A| + r (M_3/2)^2
           <= (K_0/2)(J+|A|)|A| + (K_0/2)^2 (J+|A|)^2.
```

So `W_r / r^2 <= C (J + |A|)^4`.

### m >= 2

```
|det H / r| <= C (J+||A||) ||A||^m + C (J+||A||)^2 ||A||^{m-1},
W_r / r^2   <= Q_m(J, ||A_M||, ||A_S||)
```

with `deg Q_m = 2m+2`. Bound (5.1) also gives `||A_S - A_M|| <= r M_3`, so both blocks are controlled by `J + ||A_M||`.

## Why this is uniformly integrable

(3.5) is `density_Q(A) <= C_0 exp(-c_0 ||A||_F^2)`. Polynomials in `||A||` therefore have uniformly bounded `Q`-moments of every order. `J_r` has uniformly bounded moments of every order by §4. Independence `J_r ⊥ A_r` multiplies the moments. The family `{W_r/r^2 : 0 < r <= 1, marks compact}` is bounded in every `L^p`, hence uniformly integrable.

This is the comparison the parent cites as “UI from (5.3)”. It uses only (5.1), (5.3), (3.5), §4, and `||adj|| <= C ||A||^{m-1}`. It does **not** use the wrong congruence factor.

## What this does and does not give

Granted A2 moments and `A_i → A_0` in probability (Gaussian regression of means and covariances, plus `||A_S-A_M|| → 0` in `L^p`), continuous mapping plus UI give

```
Z_r / r^2  →  (6k)^2 E[det(A_0)^2 1{A_0 < 0}]
```

in `L^1`, under the *corrected* `D_r`. Compactness of marks then yields a finite-r floor `z_* > 0` after reducing `r_*`.

**Still not Theorem A.** A7 still has to divide a verified numerator by this floor, and `G_r ⇒` elder pairing remains a separate cap import.

## Verdict on the floor itself

- ACCEPT the *structure*: parent §4+§5 + landed erratum + the displayed majorant above is a complete Gaussian UI + continuous-mapping argument for (5.4)–(5.5).
- Do not treat this file as an independent matrix-density theorem for (3.5), and do not invent a numerical `z_*`.
- Organizational independence: none (same-account replay).
