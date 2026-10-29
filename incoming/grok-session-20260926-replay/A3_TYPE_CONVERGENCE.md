# A3 type-convergence and z_* floor — scoped note (not acceptance)

**Scientific effect:** NONE.
**Does not accept Theorem A.** Does not close main #63.
**Object:** parent Section 5 type-convergence + (5.4)–(5.5) floor, read with Math- PR64 erratum.
**Parent SHA256:** `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`
**Erratum object:** UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1-ERRATUM-1 (Math- PR64; copied beside this file).

## What A3 claims

On compact marks `k >= k_- > 0` and compact frames,

    Z_r / r^2  →  z_0(b,k,R) = (6k)^2 E[ det(A_0)^2 1{A_0 < 0} ] > 0

uniformly, hence a floor `z_* <= Z_r/r^2` for small r. The mechanism is:

1. physical Hessian `H_i = [[r α_i, r β_i^T],[r β_i, A_i]]`;
2. `α_M → -6k`, `α_S → 6k` from (5.2);
3. `A_i → A_0` in law from Gaussian regression;
4. congruence-scaled matrices converge to `diag(±6k, A_0)`;
5. type indicator → `1{A_0 < 0}`;
6. uniform integrability of `W_r/r^2` from (5.3)+(5.1)+(3.5).

## Correct vs displayed D

Let

    D_r = diag(r^{-1/2}, I)     (erratum)
    D_old = diag(r^{1/2}, I)    (displayed sentence)

Then, identically in 2×2 and 3×3,

    D_r H D_r = [[α, √r β^T],[√r β, A]],
    det(D_r H D_r) = det(H)/r = α det A − r β^T adj(A) β.

That last identity is parent (5.3). Because `D_r` is positive definite, `H` and `D_r H D_r` have the same inertia, so type is read on the scaled matrix.

`D_old H D_old = [[α r^2, β r^{3/2}],[β r^{3/2}, A]]`.
As `r → 0` this limit is `diag(0, A)`, which is **singular**. Type-convergence **fails** if the displayed factor is used as written. That is the actual counterexample to the uncorrected sentence, not a counterexample to (5.3)–(5.5) after the erratum.

## Mechanism under the corrected D

Bound (5.1) says `||β_i|| <= M3/2`. On compact marks, `M3` has uniformly bounded moments by §4. Therefore

    √r β_i = O_p(√r) → 0.

Together with `α_M + 6k = O_p(r)` from (5.2) this is the type-convergence mechanism:

    D_r H_M D_r → diag(-6k, A_0),
    D_r H_S D_r → diag( 6k, A_0)

in probability, uniformly on the compact parameter set **if** A2 residual moments and A_i → A_0 are granted.

## What a counterexample to the floor would have to be

The floor `z_* > 0` can fail in four structurally different ways. Only (C1) is realized by the displayed sentence.

| ID | Failure mode | Status this session |
|---|---|---|
| C1 | Wrong congruence: scaled limit is singular, type indicator does not go to `1{A_0<0}` | Realized by `D_old`. Repaired by erratum. |
| C2 | `k_- = 0` allowed, so `(6k)^2 → 0` | Not the compact-mark theorem. Theorem A assumes `k >= k_- > 0`. |
| C3 | Law of `A_0` does not charge `{A<0}` | Not available for a full-support nondegenerate Gaussian on `Sym_m`, m≥1. Parent cites (3.5) plus open negative-definite ball. Not re-proved here as a matrix-density theorem. |
| C4 | `W_r/r^2` not uniformly integrable, so `E W/r^2` does not pass to the limit | Parent claims UI from (5.3)+(5.1)+(3.5). Not independently re-derived this session. |

No C2/C3/C4 counterexample was constructed. Absence of a counterexample is not acceptance of the floor.

## What is still required for A3 ACCEPT

1. Default-branch readers of Math- must see the erratum (PR64 is still draft; file 404s on Math- main). Copying it into this incoming packet is custody, not merge.
2. Reviewer must accept A2 residual moments that bound `β` and `M3`.
3. Reviewer must accept `A_M → A_0` in the Q-law, uniformly in frames.
4. Reviewer must accept uniform integrability of `W_r/r^2`.
5. Reviewer must accept that the type discontinuity set `{det A_0 = 0}` has Gaussian probability zero, uniformly in the compact parameter set.

Until those five are source-bound, A3 stays **AMEND / ACCEPT_ERRATUM only**. A7 (`1-p <= C r^3`) remains blocked because it divides by this floor.

## Non-claims

- Not Theorem A.
- Not a numerical `z_*` or `r_*`.
- Not unrestricted marks.
- Zero organizational-independence credit (same-account multi-model replay).
