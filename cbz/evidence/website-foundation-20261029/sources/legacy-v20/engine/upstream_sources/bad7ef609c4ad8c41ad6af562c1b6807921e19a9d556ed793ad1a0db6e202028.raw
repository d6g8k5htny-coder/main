# Additive erratum — endpoint Hessian congruence in UNIFORM_MATRIX_CAP_AND_LIFETIME

**Object:** UNIFORM-MATRIX-CAP-LIFETIME-20260924-v1-ERRATUM-1  
**Scientific effect:** correction of a displayed linear-algebra sentence only; no theorem promotion.

Source object: `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md`, raw SHA256 `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7`.

## Correction

In Section 5, immediately after the convergence of endpoint Hessians, the source says:

> Here the congruence uses `diag(sqrt(r),I)`, so its off-diagonal entries are `sqrt(r) beta_i`.

Given the source's own definition

    H_i = [[r alpha_i, r beta_i^T],
           [r beta_i, A_i]],

that displayed congruence factor is incorrect. The matrix used by the argument is

    D_r = diag(r^(-1/2), I),

because

    D_r H_i D_r
      = [[alpha_i, sqrt(r) beta_i^T],
         [sqrt(r) beta_i, A_i]].

It also satisfies

    det(D_r H_i D_r) = det(H_i)/r,

which is exactly the determinant normalization used in Sections 5–7.

With `alpha_M -> -6k`, `alpha_S -> 6k`, bounded `beta_i`, and `A_i -> A_0`, this corrected congruence gives

    D_r H_M D_r -> diag(-6k,A_0),
    D_r H_S D_r -> diag( 6k,A_0).

No formula (5.3), (5.4), (5.5), or the subsequent soft-factor/eigenvalue integration is changed by this correction. The original file is retained byte-for-byte for provenance; consumers of its Section 5 type-convergence argument must read this erratum with it.

## Review requirement

Because this correction touches the load-bearing full-normalizer/type-convergence interface A3, the pending nonauthor A1–A7 review on main #63 should review the source **together with this erratum**. A green test or obvious intended meaning is not acceptance.
