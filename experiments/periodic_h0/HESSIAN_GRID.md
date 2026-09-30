# A sharper spatial bound from the finite field itself

[Experiment](README.md) · [Evaluated result](results/hessian1/RESULTS.md) ·
[Interpolation and stability argument](APPROXIMATION.md)

The earlier derivative certificate bounds every Fourier term separately.
That is rigorous, but it loses cancellation between the modes. This note
recovers part of that cancellation by evaluating the Hessian on a grid and
bounding its variation between grid points. It then applies the existing
quadratic interpolation theorem to a finer grid of exactly defined dyadic
samples.

The object is the same **finite polynomial defined by the frozen rounded
coefficients**, not the ideal Gaussian field. The new samples below are exact
integer-evaluator centers. They are not the NumPy samples in `nodal1`,
`pilot32`, or `refinement8`. No stored historical receipt is amended.

## Mathematical inputs

Write the real finite polynomial on the side-L square torus as

```text
f(x) = z_0 + 2 Re sum_{k in R} z_k exp(i q k·x),  q = 2 pi/L.
```

There is one representative of each pair `{k,-k}` in `R`. The real and
imaginary parts of every `z_k` are specified dyadic rationals. Put
`rho_k = 2|z_k|`, and use rational upper bounds `rho_k^+ >= rho_k` and
`q_+ >= q`. The Machin pi enclosure and integer square-root bounds in
`finite_certificate.py` supply those upper bounds. No sampling-law or
independence assumption is used.

Two grids have separate roles: a derivative grid of size `m×m`, spacing
`h_m=L/m`, and a sample grid of size `n×n`, spacing `h_n=L/n`. They need not
have equal sizes. The published replay uses the frozen seed label 34000,
`m=256` and `n=1024`. A seed label identifies a stored coefficient record;
it is not a distributional certificate.

## Hessian values at the derivative-grid nodes

For `i,j` in `{x,y}`, multiply each stored complex coefficient by
`-k_i k_j`, set the DC coefficient to zero, and evaluate the resulting
finite polynomial `G_ij` with the unchanged `nodal_core.py`. Then

```text
f_ij(v) = q^2 G_ij(v).
```

Multiplication by an integer preserves the dyadic coefficient convention.
In particular the sign of `k_x k_y` is retained. The core validates its
supported grid, nonaliasing mode layout and fixed-point representability.
Its positive-exponent, unnormalized inverse DFT returns integer centers
and a rational error `E_ij` bounding each complex component of every node.
The [nodal derivation](NODAL_CERTIFICATE.md) supplies the existing stage
recurrence and twiddle enclosure. This delivery does not silently replace
that argument with agreement between two floating programs.

Let `S=2^96`. At a given grid node, arrange the real integer centers as
`A_v=[[a,b],[b,c]]`. The exact spectral norm of its center matrix is

```text
||A_v/S||op = (|a+c| + sqrt((a-c)^2 + 4b^2))/(2S).
```

An integer ceiling square root gives an outward rational endpoint. The
entry-error matrix has operator norm at most its maximum absolute row sum,
so a uniform grid error is

```text
e_H = max(E_xx+E_xy, E_yy+E_xy).
```

Consequently a certified grid-node Hessian bound is

```text
G_H = q_+^2 (max_v ||A_v/S||op^+ + e_H).
```

Using `q_+` here is legitimate because `q^2` multiplies the whole real
matrix and its norm is nonnegative. Every derivative-grid node is consumed;
a numerical search for a likely maximizer is not sufficient.

## Covering the space between grid points

Define the symmetric positive-semidefinite matrix

```text
A = q_+^4 sum_k rho_k^+ (|k_x|+|k_y|)^2 k k^T.
```

Its off-diagonal entry can be negative. Keeping that signed entry is
correct: each summand is a nonnegative multiple of the outer product
`k k^T`. Thus `A` is positive semidefinite even when `A_xy` is negative.
Its largest eigenvalue is enclosed by rational arithmetic and an outward
square root of `(A_xx-A_yy)^2+4A_xy^2`.

Here is the covering argument. Fix any real unit vector `u` and consider
`g_u(x)=u^T Hess f(x) u`. Its Fourier amplitudes are bounded by
`rho_k q^2 (u·k)^2`. The componentwise interpolation estimate in
[APPROXIMATION.md](APPROXIMATION.md), applied now to `g_u`, gives

```text
|g_u - I_m g_u|
  <= h_m^2 q_+^4 sum_k rho_k^+ (u·k)^2
                  (k_x^2 + 2|k_x k_y| + k_y^2)/8
   = h_m^2 u^T A u/8
  <= h_m^2 lambda_max(A)/8.
```

The same fixed triangulation is used for all matrix entries; interpolation
commutes with the quadratic form. For symmetric matrices, the operator norm
equals the supremum of the absolute quadratic form over unit vectors. It
follows that

```text
||Hess f - I_m Hess f||op <= h_m^2 lambda_max(A)/8.
```

The interpolated matrix is a convex combination of its three vertex
matrices, so its norm is no larger than the largest grid-node norm.
Therefore a global bound is

```text
H_grid = G_H + h_m^2 lambda_max(A)^+/8.
```

The same argument for each scalar component gives

```text
||f_ij||_infinity <= M_ij := max_v |f_ij(v)|^+
       + h_m^2 q_+^4 sum_k rho_k^+ |k_i k_j|
                                      (|k_x|+|k_y|)^2/8.
```

The replay records the earlier coefficient-triangle spatial bound separately
for comparison. Set `H=H_grid`. From the new global bounds it uses

```text
C = min{ H/4, (M_xx+M_yy)/8 + M_xy/4 },
B_n = h_n^2 C.
```

This is the original interpolation theorem with sharper certified inputs.
The torus seams are covered by the same periodic triangles; there is no
unexamined boundary strip. Every covering constant is evaluated using the
full retained finite mode set.

## A million samples without a million stored values

For the finer grid, run the unchanged exact integer evaluator on the
original coefficient record. If its real center is `a_v` and its final
component error is `E_n`, **define** the supplied sample to be
`s_v=a_v/S`. This is an exact dyadic value; no float conversion is performed.
The existing enclosure immediately gives

```text
max_v |s_v-f(v)| <= E_n = eta_n.
```

The certificate records every stage error bound and the SHA-256 of the
canonical JSON list of decimal real-center integer strings, in row-major
x-then-y order. Replay regenerates the entire array and checks that digest.
The digest establishes identity; the error proof and exact replay establish
the bound. There is no claim that a digest alone authenticates an execution.

The existing interpolation/stability argument now gives

```text
d_B(D(f), D(C_s)) <= eta_n+B_n,
```

where `C_s` is the mathematical vertex cubical filtration of these dyadic
samples. The same bound applies separately to the PL filtration on either
declared diagonal. This uses the previously stated q-tameness and matching
theorems with exactly the hypotheses in `APPROXIMATION.md`; it introduces
no new external theorem or claim of novelty.

This certificate does **not** run a persistence implementation or identify
any numerical barcode with `D(C_s)`. That implementation boundary remains
open. It does not certify the historical experiment's sample arrays or
relabel their histogram counts.

## What the improved budget permits

For `epsilon=eta_n+B_n`, the existing endpoint-safe bin inequalities require
`a>2epsilon` for a clean upper bound on `[a,b)` and `b-a>4epsilon` for a
nonempty contracted interval. The result table evaluates those strict
rational conditions. It supplies no bar counts. Even when both conditions
hold, one must compute and justify the expanded/contracted counts; agreement
of unexpanded histogram bins is not a substitute.

The result concerns one finite field and new dyadic samples. A claim about
the intended Gaussian ensemble additionally needs its model transfer,
infinite-field tail, sampling uncertainty, justified asymptotic lifetime
range and remainder. These obligations and human/independence review gates
are unchanged.

## Replay and evidence

From the repository root, with ordinary Python and no numerical packages:

```sh
python -B -S experiments/periodic_h0/hessian_grid_certificate.py --verify experiments/periodic_h0/results/hessian1
```

The compact certificate records exact rationals; displayed decimal upper
endpoints are rounded upward. `RUN.json` binds the derivation, evaluator,
certificate implementation, exact coefficient file and output bytes;
`RUN.sha256` binds that complete receipt. The producer refuses an existing
output directory. The verifier rejects altered sources, values, bounds,
scope flags, metadata types, receipt bytes and rendered result text.

Tests exercise signed mixed modes, exact small-grid formulas, spectral norm
enclosures, rational boundary equalities and deliberately modified
certificates. Review disposition and actual reviewer lineage belong to the
source-bound PR review record; this note does not anticipate its verdict.
