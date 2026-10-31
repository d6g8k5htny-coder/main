# Second pass after landing the replay packet

Scientific effect: NONE. Does not close D1 or D5.

## D1 Section 5 — physical Hessian versus congruent matrix

Parent `UNIFORM_MATRIX_CAP_AND_LIFETIME.md` §5 defines

```
alpha_i = f_xx(i) / r
beta_i  = nabla_y f_x(i) / r
A_i     = D_y^2 f(i)
H_i     = [[ r alpha_i , r beta_i^T ], [ r beta_i , A_i ]]
```

so `H_i` is the *physical* Hessian. The block-determinant identity written there,

```
det H_i / r = alpha_i det A_i - r beta_i^T adj(A_i) beta_i
```

is the matrix-determinant lemma for that physical matrix and matches Lemma R3.2 after the substitution `t = r^2` on off-diagonals of size `r beta` (equivalently `t = r` on off-diagonals of size `sqrt(r) beta` of the *congruent* matrix).

The same section then says the congruence is `diag(sqrt(r), I)` and that off-diagonal entries are `sqrt(r) beta_i`. That sentence describes the congruent matrix

```
K_i = [[ alpha_i , sqrt(r) beta_i^T ], [ sqrt(r) beta_i , A_i ]]
```

not `H_i` itself. The two matrices are related by `K_i = D^{-1} H_i D^{-1}` with `D = diag(sqrt(r), I)`.

Limit claimed after congruence:

```
K_M → diag(-6k, A_0),    K_S → diag(+6k, A_0)
```

in probability, and

```
Z_r / r^2 → (6k)^2 E[ det(A_0)^2 1{A_0 < 0} ].
```

Tracking check, not a closure:
- (5.2) gives `alpha_M → -6k`, `alpha_S → +6k`.
- `||A_S - A_M|| ≤ r M_3` so both transverse blocks couple to one `A_0`.
- `det H_i ∝ r alpha_i det A_i` at leading order, so `det H_M det H_S ∝ r^2 (6k)^2 det(A_0)^2`.
- The displayed normalizer floor is therefore the *same* leading-order bookkeeping as the fold ledger, provided the type indicators and uniform integrability in (5.3)–(5.5) hold.

What this does **not** do: it does not repair the reopened §2–7 selection chain, residual finite-jet rank, or the global elder convention. Those remain the D1 blockers. The useful addendum is only: keep `H_i` and `K_i` in separate symbols so the off-diagonal power of `r` cannot drift between `r` and `sqrt(r)` in later citations.

## D5 — axial cubic identity (deterministic slice)

On the axis, two critical points at `x = ± r/2` with `f'(± r/2) = 0` force the unique cubic model

```
f'(x) = 6k (x^2 - r^2/4)
```

once the height gap is normalized to `f(r/2) - f(-r/2) = -k r^3`. Check:

```
∫_{ -r/2 }^{ r/2 } 6k (x^2 - r^2/4) dx = 6k [ x^3/3 - (r^2/4) x ]_{ -r/2 }^{ r/2 }
  = 6k ( - r^3/6 ) = -k r^3.
```

Hence the repair identity

```
f_x(ru, 0) / r^2 = 6k (u^2 - 1/4)
```

is exact for the cubic, and the published form

```
f_x(ru, rv) / r^2 = 6k (u^2 - 1/4) + f_xxz(0) u v + (1/2) f_xzz(0) v^2 + O(r)
```

is that cubic plus the next transverse jets. At the pins `u = ± 1/2` the `6k` term vanishes, as it must.

This closes only the deterministic cubic bookkeeping. It does not supply the inner microdisk bound, the collar to the reviewed annulus, shrinking-collision moments, or a conditional Gaussian compensation density. Those remain AMEND.

## Still fail-closed

No ACCEPT flip. No `lemma_closed` flip. Next status-raising step is still a source-bound independent review of D1 §2–7.
