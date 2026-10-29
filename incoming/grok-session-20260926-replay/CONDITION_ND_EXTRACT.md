# Condition (ND) — File 3 §7.3 extract (still OPEN)

**Scientific effect:** NONE.
**Does not accept Lemma I, File-1 Theorem A, or GitHub D1.**
Source: attached File 3 (2026-07-05), §7.3. Not a Math- default proof body.

## Exact matrix condition

Let `Σ_2(r)` be the covariance of the **15-coordinate** nested frame

- 6 gradient-normalized: pair four G's + third-point `W_t, W_s`
- 3 marked values: `V+, V-, W_0`
- 6 pair Hessians: `H^+, H^-`

`Σ_2(r) → Σ_2(0)` entrywise, uniformly in `(z,θ)` on compacts.

**Condition (ND).** For every `z ∈ D(Z_0)` off Appendix-B exceptional curves (14-frame replacement on them), and every `θ`:

    det Σ_2(0)(z, θ) > 0

with a positive lower bound uniform on those compact strata.

Equivalently: leading order of `det Σ_raw(r)` under the two-scale degeneration equals **exactly** the divided-difference ladder order, with nonvanishing coefficient. Excess vanishing (hidden constraint) falsifies (ND).

## What is known vs not

| Fact | Status in File 3 |
|---|---|
| Each limiting coordinate has positive variance (Lemma H) | claimed |
| `Ξ_1(0)` pair block nondegenerate | claimed |
| Cross-block collapse of `(W_0,W_t,W_s)` against the pair frame | **OPEN** |
| Designated resolution | BF symbolic + interval `det` on a certified `(z,θ)` grid |
| Fallback | integrable vanishing still gives some `β>0`; open-set collapse kills the route |

## What (ND) would buy

If (ND) holds with any positive lower bound:

    E[N_inn | pair] ≤ C r^4.

Window mark is void on the inner `r^2`-balls (`|f(y)-f(s)|=o(ℓ)`). So `β` and the outer `ℓ ~ r^3` window cost are independent. On the clean branch the inner term is *sub-leading* vs `r^3`.

Lemma I without (ND) stays **OPEN**. File 5 Theorem B does not upgrade this.

## Not GitHub D1 / D5

Parent `T_r` is a 2(d+1)-pin contact map, `|det T_r|=12 r^{-(d+3)}`. Different matrix. D5 microdisk is a later pin-neighborhood chart. Do not identify them.

This session did **not** assemble `Σ_2(0)` or evaluate the determinant.
