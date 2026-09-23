# C095 Freeze — Q0-SHARP

## Target claims

| ID | Statement | Status |
|---|---|---|
| `LOWER_084_RLE005` | 1-q(r,6/5) >= 0.84 r^3 for 0<r<=0.05 | BLOCKED |
| `UPPER_099_NOMINAL_RLE005` | 1-q(r,6/5) <= 0.99 r^3 for 0<r<=0.05 | BLOCKED |
| `UPPER_101_BANDED_RLE005` | 1-q(r,6/5) <= 1.01 r^3 for 0<r<=0.05 | BLOCKED |
| `B_OVER_4_HXX_CANDIDATE` | operator-line Richardson candidate | BLOCKED-EXTERNAL |

## Active gates

### `BR-REP`

- **pass:** Uniform typed six-pin spatial saddle repulsion with explicit constant.
- **kill:** Counterexample to proposed d^3 log(e/d) class in the target law.

### `BR-MARK`

- **pass:** Uniform density bounds for A=(u1+u2)/2 and Z=(u2-u1)/d^3 under typed two-saddle pair-Palm.
- **current:** Mean factorization closed; residual covariance, determinant weight, typing, and domain uniformity open.

### `H4-PATH`

- **pass:** Frozen path family, interval Gaussian blocks, nonnegative Chernoff witnesses, shifted Palm moments, sector weights, path entropy, one-sided uncertainty.
- **kill:** Any required corridor bound exceeds the proposed budget.

### `U-SHAPE`

- **pass_alternative_A:** U'(r)>=0 on (0,0.05]
- **pass_alternative_B:** U''>=-1510.4 on [0.0125,0.025], U''>=-64 on [0.025,0.05], plus origin patch.

### `U-UNCERTAINTY-CAL`

- **pass:** Recover the 0.02 construction or replace it with a new one-sided band containing confidence/tolerance, sample size, multiplicity, domain, and source hash.

### `UB_G_RESIDUAL_UNIFORM`

- **pass:** If the sharpened route consumes UB-G components, Gamma and collar residual coefficients must be explicit.

The full-domain lower product target is **79.988**, not 80.

No new decimal is promoted from sampled rungs.