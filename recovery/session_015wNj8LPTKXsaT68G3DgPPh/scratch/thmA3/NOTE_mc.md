`theorem_a3.py --mc R B K N SEED` is an end-to-end control computed from the definitions. It works as follows:
- It samples the degree-7 Taylor jet at the midpoint under `Q`, from the exact pin series evaluated in floating point.
- It evaluates `W_r` from the Hessians at `M` and `S`.
- It evaluates `M_3`, `M_4` on a 23-point grid of `D`, maximizing the binary forms over 24 angles.

Pooled over 16 seeds:

| `(r, b, k)` | samples | `Q^W(G_r^c)/r^3` | `Z_r/r^2` | certified `C3(r)` |
|---|---|---|---|---|
| `(1/512, 0, 2)` | `320000` | `1.572e7 +- 0.026e7` | `95.7 +- 1.1` | `23332922.48` |
| `(1/1024, 0, 2)` | `640000` | `1.906e7 +- 0.028e7` | `97.5 +- 0.9` | `22974134.29` |

- **The normalizer.** `Z_r/r^2` agrees with the contact value `36 k^2 m_(3,0) = 96.68`.
- **The failure probability.**
  - `Q^W(G_r^c)/r^3` lies below `c_G^(3)(0, 2) = 2.2424e7` at these radii.
  - It approaches the limit linearly: `2 x 1.903e7 - 1.569e7 = 2.237e7`, within `0.3%` of `c_G^(3)` ([G3]).
  - The first-order correction is negative and about three times the planar one (the planar control of [E] at
    `(1/512, 0, 2)` was `9%` below `c_G`). It comes from the eigenvalue Jacobian `lambda_2 - lambda_1`, which the certificate
    bounds by `lambda_2`.
- **Caveats.** The grid suprema lie slightly below the true ones, so the Monte Carlo failure probability is, if anything,
  underestimated. These are consistency checks, not part of the proof.
