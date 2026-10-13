**Monte Carlo of the genuine coefficient.** `theorem_b.py --controls` estimates `(A_r / A_0^ref - 1) / r^2` for the reference
kernel at `r = 1/8` and `1/16` and at the four corners of the band:
- the law at `r` comes from the exact series, which is valid at every `r` (the Taylor models stop at `1/64`);
- the pin ratio is exact;
- `Z_r / r^2` is sampled as `P1 P2 1{typed}` (seed `20261001`; `2 x 10^5` samples per point in the plane, `10^5` in `d = 3`).

Values are shown with one standard error:

| `d` | `r` | `(b, k) = (0, 1/2)` | `(0, 2)` | `(1, 1/2)` | `(1, 2)` |
|---|---|---|---|---|---|
| `2` | `1/8` | `0.08 ± 0.32` | `13.19 ± 0.39` | `0.64 ± 0.22` | `13.43 ± 0.26` |
| `2` | `1/16` | `0.16 ± 1.28` | `12.74 ± 1.34` | `1.09 ± 0.86` | `13.35 ± 0.90` |
| `3` | `1/8` | `2.24 ± 1.61` | `10.89 ± 1.75` | `0.84 ± 0.73` | `12.88 ± 0.87` |
| `3` | `1/16` | `-1.95 ± 5.71` | `17.29 ± 6.16` | `2.64 ± 2.99` | `17.45 ± 3.15` |

The certified `a_up` at `R = 1/4096` on the four corner boxes (`b`- and `k`-width `1/8`):

| `d` | `(0, 1/2)` | `(0, 2)` | `(1, 1/2)` | `(1, 2)` |
|---|---|---|---|---|
| `2` | `4.818760` | `13.822900` | `3.918911` | `13.742115` |
| `3` | `25.666574` | `16.876052` | `12.525134` | `14.762963` |

What the comparison shows:
- **Plane, `k = 2`.** The coefficient is `13.2`–`13.4`, and `a_up = 13.823` is attained here, about `4%` above it.
  - Most of `a_up` is the exact pin-density limit `12.75 r^2` (`0.75` from `det`, `12` from `e_kk`).
  - The `Z`-part is small and negative in the Monte Carlo (about `-0.7`); the certificate bounds it above by `1.07`.
- **Plane, `k = 1/2`.** The coefficient is `0.1`–`1.1`, against box constants `3.9`–`4.8`.
  - Here the `Z`-part is about `-0.8` to `-1.4`, while the certificate bounds it above by `2.9`.
  - This region carries most of the weight of `c'`, so `c'_up` is several times the genuine weighted coefficient.
- **`d = 3`.** At `k = 2` the coefficient is about `11`–`17`, against box constants `14.8`–`16.9`.
  - At `(0, 1/2)` the coefficient is about `0`–`2`, against the box constant `25.67`, which is the global `a_up`.
  - There `23.5` of that constant is the `Z`-part. It consists mostly of the Hölder bounds of `r^2 E[D xi_3 1]` (`13.9`) and
    `r^2 E[(v' A* v)(e' A* e) 1]` (`8.7`), relative to `36 k^2 m_3`.

**Cap-route limits** (section 0). In floating point, `Gamma_{B,K} = |S^(d-1)| int A_0^ref c_G / (3 k^(5/3))` is:
- `Gamma_2 = 307.51`, with the planar `c_G = R(b) J(k)` of [G] (`J(k) = 2359296 k^3 (1 + delta(k))`);
- `Gamma_3 = 540.97`, with the `d = 3` ratios of [G3].

So `C_{B,K}(1/4096)` is `1.31 Gamma_2` and `2.23 Gamma_3`.
