## 0. Statement

**Setting** ([LP] section 1 with `d = 3`, `m = 2`; as in [G3] section 0):
- The torus `X = (R / L Z)^3` and the kernel `K_L(z) = sum_n exp(-|z + Ln|^2/2) / sum_n exp(-|Ln|^2/2)`.
- An orthonormal frame `R` with axial unit vector `u`, and the pins `M = -(r/2) u`, `S = (r/2) u`, `f(M) = b`,
  `f(S) = b - k r^3`, `grad f(M) = grad f(S) = 0`.
- The Gaussian regression law `Q = Q_{r,b,k,R}`, the typed weight `W_r = |det H_M det H_S| 1{H_M < 0, index H_S = 2}`,
  `Z_r = E_Q W_r`, and `dQ^W = (W_r / Z_r) dQ`.
- In frame coordinates `(x, y) in R x R^2`: the cylinder `D = [-2r, 2r] x ball(0, 2r)`, the partial-block norms
  `M_j = max_{a + c = j} sup_D ||partial_x^a D_y^c f||_op` ([CAP] section 1), and the good event
  `G_r = {lambda_1 > (4/(3k)) r M_3^2, r M_4 <= 3k/10}` with `lambda_1 = lambda_min(-D_y^2 f(M))`.
- Theorem A of [LP] is `0 <= 1 - p_r <= C r^3` ([LP] (1.1)). Its proof bounds the larger probability `Q^W(G_r^c)` in (7.8) and
  uses `1 - p_r <= Q^W(G_r^c)` (§8).
- A **cap-route constant** is any `C` with `Q^W(G_r^c) <= C r^3` for `0 < r <= r_*`. Every cap-route constant is a constant
  for [LP] (1.1); the converse fails (Math-#203 v1.2, section 5, item 1).

**Theorem E3.** Let `d = 3`, `B = [0, 1]` and `K = [1/2, 2]`. For every torus side `L >= 10`, every orthonormal frame `R`, every
`b in B`, `k in K`, every `r_*` in the table, and every `0 < r <= r_*`,

    Q^W(G_r^c) <= C3(r_*) r^3        and        Z_r / r^2 >= z3(r_*).

The same holds for the reference kernel `exp(-|z|^2/2)` on `R^3`. Every enclosure below contains the case `M_1 = 0`,
`Theta = 1` of section 2. The table lists, for each `r_*`:
- `C3(r_*)`, an exact rational, with its decimal;
- upper bounds for `C3 / c_G^(3)(0, 2)` and `C3 r_*^3`, and the floor `z3(r_*)`;
- the cell (`r`-band; `b`; `k`) at which the maximum is attained.

| `r_*` | `C3(r_*)` | `C3 / c_G^(3)(0,2) <=` | `C3 r_*^3 <=` | `z3(r_*)` | maximizing cell |
|---|---|---|---|---|---|
| `1/4096` | `56887004491/2500 = 22754801.7964` | `1.014738` | `0.0003311260` | `6.021091` | `[15/65536, 1/4096]`; `[0, 1/512]`; `[63/32, 2]` |
| `1/2048` | `57060167561/2500 = 22824067.0244` | `1.017827` | `0.0026570712` | `5.998005` | `[15/32768, 1/2048]`; `[0, 1/512]`; `[63/32, 2]` |
| `1/1024` | `5743533573/250 = 22974134.2920` | `1.024519` | `0.0213963299` | `5.951763` | `[15/16384, 1/1024]`; `[0, 1/512]`; `[63/32, 2]` |
| `1/512` | `14583076547/625 = 23332922.4752` | `1.040519` | `0.1738438195` | `5.859005` | `[15/8192, 1/512]`; `[439/1024, 421/512]`; `[31/16, 63/32]` |
| `1/256` | `3714819319587/625 = 5943710911.3392` | `265.06` | `354.3` | `5.672384` | `[7/2048, 15/4096]`; `[0, 1/2]`; `[1/2, 3/4]` |
| `1/128` | `69971360017.7629` | `3120.3` | `33365` | `5.294648` | `[15/2048, 1/128]`; `[1/2, 1]`; `[1/2, 3/4]` |
| `1/64` | `17254355056897.0472` | `769448` | `6.6 x 10^7` | `4.520628` | `[15/1024, 1/64]`; `[1/2, 1]`; `[1/2, 3/4]` |

`c_G^(3)(0, 2)` is replaced by its certified lower end `22424320.655069` ([G3]).

**Corollary (explicit Theorem A in `d = 3` on the band).**
- **Explicit pairs.** Since `0 <= 1 - p_r <= Q^W(G_r^c)` ([LP] §8), [LP] (1.1) holds in `d = 3` on `B x K`, uniformly over frames
  and `L >= 10`, with every pair `(C3(r_*), r_*)` of the table. The pairs that are informative at `r = r_*` (`C3 r_*^3 < 1`)
  are exactly those with `r_* <= 1/512`. Examples:
  - `(C, r_*) = (23332922.4752, 1/512)`, where the bound at `r = r_*` is below `0.174`;
  - `(22974134.2920, 1/1024)`, where it is below `0.0214`.
- **What was missing.** [LP] §16 lists "a numerical `r_*` or `C` on a prescribed band" among the items it does not supply.
  [CAP]'s probabilistic corollary is planar only ("No 3D eight-pin matrix-boundary estimate ... is proved here"). Before
  this packet no explicit constant was on record in any dimension `d >= 3`.
- **Near-optimal for the cap route.**
  - Every `d = 3` cap-route constant valid for the reference kernel at `(b, k) = (0, 2)` is at least
    `c_G^(3)(0, 2) > 22424320.65` ([G3] Corollary 1).
  - This certificate's scope includes the reference kernel.
  - The constants are within `1.5%`–`4.1%` of that limit for `r_* <= 1/512`.
  - The cap route in `d = 3` can be informative only below `r = c_G^(3)(0, 2)^(-1/3) = 3.546 x 10^-3`, so `r_* = 1/256`
    (`C3 r_*^3 >= c_G^(3) r_*^3 = 1.34`) is uninformative for every cap-route constant. At `1/256` and above, the certified
    constants are dominated by the rare branches at `k near 1/2`.
- **Against the plane.**
  - `C3(r_*)/C(r_*)` is `4.249` at `r_* = 1/4096` and `4.308` at `1/512` ([E] Theorem E).
  - The sharp ratio is `c_G^(3)(0, 2)/c_G(0, 2) = R_3(0) = (32 + 28 sqrt2)/17 = 4.2116` ([G3]).
- **Normalizer floor.** `z3(r_*)` is an explicit `z_*` for [LP] (5.5) in `d = 3` on `(0, r_*]`.
  - The contact value at the band's weakest corner `(b, k) = (0, 1/2)` is `36 k^2 m_(3,0) = 9 (7 - 4 sqrt2)/2 = 6.0442`.
  - `z3(1/4096) = 6.0211` lies within `0.4%` of it; the last band `[0, 2^-29]` gives `6.044155`.
  - `Z_r / r^2 >= 4.5206` on all of `(0, 1/64]`.
