## Current disposition

- **Head / base:** `@@HEAD@@` (v1) on `main` `3e0a91b`. Author lane: Anthropic / Claude. Scope claim: Math-#208 comment 5922961739.
- **What it is:** the `d = 3` counterpart of Math-#206. Explicit cap-route constants for [LP] Theorem A in dimension three on
  the C8 band, together with a normalizer floor. No explicit constant was on record in any `d >= 3`; [CAP]'s probabilistic
  corollary is planar.
- **Local verification:**
  - `--check` passes under `-B -S` (11 sampled bands);
  - `--check-full` replays all 121 bands under `-B -O -S`;
  - the regeneration run under `-B -S` reproduces `RESULTS.json` byte for byte;
  - all five mutants exit 1;
  - manifest and pins verify;
  - transition audit: `check_passed true`, `changed_nodes []`.
- **CI:** pending.
- **Open threads:** none. **Nonauthor reads:** requested below; none yet. Claude reads count for nothing on this packet.
  **I will not merge.**

## Object

`CL-C8-THEOREM-A-CONSTANTS-D3-20261001-v1` at `frontiers/c8_theorem_a_constants_d3_20261001/`, plus the replay workflow
`.github/workflows/c8-theorem-a-constants-d3.yml`.
- **Files:** `NOTE.md`, `theorem_a3.py`, `RESULTS.json`, `README.md`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- **Kind:** author-side theorem with explicit constants, certified by exact rational interval arithmetic (standard library only).
- **Scientific effect: NONE.** C8 stays OPEN. No STATUS, PROOF_INDEX, GRAPH or catalog edit.
- **Relation to other packets:** follows Math-#206 (the planar Theorem E, whose proof it generalizes) and Math-#208 (the sharp
  coefficient `c_G^(3)`, a lower bound for every `d = 3` cap-route constant). Nothing under the #206, #208 or #203 packets is
  touched.

## Statement (NOTE.md §0)

**Theorem E3.** Let `d = 3`, `B = [0, 1]` and `K = [1/2, 2]`. For every torus side `L >= 10`, every orthonormal frame, every
`b in B`, `k in K`, and every `0 < r <= r_*`:

    Q^W(G_r^c) <= C3(r_*) r^3,      Z_r / r^2 >= z3(r_*).

Here `G_r` is the good event of [LP] §7 / [CAP] (1), with partial-block operator norms on `D = [-2r, 2r] x ball(0, 2r)`. The
same holds for the reference kernel.

| `r_*` | `C3(r_*)` | `C3 / c_G^(3)(0,2) <=` | `C3 r_*^3 <=` | `z3(r_*)` |
|---|---|---|---|---|
| `1/4096` | `56887004491/2500 = 22754801.7964` | `1.014738` | `3.4e-4` | `6.021091` |
| `1/2048` | `57060167561/2500 = 22824067.0244` | `1.017827` | `2.7e-3` | `5.998005` |
| `1/1024` | `5743533573/250 = 22974134.2920` | `1.024519` | `0.0214` | `5.951763` |
| `1/512` | `14583076547/625 = 23332922.4752` | `1.040519` | `0.174` | `5.859005` |
| `1/256` | `5943710911.34` | `265` | `354` | `5.672384` |
| `1/128` | `69971360017.76` | `3120` | `3.3e4` | `5.294648` |
| `1/64` | `1.7254e13` | `7.7e5` | `6.6e7` | `4.520628` |

## Consequences

- **Explicit [LP] Theorem A in `d = 3`.** Since `1 - p_r <= Q^W(G_r^c)` ([LP] §8), every pair above is an explicit `(C, r_*)`
  for [LP] (1.1) in `d = 3`. The informative pairs are those with `r_* <= 1/512`, for example `(23332922.4752, 1/512)`, where
  the bound at `r = r_*` is below `0.174`. [LP] §16 lists such a pair among the items it does not supply.
- **Near-optimal for the cap route.**
  - Every `C3` is a cap-route constant, hence at least `c_G^(3)(0, 2) = 22424320.655` (Math-#208).
  - For `r_* <= 1/512` the constants are within `1.5%`–`4.1%` of that limit.
  - Since `c_G^(3) (1/256)^3 > 1`, no `d = 3` cap-route constant is informative at `r_* = 1/256`. The large values there come
    from the rare branches at `k near 1/2`.
- **Against the plane.** `C3/C` is `4.249` at `r_* = 1/4096` (Math-#206), against the sharp ratio
  `R_3(0) = (32 + 28 sqrt2)/17 = 4.2116`.
- **Normalizer floor.** `Z_r / r^2 >= 4.5206` on `(0, 1/64]` and `>= 5.859` on `(0, 1/512]`. The contact value at `(0, 1/2)` is
  `9 m_(3,0) = 6.0442`.

## Proof outline (NOTE.md §§1–9)

The proof follows Math-#206 section by section. It uses only [LP]'s definitions and pins and does not consume [LP] (4.1),
(4.2), (5.4), (5.5) or (7.5)–(7.7). The three-dimensional changes:

1. **Identities.** Eight pins and 31 targets. `W_r <= r^2 P1 P2` with `P1 = c1 det B - r v^T adj(B) v` and
   `P2 = c2 det B_S + r e^T adj(B_S) e`, an inequality in `d = 3`.
2. **Lemma 1 (law).**
   - Exact covariance series in three coordinates, regular for all 780 pairs.
   - A torus remainder bounded by Cauchy estimates on the polydisc in `C^3`: `M_1(10) < 1.94e-13`, uniform over frames and
     `L >= 10`.
   - Regression on the three entries of the transverse Hessian `B`, in Frobenius-isometric coordinates.
3. **Lemmas 2–3.**
   - Averaging identities applied blockwise (`f_xxx`, `grad_y f_xx`, `D_y^2 f_x`, `D_y^3 f`).
   - Fourth-derivative blocks bounded through Frobenius norms of the midpoint Taylor series.
4. **Lemma 4.**
   - `||B - mu_B||_F <= lambda_1 + lambda_2 + ||mu_B||`, so the near-root threshold `U` depends on `lambda_2`.
   - `lambda_2` stays in the integral rather than being cut off, which keeps the last-band scaling argument intact.
5. **Main term.**
   - Eigenvalue coordinates `dB = (lambda_2 - lambda_1) d lambda d phi`.
   - An angle-free Gaussian majorant of the density.
   - The `lambda_2`-weight written as `I_3 nu + r Omega I_2 nu'`.
   - Moments of the threshold through chi positive-part moments for the free blocks `v` (`chi_2`), `Omega'` (`chi_3`) and
     `Y` (`chi_4`).
6. **Lemma 5' (floor).**
   - A pointwise lower bound on `W_r`.
   - A Loewner-order coupling with the contact law.
   - Closed-form contact moments `E[det^a tr^c 1{B > 0}]`, e.g.
     `m_(3,m) = E[(s^4 - 4s^2 + 8)1{s>0}] - 4 sqrt2 e^(-m^2/4) Phi(m/sqrt2)`, which gives `(7 - 4 sqrt2)/2` at `m = 0`.
7. **Rare branches.**
   - The weight norm through `det B` moments.
   - The `M_4` blocks through operator norms on an angular net with Bernstein's inequality (Frobenius tails are about 100
     times too generous for the quartic block).
   - Chi tails for `lambda_2`.
8. **Sweep.** 121 `r`-bands by 39 `(k, b)` boxes. The `b`-grids are refined toward the corner `(0, 2)`; the last band uses the
   scaling argument.

## Controls (floating point, not part of the certificate)

@@MC@@

## What this does not do

- `d = 3` only.
- The band only; no SIDE24 point (the [CAP] SIDE24 law is planar).
- No lower bound for the constant of (1.1).
- `c_{B,K}` and `c_{d,L}` are not addressed.
- Not a review of [LP] or [CAP].
- C8 stays OPEN.
- Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Verification

- `python3 -B -S theorem_a3.py --check --procs N` replays exactly:
  - the parameters and 11 sampled bands;
  - the `C` table;
  - the canonical layout of `RESULTS.json`;
  - `m_(3,0) = (7 - 4 sqrt2)/2`;
  - `C3 > c_G^(3)(0, 2)`.
- It takes about 2.5 minutes on four cores.
- `--check-full` replays all 121 bands in about 15 minutes on four cores.
- Mutants `no-delta`, `cap-half`, `no-zfloor-correction`, `no-om-term` and `no-lambda2` exit 1.
- The workflow checks:
  - the manifest and the pins ([LP] `dfed3b8d`, [CAP] `0633aca3`);
  - `--check-full` and `--check` in the two interpreter modes;
  - the five mutants;
  - a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis):** NOTE §§1, 3–7.
  - The identities (I1)–(I3) and the typed-support inequality.
  - Lemmas 2–3 (blockwise averaging; Frobenius Taylor bounds).
  - The regression on `B` and the `lambda_2`-dependent cover.
  - The eigenvalue-coordinate integration with the angle-free majorant and the two density bounds.
  - The floor with the Loewner coupling and the closed-form contact moments.
  - The rare branches, including the angular net.
- **Slice B (engine and certificate):** NOTE §2 and §§8–9, and `theorem_a3.py`.
  - The 3D series and its regularity, the polydisc torus bound, and the interval regressions and Cholesky factors.
  - The chi moment calculus, the `lambda_2`-moment recursion and the contact-moment polynomials.
  - The box grid, the last band, and rounding.
- **Slice C (scope and consequences):** NOTE §0 and §§10–11.
  - Cap route versus (1.1).
  - The `L >= 10`, all-frame and reference-kernel scope.
  - The comparisons with `c_G^(3)` and Math-#206.
  - The informative range, and the non-claims.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
