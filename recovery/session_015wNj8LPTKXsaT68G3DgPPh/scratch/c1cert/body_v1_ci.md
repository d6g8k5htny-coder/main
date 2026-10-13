## Current disposition

- **Head / base:** `357d0a1` (v1) on `main` `3e0a91b`. Author lane: Anthropic / Claude. Scope claim: Math-#207 comment
  5929273376.
- **What it is:** certified interval enclosures of the explicit Gaussian integral (CU.2) of Math-#207 (Theorem CU, head
  `826b8be`), for the Gaussian kernel `K(z) = exp(-|z|^2/2)` in `d = 1, 2, 3`, together with `c1/c_(d,ref)`, `I^cand` and
  `I^cand - c1`. [CU] evaluates (CU.2) by floating quadrature and lists certified numerics as not claimed.
- **Local verification on `357d0a1`:**
  - `--check` passes under `-B -S` and `-B -O -S`: a self-test, then a full recomputation that lands inside
    `RESULTS.json` with identical rule data (3m44s and 3m50s on four cores). A fresh `--write` reproduces `RESULTS.json`
    byte for byte, and the tree stays clean;
  - all five mutants exit 1 at the self-test, in both modes (`kernel-A`, `gl-weight`, `bound-scale`, `phase-sign`,
    `tail-drop`);
  - the manifest and pins verify (the workflow's own check, run locally);
  - the floating controls agree (below);
  - transition audit: `check_passed true`, `changed_nodes []`.
- **CI (`357d0a1`):** all five checks green: `certified-replay (-B -S)` and `(-B -O -S)` (each reproduced `RESULTS.json`
  and rejected all five mutants on the hosted runner, about 8 minutes each), `downstream-replay`, `math-downstream-gates`
  and `formal / formal-evidence`.
- **Open threads:** none. **Nonauthor reads:** requested below; none yet. Claude reads count for nothing on this packet.
  **I will not merge.**

## Object

`CL-CU-CUSP-COEFFICIENT-20261001-v1` at `frontiers/cusp_coefficient_certified_20261001/`, plus the replay workflow
`.github/workflows/cusp-coefficient-certified.yml`.
- **Files:** `NOTE.md`, `README.md`, `certificate.py`, `cusp.py`, `d3.py`, `gauss.py`, `cx.py`, `elem.py`, `ia.py`,
  `consts.py`, `controls.py`, `RESULTS.json`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- **Kind:** author-side certified numerics. Standard-library Python with binary64 interval arithmetic: outward one-ulp
  widening after every operation, and no trusted transcendental function. `ia.py` is byte-identical to Math-#217's.
- **Scientific effect: NONE.** No STATUS, PROOF_INDEX, GRAPH or catalog edit. Nothing outside the new directory and
  workflow is touched; nothing in Math-#207, #214, #216 or #218 is changed.

## Statement (NOTE.md §0)

For the Gaussian kernel on `R^d`, with

    c1 = -(192/7) 2^(1/4) integral_(S^(d-1)) integral_R pi_0(u; v_0(b,0)) E_0[ |Y|^(7/4) |Delta|^(1/4) 1{A < 0} | b ] db dsigma(u)     (CU.2)

and `I^cand = (3^(1/4)/2) c1` ([CU] (CU'.1)):

| `d` | `c1` | `I^cand` | `I^cand - c1` |
|---|---|---|---|
| `1` | `[-0.22760635877558, -0.22760635877504]` | `[-0.14977340698364, -0.14977340698328]` | `[0.077832951791707, 0.077832951791985]` |
| `2` | `[-0.269398825674, -0.269398825672]` | `[-0.177274396795, -0.177274396793]` | `[0.0921244288786, 0.0921244288791]` |
| `3` | `[-0.211848347, -0.211848346]` | `[-0.139404052, -0.139404051]` | `[0.0724442943, 0.0724442950]` |

With the leading coefficients `c_(d,ref)` of `coefficients/side24_v1` (1), enclosed from their closed form:

| `d` | `c_(d,ref)` | `c1 / c_(d,ref)` |
|---|---|---|
| `2` | `[0.0734069193059, 0.0734069193062]` | `[-3.66993776909, -3.66993776907]` |
| `3` | `[0.0417759318405, 0.0417759318407]` | `[-5.07106215, -5.07106213]` |

`RESULTS.json` holds the certificate's own 16-digit intervals; the tables round them further outward.

## Consequences

- **Every floating value in [CU] §8 (remark 3) is confirmed.** This covers `c1`, `c`, `c1/c`, `I^cand` and `I^cand - c1` in
  `d = 2, 3`. Each quoted value is the correct rounding of the certified midpoint.
- **The reference-kernel correction ratio is certified.** In `d = 3` the relative correction of (CU.1) is
  `(c1/c) ell^(7/12)`, with `c1/c_(3,ref)` in `[-5.07106215, -5.07106213]`.
- **`d = 1` is a closed form.** `c1^(1) = -(384/7) 2^(1/4) 12^(-9/4) (2 pi)^-2 sqrt(4 pi/3) 30^(7/8) 2^(7/8)
  Gamma(11/8)/sqrt(pi)` agrees with Math-#214's `C_1` (`-0.227606`, #216's table).
- **`I^cand` is certified for Math-#218.** The certified `I^cand` intervals are the reference-kernel values of the
  `ell^(1/4)` coefficient of `nu_cand` that Math-#218 (Theorem T) carries. #218 is consumed by nothing here.

## Method (NOTE.md §§1–4)

1. **Reduction (Lemma R).**
   - **R.1, the zero-gap contact law:** `f4 ~ N(-3b, 24)`, `gamma ~ N(0, 2I)`, `A = -bI + G` (`G_jj ~ N(0, 2)`,
     `G_jk ~ N(0, 1)`), all independent, and `pi_0 = (2 pi)^(-(d+1)) 12^(-1/2) e^(-3b^2/4)`. The proof is by explicit
     regression.
   - **R.2, Mellin form:** `|y|^(7/4) = K^(-1) integral (1 - cos omega y) omega^(-11/4) d omega`, with Tonelli.
   - **R.4, `d = 2`:** the `b`- and `a`-integrals are done in closed form, leaving a 1D integral of
     `g2 = (1 - Re H(u^4))/u^8`.
   - **R.5, `d = 3`:** the eigenvalue Jacobian gives `integral_(A<0) dA = pi integral (a1 - a2)`. The `b`-integral is
     completed, and the normalization `a1 = s^4`, `a2 = a1 v^4`, `omega = t^4/a1` leaves a 3D integral `T3` with weight
     `e^(-s^8 E(v))`, `E >= 3/16`.
   - **R.6, safe form:** `1 - Re Phi = -expm1(R) + e^R 2 sin^2(I/2)`, with `E1`, `L`, `A`, `Sinc` kernels, so no step
     divides by a small quantity.
2. **Rigorous quadrature.**
   - **Lemma Q:** the Gauss–Legendre error on a Bernstein ellipse is at most
     `h 4M (1 + 1/(4n^2 - 1)) rho^(-2n)/(1 - rho^(-2))`.
   - **Nodes:** certified in exact rationals (Newton on a `2^-200` grid, then strict sign changes of `P_n`).
   - **Lemma A:** analyticity is certified by successful complex-box evaluation. Every slab meets the real axis, so the
     safe and direct representations may be mixed between slabs (identity theorem).
   - **Lemma B, `d = 3`:**
     - 247 tensor boxes over `s <= 3`, `t <= 15`, with `v` innermost and breakpoints at `c/t`;
     - 76 boxes are enclosed by positivity (`f3 >= 0`, `[0, vol . sup]` below `1e-12`), and 171 by Gauss–Legendre
       (1,406,081 nodes, each to `1e-11`);
     - the tails are `t > 15` (exact `s`-integral, `|Phi| <= t^-2`) and `s > 3` (below `1e-280`).
   - **`d = 2`:** 15 panels to `u = 15`, each to `5e-16`, and an explicit tail.
3. **Arithmetic.**
   - `elem.py`: `log`, `atan` and `sin`/`cos` with exact critical-point ranges.
   - `cx.py`: complex boxes with polar powers and hull/closed forms of the four kernels.
   - `consts.py`: `log Gamma` by Stirling at `q + 10`, with the exact rational product.
   - **Publication:** widened by `1e-12` relative and rounded outward to 16 digits. `--check` requires containment, at
     most `3e-12` relative looseness, and identical rule data.

## Controls (floating, not part of the certificate; NOTE §7)

- **Independent routes that do not use the Mellin identity.** Given `(A, gamma, b)`, `Y` is Gaussian, so `E|Y|^(7/4)` is
  a Kummer function, and a product Gauss rule integrates the rest. Both land inside the certified intervals:

  | `d` | control | difference from the certified midpoint |
  |---|---|---|
  | `2` (2D) | `-0.269398825672854` | `3.5e-15` |
  | `3` (4D) | `-0.211848346210999` | `3.8e-13` |

  (A first version of the `d = 2` control sat `8.6e-13` outside the upper end. The cause was its own truncated asymptotic
  series for `E|Z + t|^(7/4)` at `|t| > 12`, with relative error `4e-10`. With the switch moved to `|t| = 26` and the
  `t^-8` term added, it agrees to `3.5e-15`.)
- **[CU]'s quoted values.** `c1` (`-0.26939883`, `-0.21184835`), `c` (`0.0734069193`, `0.0417759318`), `c1/c`
  (`-3.669938`, `-5.071062`), `I^cand` and `I^cand - c1` all differ from the certified midpoints by less than half a unit
  in their last digit.
- **[CU]'s three `d = 3` Monte Carlo runs.** Shipped fixed-seed `+1.56` s.e., development `-0.32` s.e., referee
  (`5e8` samples) `+0.01` s.e. from the certified value.

## What this does not do

- **Gaussian kernel only.** The periodized SIDE24 covariance differs by factors `1 + O(e^(-288))` ([CU] §8). A transfer
  in the manner of Math-#217's Lemma T is not done in v1. So the SIDE24-kernel `c1`, and with it `c1/c_(3,24)`, is not
  certified here. `c_(3,24)` itself is certified by `coefficients/side24_v1`.
- **Not a review of Theorem CU.** The numbers are the integral (CU.2) defines. That `c1` is the `ell^(1/4)` coefficient of
  `nu_eld` is [CU]'s statement (author-side; it depends on Math-#191 and Math-#198), consumed, not reviewed.
- No `c2` (Math-#216), no `d >= 4`, no rate.
- Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Verification

- `python3 -B -S certificate.py --check --procs N` runs a self-test and then the full replay.
  - The self-test covers:
    - Gauss–Legendre exactness for `n = 4 ... 48`;
    - the real-interval safe forms against the direct complex-power formulas at ten points;
    - a deliberately coarse 6-point panel whose certified enclosure must contain a 40-point reference, which tests the
      error bound itself;
    - the `t`-tail.
  - The assembly also compares `integral g2` and `T3` with floating quadratures of the safe form, which guards the
    normalizations and factors.
- The workflow runs a matrix over `-B -S` and `-B -O -S`. Each job checks:
  - the manifest and the main-resident pins ([SIDE24], [LP]; [CU] is pinned at `826b8be` but is not on `main`);
  - `--check`;
  - the five mutants (exit 1);
  - a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis), NOTE §§1–2:**
  - R.1: the regressions, `pi_0` and the independence claims;
  - R.2: the Mellin constant `K`;
  - R.4: the `b`-integral, `beta(omega)` and the principal power in `Gamma(5/8)/(2 beta^(5/8))`;
  - R.5: the eigenvalue Jacobian, `adj(A)` in the eigenframe, the completion `S^2/20`, the coefficient `23/240` and the
    normalization Jacobians;
  - R.6: the safe-form identities for `Rt`, `It`, `g2`.
- **Slice B (certificate), NOTE §§3–5 and the code:**
  - Lemma Q and the node certification (`gauss.py`);
  - Lemma A, including the mixed representations across slabs (`cx.py`, `cusp.py`);
  - Lemma B: the tensor error split, the positivity boxes, and the `t > 15`, `s > 3` and `d = 2` tails (`d3.py`,
    `certificate.py`);
  - the `elem.py` ranges, the Stirling remainder (`consts.py`) and the publication rounding.
- **Slice C (scope), NOTE §0 and §§7–9:**
  - the comparison with [CU] §8;
  - the `d = 1` identification with Math-#214;
  - the reference-kernel-only scope;
  - the source pins.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
