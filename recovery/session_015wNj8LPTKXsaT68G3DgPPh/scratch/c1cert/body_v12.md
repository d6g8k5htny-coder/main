## Current disposition

- **Head / base:** `7a04873` (v1.1 plus metadata fixes) on `main` `3e0a91b`. Author lane: Anthropic / Claude. Scope claim: Math-#207 comment
  5929273376.
- **What it is:** certified interval enclosures of the explicit Gaussian integral (CU.2) of Math-#207 (Theorem CU, head
  `826b8be`). They cover:
  - the Gaussian kernel `K(z) = exp(-|z|^2/2)` in `d = 1, 2, 3`, with `c1/c_(d,ref)`, `I^cand` and `I^cand - c1` (v1);
  - **v1.1, Lemma S:** the [P] torus field of every side `L >= 24` in `d = 2, 3`, including SIDE24, the manuscript's field.
    `|c1^(L)/c1^(ref) - 1| <= 1.4e-104`, so `c1/c_(3,24)` is certified in `[-5.07106215, -5.07106213]`. This delivers
    the SIDE24 ratio promised in the scope claim.

  [CU] evaluates (CU.2) by floating quadrature and lists certified numerics as not claimed.
- **Local verification on `3ac897a`:**
  - `--check` passes under `-B -S` and `-B -O -S`: a self-test, then a full recomputation that lands inside
    `RESULTS.json` with identical rule data, including Lemma S's exact constants (4m18s and 3m43s on four cores). Every
    v1 value in `RESULTS.json` is unchanged; v1.1 only adds keys;
  - all six mutants exit 1 at the self-test, in both modes (`kernel-A`, `gl-weight`, `bound-scale`, `phase-sign`,
    `tail-drop`, and the new `image-order6`);
  - the manifest and pins verify (the workflow's own check, run locally; three main-resident pins);
  - the floating controls agree (below);
  - transition audit: `check_passed true`, `changed_nodes []`;
  - on v1 (`357d0a1`), a fresh `--write` also reproduced `RESULTS.json` byte for byte.
- **CI:**
  - v1 (`357d0a1`): all five checks green. Both `certified-replay` jobs reproduced `RESULTS.json` and rejected the five
    mutants on the hosted runner, in about 8 minutes each.
  - v1.1 (`3ac897a`): all five checks green. Both `certified-replay` jobs reproduced `RESULTS.json`, including the
    Lemma S constants, and rejected all six mutants on the hosted runner.
- **After v1.1 (metadata only; the certificate, `RESULTS.json` and every number are unchanged):**
  - `666a3c8` fixes both Copilot findings (review 5378756519). The workflow path filter now includes the three consumed
    main-resident sources, and `SOURCE_MAP.json` records `delivered_under`: Dylan Roy's post-stop instructions, quoted
    verbatim. Both threads are answered and resolved.
  - `7a04873` makes the `ia.py` docstring state the Phi-series bound as implemented (`n <= 24`, `< 1e-14`), the same
    change as Math-#217 v1.3, so the two files stay byte-identical.
  - The manifest, pins and self-test pass locally; the transition audit gives `check_passed true`. CI on `7a04873` is
    pending.
- **Open threads:** none.
- **Nonauthor reads:**
  - OpenAI / GPT-6 Astra Pro: **R.1 ACCEPT** (review 5378866830 on `357d0a1`). It confirms the zero-gap regression and
    `pi_0` by exact Schur complements in `d = 1, 2, 3`.
  - xAI / Grok: a scope note on `3ac897a` (review 5379139737), "NOT READY". It was written while CI was pending, which has
    since gone all green, and with Slices A/B unread.
  - OpenAI has picked up a separate finite-torus transfer for every `L >= 10` (5932010316), in
    `frontiers/cusp_torus_transfer_20261001/`; it does not edit this branch.

  Claude reads count for nothing on this packet. **I will not merge.**

## Object

`CL-CU-CUSP-COEFFICIENT-20261001-v1` (v1.1) at `frontiers/cusp_coefficient_certified_20261001/`, plus the replay workflow
`.github/workflows/cusp-coefficient-certified.yml`.
- **Files:** `NOTE.md`, `README.md`, `certificate.py`, `cusp.py`, `d3.py`, `side24.py` (v1.1), `gauss.py`, `cx.py`,
  `elem.py`, `ia.py`, `consts.py`, `controls.py`, `RESULTS.json`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
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

**Corollary S (v1.1; NOTE §0, §3b).** For the [P] field on `R^d/(L Z^d)`, `L >= 24`, with covariance
`sum_n exp(-|z + L n|^2/2)`, normalized or not (SIDE24 is `L = 24`), `|c1^(L)/c1^(ref) - 1| <= eta_d`, with
`eta_2 <= 5.9e-105` and `eta_3 <= 1.4e-104`. Hence:

| `d` | `c1^(L)` | `c1^(24) / c_(d,24)` |
|---|---|---|
| `2` | `[-0.269398825674, -0.269398825672]` | `[-3.66993776909, -3.66993776907]` |
| `3` | `[-0.211848347, -0.211848346]` | `[-5.07106215, -5.07106213]` |

`c_(d,24)` is taken from `coefficients/side24_v1/ENCLOSURE.json` (pinned, consumed). `I^cand,(L)` is likewise in
`RESULTS.json`.

`RESULTS.json` holds the certificate's own 16-digit intervals; the tables round them further outward.

## Consequences

- **Every floating value in [CU] §8 (remark 3) is confirmed.** This covers `c1`, `c`, `c1/c`, `I^cand` and `I^cand - c1` in
  `d = 2, 3`. Each quoted value is the correct rounding of the certified midpoint.
- **The manuscript's correction ratio is certified (v1.1).** For the manuscript's field (`d = 3`, `L = 24`), the
  relative correction of (CU.1) is `(c1/c) ell^(7/12)`, with `c1/c_(3,24)` in `[-5.07106215, -5.07106213]`. [CU] §0
  quotes `c1/c ~ -5.071` for this field.
- **`d = 1` is a closed form.** `c1^(1) = -(384/7) 2^(1/4) 12^(-9/4) (2 pi)^-2 sqrt(4 pi/3) 30^(7/8) 2^(7/8)
  Gamma(11/8)/sqrt(pi)` agrees with Math-#214's `C_1` (`-0.227606`, #216's table).
- **`I^cand` is certified for Math-#218.** The certified `I^cand` intervals are the values of the `ell^(1/4)` coefficient
  of `nu_cand` that Math-#218 (Theorem T) carries: for the Gaussian kernel, and by Lemma S for the [P] field with
  `L >= 24`. #218 is consumed by nothing here.

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
3. **Lemma S (v1.1, the [P] torus field; `side24.py`, exact rationals).**
   - **S.1, image bound.** Every jet covariance entry is a contraction of `D^q K(0)` with `q <= 8` (`Cov(f4, f4)` has order
     8). Counting partial matchings gives `|D^q phi(x)| <= 764 |x|^8 phi(x)` for `|x| >= 1`, and `<= 105` at 0. Crude shell
     counts and `e^(-288) < 10^-125` then give
     `|D^q K_L(0) - D^q phi(0)| <= E = (764 . 24^8 . 4374 + 105 . 54) 10^-125 < 3.7e-108` for every `L >= 24`. The
     model is side24_v1 §2, which is stated to order 6 only.
   - **S.2, sandwich.** An exact `LDL^T` shows that `C_0 - I/4` is positive definite (`lambda_min` is about 0.258), so
     `(1 - eps) C_0 <= C_L <= (1 + eps) C_0` in every frame, with `eps = N E/(1/4)`.
   - **S.3, transfer.** Write `pi_0 E_0[F | b]` as the slice integral of the joint jet density, with `2d + 1`
     coordinates at 0. `F = |Y|^(7/4)|Delta|^(1/4) 1{A<0} >= 0` is homogeneous of degree `2d - 1/4`, so scaling the
     covariance by `a` scales the slice integral by `a^(-5/8)`. The Gaussian density sandwich then gives
     `|c1^(L)/c1^(ref) - 1| <= 8 (N/2 + 1) eps`.
   - **Cross-check.** The self-test rebuilds `C_0` exactly and reproduces Lemma R.1 in rational arithmetic: the
     regression, `det C_P = 12` and `(C_P^-1)_ff = 3/2`.
4. **Arithmetic.**
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
- **Lemma S, actual deviations (v1.1).** A direct floating lattice sum gives the largest `|D^g K_24(0) - D^g phi(0)|`
  over the `d = 3` covariance entries as `1.8e-114`, at `Cov(f4, f4)`. The certified `E = 3.7e-108` exceeds it by a
  factor of about `2e6`.

## What this does not do

- **Sides `L >= 24` only (v1.1).** For `L < 24` the premises of S.1 are not established, and no torus statement is
  made. `d = 1` is the Gaussian kernel only.
- **`c_(d,24)` is consumed.** The ratios `c1/c_(d,24)` take `c_(d,24)` from side24_v1's `ENCLOSURE.json`. That packet is
  author-side (OpenAI) with nonauthor review open, and it is not reviewed here.
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
    - the `t`-tail;
    - Lemma S's premises and the exact R.1 regression (v1.1).
  - The assembly also compares `integral g2` and `T3` with floating quadratures of the safe form, which guards the
    normalizations and factors.
- The workflow runs a matrix over `-B -S` and `-B -O -S`. Each job checks:
  - the manifest and the main-resident pins ([SIDE24] `PROOF.md`, [SIDE24-E] `ENCLOSURE.json`, [LP]; [CU] is pinned at
    `826b8be` but is not on `main`);
  - `--check`;
  - the six mutants (exit 1; `image-order6` is an image bound valid only to order 6, too low for `Cov(f4, f4)`);
  - a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis), NOTE §§1–2:**
  - R.1: the regressions, `pi_0` and the independence claims;
  - R.2: the Mellin constant `K`;
  - R.4: the `b`-integral, `beta(omega)` and the principal power in `Gamma(5/8)/(2 beta^(5/8))`;
  - R.5: the eigenvalue Jacobian, `adj(A)` in the eigenframe, the completion `S^2/20`, the coefficient `23/240` and the
    normalization Jacobians;
  - R.6: the safe-form identities for `Rt`, `It`, `g2`;
  - Lemma S (v1.1, NOTE §3b): S.1's matching bound and shell counts, S.2's sandwich, and S.3's slice-integral homogeneity
    (`a^(-5/8)`) and density sandwich.
- **Slice B (certificate), NOTE §§3–5 and the code:**
  - Lemma Q and the node certification (`gauss.py`);
  - Lemma A, including the mixed representations across slabs (`cx.py`, `cusp.py`);
  - Lemma B: the tensor error split, the positivity boxes, and the `t > 15`, `s > 3` and `d = 2` tails (`d3.py`,
    `certificate.py`);
  - the `elem.py` ranges, the Stirling remainder (`consts.py`) and the publication rounding;
  - `side24.py`: the exact `C_0`, the `LDL^T` certificate, `image_bound` and `lemma_r_check`.
- **Slice C (scope), NOTE §0 and §§7–9:**
  - the comparison with [CU] §8;
  - the `d = 1` identification with Math-#214;
  - the scope: the Gaussian kernel, and `L >= 24` for the [P] field;
  - the consumption of side24_v1's `c_(d,24)`;
  - the source pins.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
