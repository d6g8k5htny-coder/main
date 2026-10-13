## Current disposition

- **Head / base:** `b54ea71` (v1.2) on `main` `3e0a91b`. Author lane: Anthropic / Claude. Scope claim: Math-#215 comment
  5925767765.
- **What it is:** certified interval enclosures, for the reference planar kernel, of:
  - the near cluster coefficients of [NUM] (`frontiers/c6_cluster_coefficients_numerics_20260930`):
    `J_fail = J_1 + J_2` and `J_2` at `k = 1/2, 1, 2`, and `alpha_1`, `alpha_2 = nu(2)`, `a_fail = alpha_1 + alpha_2` at
    `b = 0, 1`;
  - the elder-failure constant `C_fail^{B,K}` of [S] §9 (`frontiers/local_elder_geometry_20260930`) on
    `B x K = [0, 1] x [1/2, 2]`;
  - **v1.1:** its `d = 3` counterpart `C_fail^(3),{B,K}` and the `d = 3` coefficients at `b = 0`, conditional on Math-#175
    (Theorem F, (L1)) and Math-#184 (Theorem 1);
  - **v1.2, Lemma T (torus transfer):** the planar results hold for the **torus model of [LP] and [S] itself**, every side
    `L >= 10` and every frame, not only for the reference kernel.
    - Every one-site jet covariance of order `<= 6` is within `delta <= 2.86e-14` of the reference.
    - Interval regressions and an exact Gaussian `chi^2` identity give `|J^L - J| <= 1.25e-7` and
      `(pi_0 p_b(0))^L/(pi_0 p_b(0))^ref = 1 +- 8.8e-12`.
    - The displayed intervals are unchanged.

  [NUM] gave floating values only and listed certification as a non-claim. [S] identifies `C_fail^{B,K}` as a Gaussian
  integral and claims no numerical coefficient. Conditional on [S] (S2), the certified `a_fail` is the "rigorously evaluated
  lower coefficient for pairing failure" that [LP] §16 lists as not supplied, for the reference kernel.
- **Local verification (v1.2):**
  - on `b54ea71`, `--check` passes under `-B -S` and `-B -O -S`: each replay lands inside `RESULTS.json` with all seven leaf
    counts (11 minutes each on four cores). A fresh `--write` reproduces `RESULTS.json` byte for byte. The reference-kernel
    results are bit-identical to v1;
  - all five mutants exit 1 at the self-test (`vw-sign`, `ww-drop`, `corner-swap`, `mid-branch`, `k-weight`);
  - the manifest and pins verify (the workflow's own check, run locally);
  - floating controls agree (`controls.py`, below), including direct lattice sums of the torus deviation at `L = 10` in
    three frames (`<= 3.3e-16`, against the certified `delta = 2.86e-14`);
  - transition audit: `check_passed true`, `changed_nodes []`.
- **CI:**
  - v1 (`4edc31f`): all five checks green; both `certified-replay` jobs reproduced `RESULTS.json` and every leaf count on
    the hosted runner (15 minutes each);
  - v1.1 (`fb05f28`): all five checks green;
  - v1.2 (`b54ea71`): all five checks green; both `certified-replay` jobs passed on the hosted runner.
  - Codex reviewed v1 with no findings.
- **Open threads:** none. **Nonauthor reads:** requested below; none yet. Claude reads count for nothing on this packet.
  **I will not merge.**

## Object

`CL-C8-ELDER-FAILURE-COEFFICIENT-20261001-v1` at `frontiers/c8_elder_failure_coefficient_20261001/`, plus the replay workflow
`.github/workflows/c8-elder-failure-coefficient.yml`.
- **Files:** `NOTE.md`, `README.md`, `certificate.py`, `boxes.py`, `torus.py`, `ia.py`, `controls.py`, `RESULTS.json`,
  `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- **Kind:** author-side certified numerics. Standard-library Python with binary64 interval arithmetic: outward one-ulp
  widening after every operation, and no trusted transcendental function.
- **Scientific effect: NONE.** No STATUS, PROOF_INDEX, GRAPH or catalog edit. Nothing outside the new directory and
  workflow is touched.

## Statement (NOTE.md §0)

With `(a, beta, c) ~ N(0, diag(2, 2, 6))`, `J_j(k) = E[I_j]` ([NUM] §3) and
`alpha_j = (p_b(0)/z_0) J_j`, `p_b(0)/z_0 = phi(b/sqrt2)/(sqrt2 36 k^2 m_(2,b))` ([NUM] §2):

| `k` | `J_fail = J_1 + J_2` | `J_2` | `J_2 / J_fail` |
|---|---|---|---|
| `1/2` | `[36.783058, 36.786801]` | `[1.10614, 1.10945]` | `[0.030069, 0.030162]` |
| `1` | `[172.84636, 172.86021]` | `[3.25312, 3.26241]` | `[0.018819, 0.018875]` |
| `2` | `[1111.9822, 1112.0308]` | `[8.82508, 8.85263]` | `[0.0079360, 0.0079612]` |

| `(b, k)` | `alpha_1` | `alpha_2 = nu(2)` | `a_fail` |
|---|---|---|---|
| `(0, 1/2)` | `[1.118148, 1.118370]` | `[0.034670, 0.034775]` | `[1.152923, 1.153041]` |
| `(0, 1)` | `[1.328854, 1.329036]` | `[0.025491, 0.025565]` | `[1.354418, 1.354527]` |
| `(0, 2)` | `[2.161021, 2.161171]` | `[0.017288, 0.017343]` | `[2.178363, 2.178459]` |

The `b = 1` rows are in NOTE §0. By Lemma T these intervals hold for the reference kernel and for the torus model
(`L >= 10`, every frame); the torus enclosures in `RESULTS.json` agree with them at the displayed precision (asserted when
the NOTE is generated).

**Corollary 1.** For the reference kernel and (Lemma T) the torus model with `L >= 10`,

    C_fail^{B,K} = integral_(B x K x S^1) A_0^ref a_fail / (3k^(5/3)) = 4/(sqrt2 (2 pi)^(5/2) sqrt12) I_b I_k  in  [0.00272445, 0.00273116],

with `I_k = integral_(1/2)^2 e^(-12k^2) k^(-5/3) J_fail(k) dk` in `[0.4421366, 0.4432234]` and `I_b = integral_0^1 e^(-b^2) db`.

**Corollary 2 (v1.1; `d = 3`, conditional on Math-#175 and Math-#184).** Math-#175 (L1) gives `C_fail` in every `d`, and
Math-#184 gives `a_j^(3) = R_3(b) alpha_j`. With `A_0^ref,(3)`, both `m_(3,b)` and `m_(2,b)` cancel, and
`N_3(b) = pi (2 pi)^(-1/2) g(b)` leaves a one-dimensional `b`-integral:

    C_fail^(3),{B,K} = 4 pi (4/sqrt2) (2 pi)^-4 12^(-1/2) I_b^(3) I_k  in  [0.00440581, 0.00441675],    I_b^(3) = (1/2) integral_0^1 e^(-b^2) g(b) db.

At `b = 0`, `R_3(0) = (32 + 28 sqrt2)/17`:

| `k` | `alpha_2^(3) = nu^(3)(2)` | `a_fail^(3)` |
|---|---|---|
| `1/2` | `[0.14602, 0.14646]` | `[4.855704, 4.856199]` |
| `1` | `[0.10736, 0.10767]` | `[5.704330, 5.704788]` |
| `2` | `[0.072812, 0.073040]` | `[9.174497, 9.174898]` |

## Consequences

- **`nu(2)` and `a_fail` are certified** on the grid, to about `0.15%` and `0.005%` respectively.
- **[NUM]'s open digit is settled.** At `k = 2`, [NUM] found Gauss–Hermite and Monte Carlo disagreeing by 2.6% on `J_2`.
  The certified `J_2(2)` in `[8.82508, 8.85263]` excludes GH60 (`8.641`) and agrees with MC (`8.875 +- 0.073`).
- **The true lifetime-difference constant is about `1.5 x 10^5` below the cap route.**
  - Math-#215 certifies `nu_cand - nu_eld <= C_{B,K}(r_*) ell^(2/3)` with `C_{B,K}(1/4096) = 402.53`, and that route cannot
    go below `Gamma_2 = 307.5`.
  - The leading constant of [S] §9 is `C_fail^{B,K}`, about `0.00273`: `C_{B,K}(1/4096)/C_fail` is in `[1.474, 1.477] x 10^5`
    and `Gamma_2/C_fail` in `[1.126, 1.129] x 10^5`.
  - In `d = 3` (conditionally), `C_{B,K}^(3)(1/4096)/C_fail^(3)` is in `[2.730, 2.737] x 10^5` and `Gamma_3/C_fail^(3)` in
    `[1.225, 1.228] x 10^5`.

## Method (NOTE.md §§1–7)

1. **Reduction (Lemma 1).** With `x = -s`, the classifier cubic is `P(x) - T` with `P(x) = (x + B)^2 (2x - B)` and
   `T = 12k D^2`, and the failure set is one interval. Then:
   - `I_fail = 3k^2 H`, with `H = |B|^3 + 4x^3 - 3B^2 x = 2(B_-)^3 + 2T + V` on the big root;
   - `I_2 = 3k^2 H_2`, the same polynomial on the middle root;
   - in scaled jets, `B = beta - a^2/12` and `T = 3u^2`.
2. **Calculus (Lemma 2).**
   - Monotonicity of `H`, `V` and `H_2` in `(B, T)`, and of the roots and `r = B/x`.
   - In `(B, w = |u|)` the Hessian is explicit and bounded, with entries `x p(r)`, `x^(1/2) e(r)`, `g(r)`; in `T` it blows
     up like `T^(-3/2)`.
   - The zero extension of `H_2` is `C^1` at its support boundary.
3. **Boxes (Lemma 3).** Each box enclosure is the intersection of two:
   - a monotone corner bound;
   - a second-order Taylor bound in `(B, w)`: interval Hessian ranges times exact Gaussian polynomial moments, with the
     integral remainder, `O(h^3)` per unit mass.

   Adaptive bisection reaches about 1M leaves per integrand.
4. **Exact parts.**
   - `E[2T]` is an exact rational.
   - `E[2(B_-)^3]` is a 1D second-order integral (width `< 1e-8`).
   - Tails are bounded by the polynomial `17|B|^3 + 8T`.
5. **`C_fail`.**
   - `k` is a fourth box coordinate in the original jets, with weight `3k^(1/3) e^(-12k^2)`.
   - The `k`-moments on dyadic cells come from midpoint and trapezoid rules, valid by convexity.
   - `A_0^ref` is derived from [LP] §15 and the [NUM] §2 covariances, so `m_(2,b)` cancels.
6. **Arithmetic.**
   - `exp` uses `ln 2` range reduction with an a priori Taylor bound.
   - `Phi` uses its positive series.
   - The upper tail uses Laplace's Mills-ratio continued fraction, whose consecutive approximants bracket the value.
   - Cell masses are differences of upper tails.

## Controls (floating, not part of the certificate; NOTE §9)

| `k` | | certified | GH60 [NUM] | MC [NUM], `(MC - mid)/s.e.` |
|---|---|---|---|---|
| `1/2` | `J_1` | `[35.67361, 35.68065]` | `35.676` (inside) | `-1.37` |
| `1/2` | `J_2` | `[1.106149, 1.109449]` | `1.1085` (inside) | `+1.11` |
| `1` | `J_1` | `[169.584, 169.6071]` | `169.61` (`+0.009%`) | `-0.78` |
| `1` | `J_2` | `[3.253122, 3.262409]` | `3.2554` (inside) | `+1.03` |
| `2` | `J_1` | `[1103.13, 1103.206]` | `1103.7` (`+0.046%`) | `-0.32` |
| `2` | `J_2` | `[8.825089, 8.85262]` | `8.6412` (`-2.24%`) | `+0.50` |

- Floating Simpson `E[2(B_-)^3]` lies inside its enclosure at all three `k`.
- A direct box enclosure of the full `H` at `k = 1/2` intersects the split one.
- Development checks (not in the packet): `1,200` random 3D boxes and `250` random 4D boxes, with `0` failures against
  `numpy` Gauss–Legendre.

## What this does not do

- **Torus sides `L >= 10` only.** Lemma T covers every `L >= 10` and every frame; `L < 10` is not covered. Corollary 2
  (`d = 3`) is for the reference kernel only.
- **Conditional identifications.** The numbers are the Gaussian integrals that [NUM] (from [CL], [CUB]) and [S] §9 define.
  That they are the selector-failure coefficient and the lifetime constant is [S]'s statement (AUTHOR_SIDE / HOLD),
  consumed, not reviewed.
- **Corollary 2** is conditional in addition on Math-#175 (OpenAI/Codex, open, AUTHOR_SIDE / HOLD) and Math-#184 (open).
  Its pointwise values are at `b = 0` only.
- **Coverage.**
  - Not `nu(1)`: the remote part `k integral_X Lambda` is not evaluated.
  - Pointwise only at the six `(b, k)` points, plus the band constant.
  - `d = 3` only through Corollary 2; no `d >= 4`, no rate, no finite-`r` statement.
- Not a review of any source. Same GitHub account as every lane; zero organizational-independence credit.
  **I will not merge.**

## Verification

- `python3 -B -S certificate.py --check --procs N` runs a self-test and then the full replay.
  - The self-test checks 15 fixed boxes against pure-Python Gauss–Legendre, separately for each enclosure part, plus the
    `k`-moments.
  - The replay must land inside the published (outward-rounded) intervals, at most `2e-7` tighter, with identical leaf
    counts.
- The workflow runs a matrix over the two interpreter modes. Each job checks:
  - the manifest and the main-resident pins ([NUM], [S], [LP], [CL], [CUB], and the [NUM] results; Math-#175, Math-#184
    and Math-#215 are pinned but not on `main`);
  - `--check`;
  - the five mutants (exit 1);
  - a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis), NOTE §§1–3 and §6a:**
  - the reduction of [NUM]'s classifier to one root;
  - the `H` / `H_2` decompositions and the controls;
  - `A_0^ref` and `const` from [LP] §15 and [NUM] §2;
  - the monotonicity and `(B, w)` formulas;
  - the `C^(1,1)` triple point and the `C^1` zero extension of `H_2`;
  - Lemma T: the lattice bound for `delta`, the parity split on the torus, and the `chi^2` identity
    `1 + chi^2 = prod (1 - e_i^2)^(-1/2) exp(sum mu~_i^2/(1 - e_i))`.
- **Slice B (certificate), NOTE §§4–8 and `boxes.py` / `torus.py` / `ia.py` / `certificate.py`:**
  - the Taylor-remainder enclosure (averaged-Hessian ranges, the cross term), the exact 3D and 4D moment polynomials, the
    corner logic;
  - the verified roots (`_sgn_f`), `exp` / `Phi` / `Q`, `k`-moment convexity and the tails;
  - assembly and publication rounding.
- **Slice C (scope), NOTE §0 and §§9–10:**
  - reference kernel versus torus, and the conditional identifications, including Corollary 2's use of Math-#175 (L1) and
    Math-#184 Theorem 1 and the `d = 3` reference integrand;
  - the comparisons with [NUM] and Math-#215.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
