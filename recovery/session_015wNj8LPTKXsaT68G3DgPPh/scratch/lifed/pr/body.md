## Current disposition

- **Head / base:** `d748c2a` (v1.1) on `main` `3e0a91b`. Author lane: Anthropic / Claude. Scope claim: Math-#206 comment
  5924133127.
- **What it is:** [LP] Theorem B with explicit constants on the C8 band, in `d = 2` and `d = 3`:
  - the radial ledger to second order, `|A_r / A_0^ref - 1| <= a(R) r^2 + eps_d`, with explicit `a(R)` and a
    radius-independent remainder `eps_d` ([LP] assigns no rate to `A_r -> A_0`);
  - two-sided lifetime densities about the reference constant, to `O(ell^(1/3))` plus an additive `eps'_d ell^(-1/3)`;
  - the difference constant `C_{B,K}` of [LP] (1.2), with (12.1) and (12.3).

  [LP] §16 lists "a numerical r_* or C on a prescribed band" among the items it does not supply.
- **Local verification:**
  - `--check-full` replays all 274 cap bands, exactly, under `-B -O -S` (28 minutes on four cores);
  - `--check` passes under `-B -S` (about 4 minutes);
  - `RESULTS.json` was assembled from the certificate's own functions and the stored cap sweeps, and `--check-full`
    reproduces every entry and the canonical layout exactly;
  - all five mutants exit 1:
    - `no-om-term`, `no-bad-set` and `no-det` at the rate sweep;
    - `no-torus` at the Taylor-model law;
    - `cap-half` at cap band 0;
  - the manifest and pins verify (the workflow's own check, run locally);
  - transition audit: `check_passed true`, `changed_nodes []`.
- **CI:** all four checks are green on the head `d748c2a` (and were on `2fa7c63`):
  - `certified-replay` (`--check-full` and `--check` in both interpreter modes, the five mutants, manifest, pins and a clean
    tree; 31 minutes on `d748c2a`);
  - `downstream-replay`, `math-downstream-gates` and `formal / formal-evidence`.
- **Open threads:** none. The Codex P1 finding (no vanishing-remainder rate on the torus) was fixed in `d748c2a` and its
  thread resolved. **Nonauthor reads:** requested below; none yet. Claude reads count for nothing on this packet.
  **I will not merge.**

## Object

`CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1` at `frontiers/c8_lifetime_difference_constant_20261001/`, plus the replay
workflow `.github/workflows/c8-lifetime-difference-constant.yml`.
- **Files:** `NOTE.md`, `README.md`, `theorem_b.py`, `engine_e2.py`, `engine_e3.py`, `RESULTS.json`, `SOURCE_MAP.json`,
  `SOURCE_FILES.json`.
- **Kind:** author-side theorem with explicit constants, certified by exact rational interval arithmetic (standard library only).
- **Scientific effect: NONE.** C8 stays OPEN. No STATUS, PROOF_INDEX, GRAPH or catalog edit.
- **Relation to other packets.**
  - The certificates of Math-#206 (`0cb8ad3`) and Math-#212 (`c4b8ec3`) are used as libraries, as byte-identical copies
    `engine_e2.py` and `engine_e3.py`. They supply the band laws, contact moments and box bounds, and are pinned in
    `SOURCE_MAP.json`; the workflow checks the copies against the pins.
  - Cross-checks: Math-#197 and Math-#205 (`c_{B,K}`).
  - Comparisons: Math-#203 and Math-#208 (`c_G`).
  - Nothing under those packets is touched.

## Statement (NOTE.md §0)

**Theorem L (the radial ledger to second order).** Let `d in {2, 3}`, `B = [0, 1]` and `K = [1/2, 2]`. For every torus side
`L >= 10`, every frame, every `b in B`, `k in K`, every `R` in the table and every `0 < r <= R`:

    -(a_dn(R) r^2 + eps_d)  <=  A_r(b, k, u) / A_0^ref(b, k) - 1  <=  a_up(R) r^2 + eps_d.

Here `A_r = 12 pi_r(R; v_r) Z_r / r^2` is the ledger of [LP] (10.2), and `A_0^ref` is the closed-form contact integrand.
`eps_2 <= 3.96e-10` and `eps_3 <= 1.79e-8`; these collect the torus, truncation and rounding remainders and do not depend on
`r`. The same holds for the reference kernel.

| `R` | `a_up`, `d = 2` | `a_dn`, `d = 2` | `a_up`, `d = 3` | `a_dn`, `d = 3` |
|---|---|---|---|---|
| `1/4096` | `13.822900` | `3.637234` | `25.666574` | `25.516405` |
| `1/512` | `13.841872` | `3.680046` | `29.869053` | `29.730582` |
| `1/64` | `14.032636` | `4.025510` | `70.060400` | `69.981330` |

**Corollary L1 (densities to second order).** For `0 < ell < min(r_pop, R)^3 / 2`:

    | ell^(1/3) nu_cand(ell) - c^ref_{B,K} |  <=  c'(R) ell^(2/3) + eps'_d,
    ell^(1/3) nu_cand(ell) - C_{B,K}(R) ell  <=  ell^(1/3) nu_eld(ell)  <=  ell^(1/3) nu_cand(ell).

- With the `R = 1/4096` constants, `c'_up / c^ref` is `6.2826` in the plane and `23.154` in `d = 3` (rounded up).
- `eps'_d` is below `1.2e-13` in the plane and `2.5e-12` in `d = 3`.
- `c^ref_{B,K}` is enclosed in `[0.002386938000079198, 0.002386938059531751]` (`d = 2`) and
  `[0.000974382342936858, 0.000974382390951253]` (`d = 3`). These enclosures contain the values of Math-#197 and Math-#205.

**Corollary L2 (explicit (1.2), (12.1), (12.3)).** For `0 < ell, t < min(r_pop, r_*)^3 / 2` and `q > -5/3`:

    0 <= nu_cand(ell) - nu_eld(ell) <= C_{B,K}(r_*) ell^(2/3),
    0 <= E[N_cand(0, t] - N_eld(0, t]] <= (3/5) C_{B,K}(r_*) t^(5/3),
    E sum_{candidate, not selected, ell <= t} ell^q <= C_{B,K}(r_*) t^(q + 5/3) / (q + 5/3).

| `r_*` | `C_{B,K}`, `d = 2` | `C ell/c` at `r_*^3/2` | `C_{B,K}`, `d = 3` | `C ell/c` at `r_*^3/2` |
|---|---|---|---|---|
| `1/4096` | `402.5297` | `1.3e-6` | `1205.7089` | `9.1e-6` |
| `1/2048` | `406.7323` | `1.0e-5` | `1223.4762` | `7.4e-5` |
| `1/1024` | `413.9359` | `8.1e-5` | `1270.6952` | `6.1e-4` |
| `1/512` | `428.0180` | `6.7e-4` | `2488.0639` | `9.6e-3` |
| `1/256` | `3384.7882` | `4.3e-2` | `1807674.3925` | `56` |

## Consequences

- **The second-order coefficient of `A_r`.** [LP] (10.3) proves the convergence, and §11 says it "was not assigned a
  numerical rate".
  - Theorem L gives the `r^2` coefficient `a(R)` of the deviation from the reference contact integrand, up to the
    radius-independent remainder `eps_d`.
  - There is no first-order term: for the Gaussian kernel the transverse Hessian at `M` has exactly the contact law at every
    `r` (Lemma 1), and the first-order parts of `Z_r / r^2` cancel exactly (Lemma 3).
  - `eps_d` does not vanish, so this is **not** a rate for `A_r -> A_0` on the torus. It gives `|A_0 / A_0^ref - 1| <= eps_d`
    for the torus limit.
- **Two-sided densities.** Up to an additive `eps'_d ell^(-1/3)`, `nu_cand` and `nu_eld` are both
  `c^ref_{B,K} ell^(-1/3) + O(ell^(1/3))` with explicit constants. [LP] remarks that (1.2) is "not a claim that either density
  separately has a second-order expansion with that remainder". The remainder `eps'_d` does not vanish, so for the torus
  constant `c_{B,K}` this gives `|c_{B,K} - c^ref_{B,K}| <= eps'_d`, not a vanishing-remainder expansion. For example, in the
  plane, for `ell < 2^-19`:

      -4.84 ell^(2/3) - 10^-10  <=  ell^(1/3) nu_cand(ell) / c_{B,K} - 1  <=  6.65 ell^(2/3) + 10^-10.
- **`C_{B,K}`.**
  - On their own boxes, the [E] and [E3] per-band bounds give `2325.98` and `4999.57` at `r_* = 1/4096`. The refined `k`-boxes
    near `k = 1/2` gain factors `5.8` and `4.1`.
  - As `r_* -> 0` the cap route can give no less than `Gamma_{B,K} = |S| int A_0^ref c_G / (3 k^(5/3))`. In floating point,
    `Gamma_2 = 307.5` and `Gamma_3 = 541`, so `C_{B,K}(1/4096)` is within `1.31x` and `2.23x` of the cap-route limit.
  - By Math-#203, the cap criterion overstates the actual pairing failure by `7.4e4` at `k = 1/2`, so the true difference is
    far smaller than `C_{B,K} ell^(2/3)`.
- **Precision.**
  - In the plane `a_up(1/4096) = 13.823` is attained at `k = 2`, against a Monte Carlo coefficient of `13.2`–`13.4`.
  - In `d = 3` the maximum is attained near `(b, k) = (0, 1/2)`, where the Monte Carlo coefficient
    is only about `0`–`2`, because two Hölder bounds dominate there (NOTE §§9–10). At `k = 2` the box constants (`14.8`–`16.9`)
    are comparable to the Monte Carlo values (`11`–`17`).

## Proof outline (NOTE.md §§1–8)

1. **Consumed from [LP].** §§8–12 are used as statements: the ledger (10.1)–(10.2), the pushforward (11.1)–(11.2), and
   `1 - p_r <= Q^W(G_r^c)`. Theorem L is a statement about `A_r` alone, and the corollaries follow by integration.
2. **Taylor models in `r`.**
   - Each covariance is the exact series of [E] / [E3], plus the truncation tail and the polydisc torus remainder.
   - Each is kept on `[0, 1/64]` as a degree-8 polynomial with 288-bit fixed-point coefficients and an explicit remainder.
   - Gaussian elimination gives `det Sigma_r`, the pin energies, the regression coefficients of `A` and `Omega'`, and their
     covariances, with no interval dependence loss.
3. **Lemma 1.** The kernel factorizes, so `A + f(M) I` is independent of every pin. Hence `A = -b I + GOE_m` exactly, at every
   `r` and `k`. The certificate checks the Taylor coefficients to `10^-60`.
4. **Pin-density ratio.** `pi_r / pi_0^ref` comes from the Taylor models of `det` and the energies. The dominant term is
   `+3 k^2 r^2`, from `e_kk = 24 - 6 r^2 + ...`.
5. **Lemma 3 (exact expansion).** `P1 P2` is expanded exactly, using `c1 c2 = 36 k^2 + r^2 kappa` and
   `c1 e - c2 v^2 = r xi` (`c1 e' A* e - c2 v' A* v = r xi_3` in `d = 3`). The `Omega'` first-order term becomes second order
   through the regression of `Omega'` on the Hessian. In `d = 3`, four terms are handled through conditional expectations:
   the signed `beta`-term, `det Omega'` via `E[det X']`, the `zeta`-conditioned `Omega'` terms, and the signed `kappa`.
6. **Lemma 4 (thin shell).** The typed event differs from `{lambda > 0}` / `{B > 0}` only on `{0 < lambda_min <= r h}`.
   - The plane uses a Schur-complement density bound.
   - `d = 3` uses eigenvalue coordinates with a conditional-covariance floor `cl` and half-Gaussian moment polynomials.
   - The shell is `O(r^3)` relative.
7. **Reference integrals.** The `k`-integrals are incomplete gamma functions with exact integer roots. The `b`-integrals use the
   midpoint rule on 4096 cells with per-block `|g''|` bounds from `m`, `m'`, `m''`.
8. **Cap sweep.** The [E] and [E3] `box_bound` is run on all 153 / 121 bands and 56 refined boxes per band, with `k`-boxes of
   width `1/32` near `k = 1/2`.

## Controls (floating point, not part of the certificate)

`theorem_b.py --controls`. The coefficient `(A_r / A_0^ref - 1) / r^2` is estimated for the reference kernel:
- the law at `r` comes from the exact series;
- the pin ratio is exact;
- `Z_r` is sampled (seed `20261001`; `2e5` samples per point in the plane, `1e5` in `d = 3`).

Values are shown with one standard error; the last two rows are the certified `a_up` on the corner boxes at `R = 1/4096`:

| `d` | `r` | `(b, k) = (0, 1/2)` | `(0, 2)` | `(1, 1/2)` | `(1, 2)` |
|---|---|---|---|---|---|
| `2` | `1/8` | `0.08 ± 0.32` | `13.19 ± 0.39` | `0.64 ± 0.22` | `13.43 ± 0.26` |
| `2` | `1/16` | `0.16 ± 1.28` | `12.74 ± 1.34` | `1.09 ± 0.86` | `13.35 ± 0.90` |
| `3` | `1/8` | `2.24 ± 1.61` | `10.89 ± 1.75` | `0.84 ± 0.73` | `12.88 ± 0.87` |
| `3` | `1/16` | `-1.95 ± 5.71` | `17.29 ± 6.16` | `2.64 ± 2.99` | `17.45 ± 3.15` |
| `2` | certified box `a_up` | `4.818760` | `13.822900` | `3.918911` | `13.742115` |
| `3` | certified box `a_up` | `25.666574` | `16.876052` | `12.525134` | `14.762963` |

- **Cap-route limits** (floating): `Gamma_2 = 307.51` and `Gamma_3 = 540.97`.
- **Where `d = 3` is loose.** At `(0, 1/2)` the certified constant is `25.67`, of which `23.5` is the `Z`-part. That part is
  mostly the Hölder bounds of `r^2 E[D xi_3 1]` (`13.9`) and `r^2 E[(v' A* v)(e' A* e) 1]` (`8.7`), relative to
  `36 k^2 m_3`.

## What this does not do

- No vanishing-remainder rate on the torus: `eps_d` and `eps'_d` depend on neither `r` nor `ell`.
- `d = 2, 3` and the band `B x K` only. The unrestricted densities of [LP] Theorem C (with the constant `B_{d,L}` of Math-#191
  and Math-#211) are not addressed.
- The Kac–Rice interface of [LP] §§8–12 is consumed as statements, not reviewed.
- `C_{B,K}` goes through the cap route. No lower bound for the true difference coefficient is given.
- Near `k = 1/2` the rate constants are loose:
  - by about `3`–`5` in the plane;
  - by about `24` in `d = 3`, from the Hölder bounds of two terms (NOTE §10).

  This region carries most of the weight of `c'`.
- Not a review of [LP], [CAP], Math-#206 or Math-#212; the engines are used as certified functions.
- C8 stays OPEN. Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Verification

- `python3 -B -S theorem_b.py --check --procs N` replays exactly:
  - the Taylor-model laws (with the Lemma 1 check);
  - the rate sweep (96 boxes, seven `R`, both dimensions);
  - the reference integrals and the cross-checks against Math-#197 and Math-#205;
  - a sample of cap bands (the band ending at each `r_*`, at `2^-20`, and the last band);
  - the assembled tables and the canonical layout.
- `--check-full` replays all 274 cap bands.
- The mutants `no-om-term`, `no-bad-set`, `no-det`, `no-torus` and `cap-half` exit 1.
- The workflow checks:
  - the manifest;
  - the main-resident pins ([LP] `dfed3b8d`, [CAP] `0633aca3`);
  - the engine copies against `SOURCE_MAP.json`;
  - `--check-full` and `--check` in the two interpreter modes;
  - the five mutants;
  - a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis):** NOTE §§2–5.
  - Lemma 1 and its Taylor-model check.
  - The exact expansion of `P1 P2` and the three first-order cancellations.
  - The signed `kappa` and `beta` terms, and the conditional moments (`E[det X']`, `zeta`-conditioning) in `d = 3`.
  - The thin shell: the Schur-complement density in the plane; eigenvalue coordinates, `cl`, `KB'` and `J(m, x)` in `d = 3`.
- **Slice B (certificate):** NOTE §§2, 6–8 and `theorem_b.py`.
  - The fixed-point Taylor models: rounding, truncated powers, both tails and the torus remainder.
  - Elimination, `pi_ratio`, `combine` and `exp_up`, and the `(a, eps)` bookkeeping.
  - The incomplete-gamma and midpoint-rule enclosures.
  - The cap sweep, the assembly of `C_{B,K}` and the rounding.
- **Slice C (scope and consequences):** NOTE §0 and §§1, 9–11.
  - [LP] §§8–12 used as statements, and Corollaries L1–L2 by integration.
  - The `L >= 10`, all-frame and reference-kernel scope, and the `ell` ranges.
  - The comparisons (native boxes, `Gamma_{B,K}`, Math-#203), and the non-claims.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
