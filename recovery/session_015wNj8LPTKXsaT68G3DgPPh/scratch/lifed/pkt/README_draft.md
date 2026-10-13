# Theorem B of [LP] with explicit constants (C8; `d = 2, 3`): the radial ledger to second order, two-sided lifetime densities, and `C_{B,K}`

**Object:** `CL-C8-LIFETIME-DIFFERENCE-CONSTANT-20261001-v1`. **Author lane:** Anthropic / Claude. **Kind:** author-side theorem
with explicit constants, certified by exact rational interval arithmetic (standard library only). **Scientific effect:** NONE: no
register, catalog, GRAPH or STATUS change, and catalog entry C8 stays OPEN.

**Setting.** The compact-window lifetime densities of [LP] Theorem B on the band `b in [0, 1]`, `k in [1/2, 2]`, uniformly over
frames and torus sides `L >= 10` (and the reference kernel), through the radial ledger `A_r = 12 pi_r(R; v_r) Z_r / r^2` of [LP]
(10.2) and the pushforward (11.2), which are consumed as statements.

**Results** (`NOTE.md` Theorem L and Corollaries L1, L2):
- **The ledger to second order.** For `0 < r <= R`, `|A_r / A_0^ref - 1| <= a(R) r^2 + eps` with the closed-form contact
  integrand `A_0^ref`. In the plane `a_up(1/4096) = 13.823`, `a_dn = 3.637`, `eps <= 4.0 x 10^-10`; in `d = 3`
  `a_up(1/4096) = 25.67`, `eps <= 1.8 x 10^-8`. [LP] assigns no rate to `A_r -> A_0`.
- **Densities to second order.** `|ell^(1/3) nu_cand(ell) / c_{B,K} - 1| <= 6.29 ell^(2/3) + 5 x 10^-11` in the plane and
  `<= 23.2 ell^(2/3) + 3 x 10^-9` in `d = 3` (`R = 1/4096` constants), and the same for `nu_eld` up to `C_{B,K} ell / c_{B,K}`. So
  `nu_cand, nu_eld = c_{B,K} ell^(-1/3) + O(ell^(1/3))` with explicit constants.
- **The difference constant.** `0 <= nu_cand - nu_eld <= C_{B,K}(r_*) ell^(2/3)` for `ell < r_*^3/2`: @@README_CBK@@
- `c^ref_{B,K}` is certified in both dimensions; the enclosures contain the values of Math-#197 and Math-#205.

**Files.**
- `NOTE.md`: statement; the ledger and what is consumed; Lemma 1 (for the Gaussian kernel the transverse Hessian at `M` is
  exactly contact-distributed at every `r`); the Taylor-model laws; the pin-density ratio; the exact expansion of `P1 P2` and
  the first-order cancellation; the thin typed shell; reference integrals; the cap sweep; controls; non-claims.
- `theorem_b.py`: the certificate. `--check --procs N` replays the Taylor models, the rate sweep, the reference integrals, a
  sample of cap bands and the tables exactly against `RESULTS.json`; `--check-full` replays every cap band;
  `--mutant {no-om-term, no-bad-set, no-det, no-torus, cap-half}` must fail; `--controls` prints floating controls.
- `engine_e2.py`, `engine_e3.py`: byte-identical copies of the certificates of Math-#206 (`0cb8ad3`) and Math-#212
  (`c4b8ec3`), used as libraries (band laws, contact moments, box bounds); pinned in `SOURCE_MAP.json`.
- `RESULTS.json`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- Workflow `.github/workflows/c8-lifetime-difference-constant.yml`.

Same GitHub account as every other lane; zero organizational-independence credit. The author will not merge.
