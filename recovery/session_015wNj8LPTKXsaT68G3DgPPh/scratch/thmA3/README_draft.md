# Explicit cap-route constants for Theorem A in dimension three (C8: certified `C`, `r_*`, `z_*` on a declared band, `d = 3`)

**Object:** `CL-C8-THEOREM-A-CONSTANTS-D3-20261001-v1`. **Author lane:** Anthropic / Claude. **Kind:** author-side theorem
with explicit constants, certified by exact rational interval arithmetic (standard library only). **Scientific effect:**
NONE: no register, catalog, GRAPH or STATUS change, and catalog entry C8 stays OPEN.

**Setting.** The eight-pin law of [LP] in `d = 3` with the good event `G_r = {lambda_1 > (4/(3k)) r M_3^2, r M_4 <= 3k/10}` of
[LP] section 7 / [CAP] (1) (partial-block operator norms on `D = [-2r, 2r] x ball(0, 2r)`). Theorem E3 (`NOTE.md`) holds on the
band `b in [0, 1]`, `k in [1/2, 2]`, uniformly over frames and torus sides `L >= 10`, and for the reference kernel:

    Q^W(G_r^c) <= C3(r_*) r^3   and   Z_r / r^2 >= z3(r_*)      for 0 < r <= r_*,

@@TABLE@@

**What follows.**
- **Explicit Theorem A in `d = 3`.** Since `1 - p_r <= Q^W(G_r^c)` ([LP] §8), [LP] Theorem A (1.1) holds in `d = 3` on this band
  with explicit pairs, for example @@PAIR@@. [LP] §16 lists such a pair among the items it does not supply.
- **Near-optimal for the cap route.** @@NEAROPT@@ The sharp cap coefficient `c_G^(3)(0, 2) = 22424320.655` of Math-#208 is a
  lower bound for every `d = 3` cap-route constant.
- **Normalizer floor.** `z3(1/4096) = @@Z4096@@`, against the contact value `9 m_(3,0) = 6.0442` at `(b, k) = (0, 1/2)`.

**Files.**
- `NOTE.md`: statement, the exact identities (I1)–(I3), Lemmas 1–5' (exact covariance series with a frame- and `L`-uniform torus
  remainder on the polydisc in `C^3`; regression on the transverse Hessian; blockwise averaging identities for `M_3`; Frobenius
  Taylor bounds for `M_4`; the root bounds; the normalizer floor through closed-form contact moments and a Loewner coupling),
  the main term in eigenvalue coordinates, the rare branches (angular nets for the `M_4` blocks), bands, values, Monte Carlo
  controls, and non-claims.
- `theorem_a3.py`: the certificate and its replay.
  - `--check --procs N` replays the parameters, 11 sampled bands and the `C` table exactly against `RESULTS.json`.
  - `--check-full` replays all 121 bands.
  - `--mutant {no-delta, cap-half, no-zfloor-correction, no-om-term, no-lambda2}` must exit 1.
  - With no flags it regenerates `RESULTS.json`.
  - `--mc R B K N SEED` is the floating Monte Carlo control.
- `RESULTS.json` (one line per band: 39 box bounds, 39 floors, the largest tail part), `SOURCE_MAP.json` (pins [LP] and [CAP]
  on `main`; read-only references to Math-#208 and Math-#206), `SOURCE_FILES.json`.
- Workflow `.github/workflows/c8-theorem-a-constants-d3.yml` replays the manifest, the pins, `--check-full` and `--check` in
  the two interpreter modes, the five mutants, and a clean tree.

Same GitHub account as every other lane; zero organizational-independence credit. The author will not merge.
