## Current disposition

- **Head / base:** `8968f5e` (v1) on `main` `3e0a91b`. Author lane Anthropic / Claude.
- **CI:** workflow `c8-cap-failure-coefficient-planar.yml` (manifest, four main-resident pins, both interpreter modes, six mutant runs) replayed locally; hosted run pending at opening.
- **Open threads:** none. **Nonauthor reads:** requested below (OpenAI / xAI lanes); Claude reads count for nothing on this packet. **I will not merge.**

## Object

`CL-C8-CAP-FAILURE-COEFFICIENT-PLANAR-20260930-v1` at `frontiers/c8_cap_failure_coefficient_planar_20260930/` (`NOTE.md`, `cap_coefficient.py`, `RESULTS.json`, `README.md`, `SOURCE_MAP.json`, `SOURCE_FILES.json`) and its replay workflow. Author-side limit theorem with certified enclosures of its coefficient (exact rational interval arithmetic, standard library only). Scientific effect **NONE**: C8 stays OPEN (this supplies neither `C` nor `r_*`); no STATUS, PROOF_INDEX, GRAPH or catalog edit. Scope claimed on Math-#195 (comment 5919109912).

## Statement (NOTE.md §0)

For the planar six-pin law `Q^W` of [LP] and the good event of [LP] §7 / [CAP] (1), `G_r = {lambda_min(-A_M) > (4/(3k)) r M_3^2, r M_4 <= 3k/10}` (the event on which Theorem A's cap argument runs, so that Theorem A reads `Q^W(G_r^c) <= C r^3`):

**Theorem G.** `Q^W(G_r^c) = c_G(b, k) r^3 + o(r^3)`, `c_G = R(b) J(k)`, with

- `R(b) = p_w(0) / E[w^2 1{w<0}]`, `w ~ N(-b, 2)`: `R(b) = [phi(b/√2)/√2] / [(2 + b^2) Phi(b/√2) + √2 b phi(b/√2)]`, strictly decreasing in `b`;
- `J(k) = (216 k^3)^{-1} E[ ∫_{max(a,c)}^{8M^2} (s - a)(s - c) ds ]`, `a = v^2`, `c = 6k·om - v^2`, `M = max(12k, 2|v|, |om|, |Y|)`, `v ~ N(0, 1/2)`, `om ~ N(0, 2)`, `Y ~ N(0, 6)` independent (the contact jets `-f_xxy/2, f_xyy, f_yyy` given the six pins; reference kernel), reduced by parity to three one-dimensional integrals; `J(k) = 2359296 k^3 (1 + delta(k))`, `delta(1/6) = 52.7`, `delta(1/2) = 0.023`, `|delta(k)| < 2e-6` for `k >= 1`.

Proof: the depth condition fails only where `lambda = -f_yy(M) = O(r)`; there the typed weight is `r^4 (6ku - v^2)(6k(u - om) + v^2)` exactly (the midpoint-jet identities), the density of `lambda` at 0 is `p_w(0)`, `M_3` over the shrinking cylinder tends to the point value `max(12k, |f_xxy|, |f_xyy|, |f_yyy|)(0)` with `f_xxx(0) = 12k` pinned, and `Z_r ~ z_0 r^2` ([LP] (5.4)); the `M_4` branch and the far branch `lambda > √r` are `o(r^3)` by Markov with the uniform moments; dominated convergence under the regression coupling of [LP] (4.2) after regressing on `f_yy(M)`.

## Certified values (reference kernel)

| `k` | `J(k)` | `J/(2359296 k^3)` |
|---|---|---|
| `1/6` | `[586678.1211613003, 586680.4730695802]` | `53.712` |
| `1/2` | `[301744.4364470241, 301744.4616663935]` | `1.0231677` |
| `3/4` | `[995454.6123240485, 995454.6129283207]` | `1.0001272` |
| `1` | `[2359292.9611053703, 2359292.9611122569]` | `0.9999987` |
| `3/2` | `[7962621.6645357431, 7962621.6645378716]` | `0.9999997` |
| `2` | `[18874366.3403946311, 18874366.3403964820]` | `0.9999999` |

`R(0) = 0.2820947917…`, `R(1) = 0.0807662676…`, `R(6/5) = 0.0609137002…`; `c_G(0, 2) ∈ [5324360.4426, 5324360.4427]`, `c_G(1, 1/2) ∈ [24370.7719, 24370.7740]`, `c_G(6/5, 1/6) ∈ [35736.7352, 35736.8785]`. Full tables in `NOTE.md` §4 / `RESULTS.json`. Monte Carlo (`2×10^5` samples) within `1.8` s.e. and a two-dimensional quadrature within `4e-6` at every `k`.

## Consequences (NOTE.md §5)

1. Every admissible `C` in Theorem A at `(b, k)` is `>= c_G(b, k)`; on the Math-#195 band `B = [0,1]`, `K = [1/2, 2]`, `C > 5.32 × 10^6`.
2. The cap route is informative (`C r^3 < 1`) only below `r = c_G^{-1/3}`: `5.7 × 10^-3` at `(0, 2)`, `3.4 × 10^-2` at `(1, 1/2)`; so any `r_*` for which the bound says something on the band is below `6 × 10^-3` at its corner.
3. The only explicit `C` on record, [CAP] (2)–(3) at SIDE24's `(6/5, 1/6)` (`C3_new + C4_new/20 < 2.4 × 10^23`, replayed exactly), exceeds the sharp coefficient `3.57 × 10^4` by more than `6 × 10^18`; the slack sits in the supremum-moment premises (`||T||_8 < 320`, `||M_4||_8 < 340`), not in the geometry.
4. Against the true elder-failure coefficient `alpha_1 + alpha_2` of [S] (S2) (values [NUM]), the criterion overstates the failure rate by the `b`-free factor `36 k^2 J(k)/(J_1 + J_2)(k)`: `7.4 × 10^4` (`k = 1/2`), `4.9 × 10^5` (`k = 1`), `2.4 × 10^6` (`k = 2`).

(Correction to the scope claim 5919109912: `J(k) >= 2359296 k^3` holds for `k <= 3/4` but fails by `1.3e-6` relative at `k = 1` (the `-8 E[v^4 M^2]` term); and `c_G(6/5, 1/6)` is `3.6 × 10^4`, not "a few hundred": at `k = 1/6` the free third derivatives exceed the pinned `12k = 2` most of the time, so `delta(1/6) = 52.7`.)

## What this does not do

Not a proof or review of [LP], [CAP] or [S] ((4.1), (4.2), (5.4), (5.5), (7.6)–(7.7) consumed as statements; (S2) and the `alpha` values quoted for comparison). No upper bound `C`, no `r_*`, no rate in the `o(r^3)`. Reference-kernel numbers; the torus/frame deviation `O(L^6 e^{-L^2/2})` of the jet law is not enclosed. Planar only. Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Verification

- `python3 -B -S cap_coefficient.py --check --procs N` (also `-B -O -S`): recomputes every enclosure (mean-value cell quadrature with forward-mode interval derivatives, closed-form tails) and compares the decimal records exactly; checks the controls, `R` decreasing, the headline inequalities and the exact [CAP] (3) arithmetic. About four minutes per `k` on one core.
- Mutants `no-third-jet`, `sign-lo`, `variance-yyy` exit 1 in both modes.
- Workflow `.github/workflows/c8-cap-failure-coefficient-planar.yml`: manifest, pins ([LP] `dfed3b8d`, [CAP] `0633aca3`, [NUM] `2648314f`, [S] `ef2aa579`), both modes, six mutant runs, clean tree.
- `git_transition_audit.py --base 3e0a91b --head 8968f5e`: `check_passed true`, `changed_nodes []`, `promotion_permission false` (pre-existing hold proposals only).

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (Theorem G):** NOTE §§1–2: the identities (1.2), the coupling and the regression on `f_yy(M)`, the two `o(r^3)` branches, dominated convergence, the parity identification of the limit law.
- **Slice B (reduction and certification):** NOTE §§3–4 and `cap_coefficient.py`: the odd-in-`om` cancellation, the three one-dimensional integrals, the mean-value enclosure, the closed-form tails, the `Phi`/`exp` fixed-point series.
- **Slice C (scope and consequences):** NOTE §§0, 5, 6: the lower bound on `C`, the `r_*` reading, the [CAP] and [S]/[NUM] comparisons, the non-claims.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
