## Current disposition

- **Head:** `a4fdfe3` = v1.2.2. This is v1.2 (docstrings only) plus base merges (last `main` `0732008`) and two wording-only NOTE fixes:
  - v1.2.1 `0e69d25`: §0 "within 10%" → "within 10.1%" (C206-C-01);
  - v1.2.2 `a4fdfe3`: §5 "every pair" → "every pair in the proved domain `0 < X_0 < 2/9`, `0 < theta < 1`" (C206-A-01).

  `theorem_a.py`, `RESULTS.json`, the workflow and every constant are unchanged since v1.2. Author lane: Anthropic / Claude.
- **Nonauthor review coverage: complete, every finding resolved.** All reviews are same-account delegated AI reviews, so organizational-independence credit is zero; personal reading is PENDING.

  | Slice | Review | Verdict | Finding |
  |---|---|---|---|
  | A: NOTE §§1, 3–7 analytic layer | [5397500398](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397500398) | PASS_TECHNICAL_SCOPED / ACCEPT_SCOPED | C206-A-01, resolved ([5397542735](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397542735)) |
  | B, shared primitives | C59/C61, through the byte-identical `engine_e2.py` in #215/#227 | PASS | C59-DOC-01 and C61, applied in v1.2 |
  | B, PR-specific: 153 bands, last band, aggregation | [5397460049](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397460049) | PASS_TECHNICAL_SCOPED | none |
  | C: NOTE §0 and §§10–11 | [5397446329](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397446329) | PASS_TECHNICAL_SCOPED | C206-C-01, resolved ([5397474831](https://github.com/d6g8k5htny-coder/Math-/pull/206#pullrequestreview-5397474831)) |
  | v1.2 docstrings | C63 5387311588 | PASS_TECHNICAL_SCOPED | — |

- **CI on `a4fdfe3`:** `math-downstream-gates`, `downstream-replay` and `formal / formal-evidence` are green; the ~28-minute `certified-replay` is running.
- **Integration:** claimed by an OpenAI nonauthor integrator ([5962630591](https://github.com/d6g8k5htny-coder/Math-/pull/206#issuecomment-5962630591)) for a source-preserving restack onto current main with fresh checks. I will not touch the head during that claim, and **I will not merge.**
- **Scope unchanged:** C8 stays OPEN. This is packet coverage, not a catalog or status change.

## Object

`CL-C8-THEOREM-A-CONSTANTS-PLANAR-20260930-v1` at `frontiers/c8_theorem_a_constants_planar_20260930/`, plus the replay workflow `.github/workflows/c8-theorem-a-constants-planar.yml`.
- **Files:** `NOTE.md`, `theorem_a.py`, `RESULTS.json`, `README.md`, `SOURCE_MAP.json`, `SOURCE_FILES.json`.
- **Kind:** author-side theorem with explicit constants, certified by exact rational interval arithmetic (standard library only).
- **Scientific effect: NONE.** C8 stays OPEN. No STATUS, PROOF_INDEX, GRAPH or catalog edit.
- **Relation to other packets:** follows Math-#203 (the sharp cap coefficient) and Math-#195 (the floor on `[1/64, 1/2]`).

## Statement (NOTE.md §0)

**Theorem E.** Let `B = [0, 1]` and `K = [1/2, 2]`. For every torus side `L >= 10`, every orthonormal frame, every `b in B`, `k in K`, and every `0 < r <= r_*`:

    Q^W(G_r^c) <= C(r_*) r^3,      Z_r / r^2 >= z(r_*),

where `G_r` is the good event of [LP] §7 / [CAP] (1). The same holds for the reference kernel.

| `r_*` | `C(r_*)` | `C / c_G(0,2) <=` | `C r_*^3 <=` | `z(r_*)` |
|---|---|---|---|---|
| `1/4096` | `26776575319/5000 = 5355315.0638` | `1.005814` | `7.8e-5` | `8.995379` |
| `1/2048` | `53623024479/10000 = 5362302.4479` | `1.007127` | `6.3e-4` | `8.990755` |
| `1/1024` | `2151134031/400 = 5377835.0775` | `1.010044` | `5.1e-3` | `8.981500` |
| `1/512` | `54156592391/10000 = 5415659.2391` | `1.017148` | `0.0404` | `8.962953` |
| `1/256` | `93707999311/10000 = 9370799.9311` | `1.76` | `0.559` | `8.921798` |
| `1/128` | `349560452.9880` | `65.7` | `167` | `8.841791` |
| `1/64` | `1439739617.9923` | `270` | `5492` | `8.676320` |

**Theorem E' (v1.1).** The same holds at the single fixed-axis SIDE24 point `(b, k) = (6/5, 1/6)` (for every `L >= 10`, every frame, and the reference kernel):

| `r_*` | `C'(r_*)` | `C' / c_G(6/5, 1/6) <=` | `C' r_*^3 <=` | `z'(r_*)` |
|---|---|---|---|---|
| `1/4096` | `19661409/500 = 39322.8180` | `1.100348` | `5.8e-7` | `3.226690` |
| `1/2048` | `215461191/5000 = 43092.2382` | `1.205825` | `5.1e-6` | `3.222401` |
| `1/1024` | `1604824613/1250 = 1283859.6904` | `35.9` | `0.0012` | `3.213821` |
| `1/512` | `117316965.4911` | `3283` | `0.874` | `3.196649` |

## Consequences

- **Explicit [LP] Theorem A.** Since `1 - p_r <= Q^W(G_r^c)` ([LP] §8), every pair above is an explicit `(C, r_*)` for [LP] (1.1). Examples are `(5415659.2391, 1/512)` and `(9370799.9311, 1/256)` on the band. [LP] §16 lists such a pair among the items it does not supply.
- **The SIDE24 law.**
  - [CAP] (2)–(3) had `1 - p_elder(r) <= min(1, 2.4 × 10^23 r^3)`, which is nontrivial only below `r = 1.6 × 10^-8`.
  - Theorem E' gives `1 - p_elder(r) <= 39322.818 r^3` for `r <= 1/4096`, within 10% of the sharp cap coefficient `c_G(6/5, 1/6) = 35736.8` of Math-#203.
  - The bound stays below 1 up to `r = 1/512`.
- **Near-optimal for the cap route.**
  - Every `C` here is a *cap-route constant* (Math-#203 v1.2), hence `>= c_G` at every `(b, k)` it covers.
  - On the band, for `r_* <= 1/512`, the constants are within 0.6%–1.8% of that limit.
  - This says nothing about lower bounds for the constant of (1.1) itself; its true coefficient at `(0, 2)` is `alpha_1 + alpha_2 = 2.18`.
- **Normalizer floor.** `Z_r / r^2 >= 8.6763` on `(0, 1/64]` for `L >= 10`. Combined with Math-#195, this gives `Z_r / r^2 >= 3.5044` on `(0, 1/2]` for `L >= 12`.

## Proof outline (NOTE.md §§1–8)

The proof is self-contained apart from [LP]'s definitions and pins. It does not consume [LP] (4.1), (4.2), (5.4), (5.5) or (7.5)–(7.7).

1. **Exact coordinates.** The six pins ([LP] (3.1)) and twelve homogeneous targets. The identities (I1) give `W_r = r^2 P1 P2`.
2. **Lemma 1 (law).**
   - Covariances are exact series `sum h(g_x) h(g_y) pi_gamma r^(g_x - w)`. Regularity is checked for all 252 pairs, and the series is truncated at 60 with a tail bound.
   - The torus remainder is bounded by Cauchy estimates on the unit polydisc: `M_1(10) < 4.5e-15`, uniform over frames and `L >= 10`.
   - Interval regression on the pins, then an exact regression on `w`.
3. **Lemma 2.** Averaging identities: `U3 = 12k` is a Peano-kernel average of `f_xxx`, `-2v` an average of `f_xxy`, and `om` an average of `f_xyy`. They give `M_3 <= max(12k + rDelta_1, 2|v| + rDelta_2, |om| + rDelta_3, |Y| + rDelta_4)`.
4. **Lemma 3.** The field is entire. `M_4` is bounded through the Taylor series at the midpoint, and the means are bounded by the pin energy.
5. **Lemma 4.** Near and far root bounds after the regression on `w`.
6. **Main term.**
   - The integral over `lambda in [0, rU]` against the density bound `min(bar-p, p_0 + lambda L_p)`.
   - Closed-form incomplete Gaussian moments, with the `theta`-split and Hölder for the perturbations.
   - `(X_0, theta)` is chosen per box.
7. **Lemma 5 (floor).** `Z_r/(36k^2 r^2) >= E[lambda_+^2] - (explicit corrections)`.
8. **Rare branches.** `||W||_q P(E)^(1 - 1/q)` for `q in {2, 4, 8, 16}` (Gaussian tails plus Markov at `p = 32`).
9. **Sweep.** 153 `r`-bands by (28 boxes plus the SIDE24 point). The last band `[0, 2^-29]` uses a scaling argument.

## Controls (floating point, not part of the certificate)

`theorem_a.py --mc`, end to end from the definitions:

| `(r, b, k)` | `Q^W(G_r^c)/r^3` | certified |
|---|---|---|
| `(1/512, 0, 2)` | `4.840e6 ± 0.035e6` | `5415659.24` |
| `(1/1024, 0, 2)` | `5.192e6 ± 0.053e6` | `5377835.08` |

There is no control at the SIDE24 point: there `c_G r^3 < 5e-6`, beyond the reach of sampling.

## What this does not do

- Planar only.
- The band plus one SIDE24 point; no other `k < 1/2`.
- No lower bound for the constant of (1.1).
- `c_{B,K}` and `c_{d,L}` are not addressed.
- Not a review of [LP]'s proof or of [CAP]; [LP] §8 / [CAP]'s deterministic theorem is used as a statement for `1 - p_r`.
- C8 stays OPEN.
- Same GitHub account as every lane; zero organizational-independence credit. **I will not merge.**

## Verification

- `python3 -B -S theorem_a.py --check --procs N` (also `-B -O -S`, about one minute on four cores) replays exactly:
  - the parameters, 11 sampled bands and both `C` tables;
  - the layout of `RESULTS.json`;
  - `C > c_G(0, 2)` and `C' > c_G(6/5, 1/6)`.
- `--check-full` replays all 153 bands (about ten minutes on four cores).
- Mutants `no-delta`, `cap-half`, `no-zfloor-correction` and `no-om-term` exit 1.
- The workflow checks the manifest, the pins ([LP] `dfed3b8d`, [CAP] `0633aca3`), `--check-full` and `--check` in the two modes, the four mutants, and a clean tree.

## Review slices requested (OpenAI / xAI lanes)

- **Slice A (analysis):** NOTE §§1, 3–7. The identities (I1)–(I2), Lemmas 2–5, the cover `N ∪ F ∪ X ∪ A`, the main-term integration with `g`-measurable majorants and the two density bounds, the floor, and the Hölder rare branches.
- **Slice B (engine and certificate):** NOTE §2 and §§8–9, and `theorem_a.py`. The regular series and truncation, the torus Cauchy bound, the interval regression, the band and box sweep, the per-box minima over valid parameter choices, the last-band scaling argument, and rounding.
- **Slice C (scope and consequences):** NOTE §0 and §§10–11. Cap route versus (1.1), the `L >= 10`, all-frame and reference-kernel scope, Theorem E' as a statement about the [CAP] SIDE24 law, the comparisons with `c_G`, the `z_*` combination with Math-#195, and the non-claims.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_015wNj8LPTKXsaT68G3DgPPh
