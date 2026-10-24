# H3 band certificate, successor v2

**Status: a draft for review. Scientific effect NONE. Review status REVIEW_REQUIRED.**

The author is Anthropic Claude (session `session_01Fij9F1cYrAUDW7PhWQze1G`). The package needs a reviewer from another model family. A reader from the same provider earns zero independence credit. For that reviewer, [`REVIEW_GUIDE.md`](REVIEW_GUIDE.md) states and derives, with exact line references, the four claims that rest on a written derivation rather than on a run. Nothing here changes any claim, status, HOLD or consumer pin. Those decisions belong to the H3/SIDE24 owners and the consumers.

This is the numbered successor of two frozen programs, both authored by Kimi:
- `h3_band_ceil.py` (sha256 `b97c5428…`);
- `h3_band_floor.py` (sha256 `a907eeed…`).

Both live on main under `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/UPPER2D/H3_closure/` as `.py.txt`. They bound E[G_r] for r in (0, 0.05]. G_r is the normalizer of the LPW_CONSTANT v4 / OBL-D1-PROMOTE band, defined in the frozen `H3_BAND_CEIL.md` and `H3_BAND_FLOOR.md`. In brief:
- (q_M, q_S, a_M, a_S, b_M, b_S) is the conditional Gaussian law of the six soft variables, given the corrected pins of the exact side-24 periodized field (`h2/cov_exact.py`);
- D_M = a_M q_M − r b_M² and D_S = a_S q_S − r b_S²;
- G_r = −D_M D_S on the event {q_M < 0, D_M > 0, D_S < 0}.

This package does not re-derive that reduction. It certifies the expectation as the frozen programs define it.

The review that motivated this successor is O7 on [main #419](https://github.com/d6g8k5htny-coder/main/issues/419#issuecomment-6245249123). It found that the frozen programs do not certify cell 1, (0, 0.0025], or the published uniform constants.

## What the replay certifies, subject to review

Each row is certified for every r in the cell. The decimals are rounded outward exactly: the ceiling upward and the floor downward. They come from `results/ceil.json` and `results/floor.json`.

| cell | law method | ceiling E[G_r] ≤ | floor E[G_r] ≥ |
|---|---|---|---|
| (0, 1/16384] | R-majorant | 3.65718683240099 | 2.61755715030352 |
| (1/16384, 1/400] | Neumann | 3.66271691969455 | 2.58373305711774 |
| (1/400, 1/200] | Neumann | 3.66435028046303 | 2.58174158877405 |
| (1/200, 1/100] | Neumann | 3.67360595202420 | 2.54243388604045 |
| (1/100, 1/50] | Neumann | 3.69805856644731 | 2.45593199230241 |
| (1/50, 3/100] | Neumann | 3.71138266685765 | 2.44021698557766 |
| (3/100, 1/25] | Neumann | 3.72414747013089 | 2.41538548140069 |
| (1/25, 1/20] | Neumann | 3.73661723530033 | 2.28610115562878 |
| **uniform (0, 1/20]** | | **3.73661723530033** | **2.28610115562878** |

Compared with the frozen programs:
- **The ceiling over (0, 0.0025]** is now at most 3.66271691969455. That is below the frozen cell-1 value 3.66282864761194, which LPW v4 consumes. LPW v4's r₀ = 1/2278031360 lies in the first cell, whose ceiling is 3.65718683240099.
- **The uniform ceiling** is 3.73661723530033. The frozen value was 3.74767948915996. The rigorous remainder (B2) turns out tighter than the frozen correction, so the ceiling moves down.
- **The uniform floor** is 2.28610115562878. The frozen value was 2.30659559567154. The two-tail fix (B3) raises the δ piece in the last cell to 0.121, so the floor moves down. It stays far above c_Z = 1.6154892676….

Whether any consumer re-pins to these values is the consumer's decision.

## Changes from the frozen programs

The law pipeline, CB0–CB6, is kept stage by stage, and the two envelopes share it in one program. The upper-envelope algebra (`ceil_hi`) and the lower-envelope algebra (`band_lo`) are unchanged, except for the items below and two checks. `band_lo` no longer carries the frozen floor's check that Edelta and Ebeta exceed −1e−6, and both envelopes now require their second moments to be nonnegative before a square root is taken. Neither affects validity when the enclosures are valid.

- **B1. Majorants.** `matmax` and `vecmax` now take max(|a|, |b|) of each interval. The frozen versions took the signed upper endpoint. A new guard checks that every majorant dominates every entry it majorizes. For the unmutated code it holds by construction: it is a regression guard, and it is what catches `signed_maxima`.
- **First cell.** True majorants give a majorant growth of R ≈ 2622. So the R-majorant cell is narrowed to (0, 1/16384], where R·r* ≈ 0.160 < 0.2, and the Neumann method covers the rest of (0, 0.0025]. The 0.2 and the condition θ(R) ≤ 1/2 are margins kept from the frozen program: the induction itself needs θ(R) ≤ 1, and convergence needs R·r* < 1 and r*/ρ < 1, which the tail bounds check. The frozen first cell, (0, 0.0025], fails the guard and the majorant series genuinely diverges there (R/400 ≈ 6.6); this is the `wide_first_cell` control. On the first cell the pole coefficients of μ and Σ are dropped on CB2s alone, with no numeric fallback, because 1/t is unbounded there; their vanishing follows from CB2s (c) and (d) by direct algebra.
- **B2. Neumann remainder.** The correction now comes from the exact identities

      G (w − W_P) = (zV − G_P W_P) − (G − G_P) W_P
      G (G⁻¹ − H_P) = (I − G_P H_P) − (G − G_P) H_P

  together with the exact expansions of μ = A w and Σ = GY − A G⁻¹ Aᵀ. Here:
  - W_P and H_P are the truncated Taylor series of G⁻¹zV and G⁻¹, and G_P is the truncated pin Gram;
  - the residual polynomials zV − G_P W_P and I − G_P H_P are computed in full, through order 2(J+2);
  - so are the high orders of A_P W_P and of A_P H_P A_Pᵀ, through orders 2(J+2) and 3(J+2);
  - every Gram block has a Cauchy tail from order J+3;
  - ‖G(t)⁻¹‖ is bounded on each cell by a midpoint-preconditioned Neumann contraction.

  The frozen formula is kept only as the `old_remainder` control.
- **B3.** The floor bounds the two-sided event {|δ| > η} with two Gaussian tails. Every Gaussian tail in either envelope goes through one function, `event_prob_upper`, and the number of tails follows from the event kind. At start-up that function and the tail helper are self-tested on exact point laws, for N(0, 1) and N(1/2, 1), against Gordon's lower bound φ(t) t / (t² + 1). That bound does not use `iv_Phi`.
- **B4. Negative controls.** There are fourteen mutations, each invoked by `--mutation NAME` and each caught: twelve inside the run and two by the replay; see below. A mutated run never reports a pass: when it is not caught inside the run, it ends `H3_BAND_V2 MUTATED RUN …, NOT A CERTIFICATE` and exits 3. The same holds for `structure`: under a mutation the lemma cannot see, it ends `H3_BAND_V2 MUTATED STRUCTURE RUN …, NOT A CERTIFICATE` and exits 3. `--record` is refused under a mutation, and `--record` cannot be combined with `--verify`.
- **Fail-closed guards.** Four conditions that no current cell reaches now stop the run at a named check rather than at an mpmath assertion or a traceback: r < ρ in the Gram tail (`gram_tail`), a positive variance before each square root (`t0_lower`, `trunc_mom_iv`), and a window symmetric about 0 for the two-sided tail in `band_lo`. None changes a certified decimal, and the reviewed transcripts are byte-identical.
- **O2.** The Gordon–Mills branch of `iv_Phi` keeps its endpoints as mpf, not float.
- **CB2s, a new exact lemma, in `fractions.Fraction` only.** It covers the exact Laurent expansions of the pin and soft functionals:
  - (a) the pins have no negative powers;
  - (b) the soft functionals have no power below z⁻¹;
  - (c) Y⁽⁻¹⁾ = E V⁽⁰⁾ for an exact rational matrix E (for the tables of record E is 0/1);
  - (d) E zV = 0.

  Hence Ỹ = Y − EV/z is regular. That makes μ = E[Ỹ | V = zV] and Σ = Cov(Ỹ | V) analytic wherever G is invertible: the pole coefficients of μ and Σ vanish exactly. The frozen programs dropped those coefficients after a numeric |c| < 1e−55 check. The numeric checks stay, as consistency checks only: they see orders from −8 up, with a 1e−60 threshold, which could not by itself protect the remainder. Exact vanishing rests on CB2s. A₋₁ = E G₀ is checked at the Gram level. The lemma and the interval series are built from the same table objects, so a change to a table reaches both, and each of the four branches has its own control.
- **Kernel window.** The series now uses `range(0, J+9−n)`; the frozen one used `J+8−n`. Every Gram block is complete through order J+2, and the Cauchy tails start at J+3. The soft-pin block is kept through J+2 as well.
- **Moments.** The Poisson sums over |n| ≤ 3 now carry a certified tail for |n| ≥ 4, built from |He_k(x)| ≤ (|x| + √k)^k. New check CB1d: every consumed a₂ₘ, m ≤ 40, must meet an independent spectral enclosure of a₂ₘ = (1/Z) Σ e^{−k²/2} k^{2m}. It sums the lattice |j| ≤ 116, adds a certified tail to the numerator and divides by the truncated lattice sum Z_T. That still encloses a₂ₘ: every omitted |k| exceeds every retained one, so N_T/Z_T ≤ a₂ₘ ≤ (N_T + tail)/Z_T (scoped CB1 review 5607583647, §5). CB1d is a consistency check, since meeting an enclosure is not inclusion. The kernel sup A_n(ρ) carries a certified lattice tail too, and it uses |k| upper endpoints.
- **Rounding.** The load-bearing upper endpoints come from intervals. That covers:
  - CB4 (θ(R), C_W, C_H, the Cauchy ratio 1/ρ, which is rounded upward, and every tail sum, which is now a sum of nonnegative terms rather than a difference). A sanity check requires every computed Laurent coefficient, orders −2 to J+2, to obey its Cauchy bound |c_j| ≤ M ρ^{−j}. The Cauchy bounds the tails use, at higher orders, rest on the derivation;
  - the Neumann bounds;
  - the Chernoff t₀ lower bounds and the Cauchy–Schwarz pieces;
  - the type and cover margins, checked exactly in `Fraction` from the mpf box constants.

  Cell endpoints are enclosed outward, so no gap opens between cells.
- **H2 library.** CB0 executes exactly the bytes whose digest it checked; no import machinery or bytecode cache is involved.
- **Output.** Certified decimals are rounded outward, exactly. The transcript has no timings: they go to stderr. The H2 path is a CLI argument; the frozen programs hard-coded `/mnt/agents/...`. Monte Carlo is opt-in (`--mc`), NON-CERTIFYING, and goes to stderr. numpy is needed only for it.
- **Removed.** The frozen ceiling checked nesting against one global floor value. That check is now a per-cell test across the two records.

## What this does not establish

- **Trust in mpmath.** mpmath's `iv` context is trusted as a correct outward-rounding interval arithmetic, as in the frozen programs. This repository's preferred enclosure library, `research/interval/`, is not present on main.
- **The envelope decomposition.** The algebra (type event, covers, p̄ ≥ |p|, the box `B_cert`) is unchanged and was checked in O7. So far it has had only an Anthropic reading.
- **Two mutations are invisible inside a run.** `drop_fail_piece` and `old_remainder` give plausible numbers. They are caught only because a replay must reproduce the reviewed record exactly: the program digest, the transcript digest, and every certified decimal.
- **The upstream reduction.** This package does not re-derive the definition of G_r or the reduction to it. It does not re-pin LPW v4, H5_ZBAND or D1.
- **The B2 remainder has no numeric test.** CB7, the direct point law at r = 1/20 in mp arithmetic, checks the heavy polynomial, not the remainder. Its printed deviation, about 2e−91, is the rounding error of its own dps-100 reference law; against a dps-200 law the Neumann midpoint at r = 1/20 is within about 1e−100. Either figure is far below the rigorous corrections 1.5e−10 and 1.2e−7, so CB7 cannot tell whether the remainder is under-estimated. The remainder rests on the written derivation in `neumann_law`, set out in `REVIEW_GUIDE.md`. A same-family check, with zero independence credit, compared every cell enclosure against a dps-200 direct law at 32 points, and every enclosure contained it.
- **The B3 control covers the shared event bound.** It does not cover a future call site that bypasses `event_prob_upper`.
- **Planar moments.** For m ≤ 4, the torus and planar moments differ by about 1e−122, which is below dps 100. CB1d separates them from m = 16 up.
- **Non-certifying parts.** CB7 is a cross-check and not load-bearing. So is the Monte Carlo.

## Run

```
python3 -m pip install --require-hashes --only-binary=:all: -r experiments/h3_band_v2/requirements.txt
python3 -B experiments/h3_band_v2/h3_band_v2.py structure
python3 -B experiments/h3_band_v2/h3_band_v2.py ceil  --verify experiments/h3_band_v2/results/ceil.json
python3 -B experiments/h3_band_v2/h3_band_v2.py floor --verify experiments/h3_band_v2/results/floor.json
python3 -B -m unittest discover -s experiments/h3_band_v2 -p 'test_*.py' -v
```

- **Runtime:** each envelope takes about 20 s on CPython 3.11 with mpmath 1.4.1.
- **Byte-exactness:** the reviewed transcripts are byte-exact under that mpmath, and the same under `-O`. CI runs the whole suite, negative controls included, in both modes; under `-O` every subprocess the tests start runs with `-O` too.
- **Output:** a run prints one summary line, `H3_BAND_V2 PASS …`, and exits 0. It exits:
  - 1 on any failed check (`H3_BAND_V2 FAIL: …`) or any record mismatch (`H3_BAND_V2 VERIFY_FAIL: …`);
  - 2 on a refused combination of options;
  - 3 for a mutated run that completed, which is never a certificate.
- **Comparing a stored transcript:** `verify-record --transcript F --results R` checks a transcript against a record without recomputing. It needs only the standard library, and it does not compare the verifier's mpmath version with the record's; only a recomputing `--verify` replay does. It does require the record to name the pinned producer mpmath (`MPMATH_PIN`, the same version as `requirements.txt`), and it reads the mode from the transcript's first line rather than from the record.

## Negative controls

`--mutation NAME`. `test_h3_band_v2.py` runs every one and requires each to fail at the named check.

| mutation | caught by |
|---|---|
| `structure_pin_pole` | CB2s (a): pin V5 has a z⁻² term |
| `structure_soft_pole2` | CB2s (b): soft Y2 has a z⁻² pole |
| `structure_soft_span` | CB2s (c): the pole part of Y0 is outside the span of V⁽⁰⁾ |
| `structure_window` | CB2s (d): E zV ≠ 0 |
| `planar_moments` (planar values in the consumed table A2) | CB1d: spectral cross-check |
| `wrong_window` | CB3: limit identity μ_aM(0) = −1 |
| `signed_maxima` (B1 regression) | CB4: majorant dominance guard |
| `wide_first_cell` (frozen first cell) | CB4: R-majorant convergence on (0, 1/400] |
| `one_tail` (B3 regression in the shared event bound) | CB5: event-bound self-test, P(|X| > t) |
| `tail_helper_one_side` | CB5: tail-helper self-test |
| `no_preconditioner` | CB6: Neumann contraction |
| `cap_raised` | CC6: consumption cap U ≤ 4 |
| `drop_fail_piece` | replay against the reviewed record; the certified decimals change |
| `old_remainder` (B2 regression) | replay against the reviewed record; the certified decimals change |

The record checks are tested too. Each of the following is rejected:
- a lowered bound in a transcript;
- a doctored record;
- a record bound to other program bytes;
- a mutated transcript, or one without `ALL_CHECKS_PASS`, even with a record made consistent with it;
- a transcript and record doctored consistently with each other. `verify-record` alone accepts that pair; only a recomputing replay exposes it;
- a transcript whose only change is a check line that carries no certified decimal. Only the transcript digest binds those lines;
- a record naming a producer mpmath other than the pin, by `verify-record` and by a recomputing replay. `verify-record` does not depend on the verifier's own mpmath; only a recomputing replay requires the verifier to run the pinned version;
- a record relabelled to the other envelope, because the mode is read from the transcript's first line;
- a record that is not a JSON object.

## Provenance

- **`h2/cov_exact.py`:** 21,961 bytes, sha256 `f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783`. It is byte-identical to the H2 library in the frozen archive `a2136bc0…` (DG-MIGRATION custody). Its digest is the one at line 404 of main's `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/RETURN_06/TREE_MANIFEST.sha256.txt`. The program refuses any other bytes.
  - **Known library defect, not consumed here.** The library's interval evaluators (`e_sum`, `s_sum`, `k1_image`, and `k1`, `cov` and `moment_a` in `mode='iv'`) divide by the truncated spectral sum Z_T and pad only the numerator. Their error term omits |r_T|·(Z − Z_T)/Z, so at non-default cutoffs or higher precision their intervals can exclude the true value (review 5607640770). Even-order spectral moments at 0, as in CB1d, are still enclosed, by the argument above.
  - The program never calls those evaluators for a certified value. A2 and DEV2/DEV4 come from Poisson sums with their own denominator tails.
  - The program's only load-bearing use of Z_T is the CB4 majorant A_n(ρ), which divides a nonnegative sum by an interval whose lower end is at most Z_T, which is at most Z. That can only make the bound larger.
  - CB1d (above) and CB7 are checks only. The library itself is frozen and stays byte-identical. A repaired library would be a separate successor.
- **Records:** `results/*.json` bind the program digest, the cov_exact digest, the producer's mpmath version, which must equal the pin `MPMATH_PIN` (a recomputing replay also requires the verifier to run it), the transcript digest and every certified decimal. They also require an unmutated run that passed every check. `results/*.transcript.txt` are the matching transcripts.
- **Re-recording:** any edit to `h3_band_v2.py` requires regenerating the records with `--record`, and that regeneration is itself a change for review.
