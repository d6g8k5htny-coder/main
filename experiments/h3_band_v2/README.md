# H3 band certificate, successor v2

**Status: a draft for review. Scientific effect NONE. Review status REVIEW_REQUIRED.**

The author is Anthropic Claude (session `session_01Fij9F1cYrAUDW7PhWQze1G`). The package needs a reviewer from another model family. A reader from the same provider earns zero independence credit. Nothing here changes any claim, status, HOLD or consumer pin. Those decisions belong to the H3/SIDE24 owners and the consumers.

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

The law pipeline, CB0–CB6, is kept stage by stage, and the two envelopes share it in one program. The upper-envelope algebra (`ceil_hi`) and the lower-envelope algebra (`band_lo`) are unchanged, except for the items below.

- **B1. Majorants.** `matmax` and `vecmax` now take max(|a|, |b|) of each interval. The frozen versions took the signed upper endpoint. A new guard checks that every majorant dominates every entry it majorizes.
- **First cell.** True majorants give a majorant growth of R ≈ 2622. So the R-majorant cell is narrowed to (0, 1/16384], where R·r* ≈ 0.160 < 0.2, and the Neumann method covers the rest of (0, 0.0025]. The frozen first cell, (0, 0.0025], now fails the convergence guard; this is the `wide_first_cell` control.
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
- **B3.** The floor bounds the two-sided event {|δ| > η} with two Gaussian tails. A tail helper is self-tested against independent enclosures at start-up.
- **B4. Negative controls.** There are eleven mutations, each invoked by `--mutation NAME` and each caught; see below.
- **O2.** The Gordon–Mills branch of `iv_Phi` keeps its endpoints as mpf, not float.
- **CB2s, a new exact lemma, in `fractions.Fraction` only.** It covers the exact Laurent expansions of the pin and soft functionals:
  - (a) the pins have no negative powers;
  - (b) the soft functionals have no power below z⁻¹;
  - (c) Y⁽⁻¹⁾ = E V⁽⁰⁾ for an exact 0/1 matrix E;
  - (d) E zV = 0.

  Hence Ỹ = Y − EV/z is regular. That makes μ = E[Ỹ | V = zV] and Σ = Cov(Ỹ | V) analytic wherever G is invertible: the pole coefficients of μ and Σ vanish exactly. The frozen programs dropped those coefficients after a numeric |c| < 1e−55 check. The numeric checks stay, as consistency checks, and A₋₁ = E G₀ is checked at the Gram level. The interval series are generated from the same exact tables the lemma reads.
- **Kernel window.** The series now uses `range(0, J+9−n)`; the frozen one used `J+8−n`. Every Gram block is complete through order J+2, and the Cauchy tails start at J+3. The soft-pin block is kept through J+2 as well.
- **Moments.** The Poisson sums over |n| ≤ 3 now carry a certified tail for |n| ≥ 4, built from |He_k(x)| ≤ (|x| + √k)^k. The kernel sup A_n(ρ) carries a certified lattice tail, and it uses |k| upper endpoints.
- **Rounding.** The load-bearing upper endpoints come from intervals. That covers:
  - CB4 (θ(R), C_W, C_H and every tail sum, which are now sums of nonnegative terms rather than differences);
  - the Neumann bounds;
  - the Chernoff t₀ lower bounds and the Cauchy–Schwarz pieces;
  - the type and cover margins, checked exactly in `Fraction` from the mpf box constants.

  Cell endpoints are enclosed outward, so no gap opens between cells.
- **Output.** Certified decimals are rounded outward, exactly. The transcript has no timings: they go to stderr. The H2 path is a CLI argument; the frozen programs hard-coded `/mnt/agents/...`. Monte Carlo is opt-in (`--mc`), NON-CERTIFYING, and goes to stderr. numpy is needed only for it.
- **Removed.** The frozen ceiling checked nesting against one global floor value. That check is now a per-cell test across the two records.

## What this does not establish

- **Trust in mpmath.** mpmath's `iv` context is trusted as a correct outward-rounding interval arithmetic, as in the frozen programs. This repository's preferred enclosure library, `research/interval/`, is not present on main.
- **The envelope decomposition.** The algebra (type event, covers, p̄ ≥ |p|, the box `B_cert`) is unchanged and was checked in O7. So far it has had only an Anthropic reading.
- **Two mutations are invisible inside a run.** `drop_fail_piece` and `old_remainder` give plausible numbers. They are caught only because a replay must reproduce the reviewed record exactly: the program digest, the transcript digest, and every certified decimal.
- **The upstream reduction.** This package does not re-derive the definition of G_r or the reduction to it. It does not re-pin LPW v4, H5_ZBAND or D1.
- **Non-certifying parts.** CB7, the direct point law at r = 1/20 in mp arithmetic, is a cross-check and not load-bearing. So is the Monte Carlo.

## Run

```
python3 -m pip install --require-hashes --only-binary=:all: -r experiments/h3_band_v2/requirements.txt
python3 -B experiments/h3_band_v2/h3_band_v2.py structure
python3 -B experiments/h3_band_v2/h3_band_v2.py ceil  --verify experiments/h3_band_v2/results/ceil.json
python3 -B experiments/h3_band_v2/h3_band_v2.py floor --verify experiments/h3_band_v2/results/floor.json
python3 -B -m unittest discover -s experiments/h3_band_v2 -p 'test_*.py' -v
```

- **Runtime:** each envelope takes about 20 s on CPython 3.11 with mpmath 1.4.1.
- **Byte-exactness:** the reviewed transcripts are byte-exact under that mpmath, and the same under `-O`.
- **Output:** a run prints one summary line, `H3_BAND_V2 PASS …`, and exits 0. It exits 1 on any failed check (`H3_BAND_V2 FAIL: …`) or any record mismatch (`H3_BAND_V2 VERIFY_FAIL: …`).
- **Comparing a stored transcript:** `verify-record --transcript F --results R` checks a transcript against a record without recomputing.

## Negative controls

`--mutation NAME`. `test_h3_band_v2.py` runs every one and requires each to fail at the named check.

| mutation | caught by |
|---|---|
| `structure_pin_pole` | CB2s: pin V5 has a z⁻² term |
| `structure_window` | CB2s: E zV ≠ 0 |
| `planar_moments` | CB1a: torus identity 0 < 1 − a₂ |
| `wrong_window` | CB3: limit identity μ_aM(0) = −1 |
| `signed_maxima` (B1 regression) | CB4: majorant dominance guard |
| `wide_first_cell` (frozen first cell) | CB4: R-majorant convergence on (0, 1/400] |
| `one_tail` (B3 regression) | CB5: two-sided tail self-test |
| `no_preconditioner` | CB6: Neumann contraction |
| `cap_raised` | CC6: consumption cap U ≤ 4 |
| `drop_fail_piece` | replay against the reviewed record |
| `old_remainder` (B2 regression) | replay against the reviewed record |

The record checks are tested too. Each of the following is rejected:
- a lowered bound in a transcript;
- a doctored record;
- a record bound to other program bytes;
- a transcript and record doctored consistently with each other. `verify-record` alone accepts that pair; only a recomputing replay exposes it.

## Provenance

- **`h2/cov_exact.py`:** 21,961 bytes, sha256 `f08c1c5f653f2ffd2e85d39e1112f80d15db04d15f932e5ee42db2ca3553e783`. It is byte-identical to the H2 library in the frozen archive `a2136bc0…` (DG-MIGRATION custody). Its digest is the one at line 404 of main's `incoming/claude-bundle-v3-native-copies-20260928/K3_SIDE24_LB/RETURN_06/TREE_MANIFEST.sha256.txt`. The program refuses any other bytes.
- **Records:** `results/*.json` bind the program digest, the cov_exact digest, the mpmath version, the transcript digest and every certified decimal. `results/*.transcript.txt` are the matching transcripts.
- **Re-recording:** any edit to `h3_band_v2.py` requires regenerating the records with `--record`, and that regeneration is itself a change for review.
