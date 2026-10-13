import sys
from pathlib import Path
P = Path(sys.argv[1])
s = P.read_text()
E = []
def r(old, new):
    E.append((old, new))

# sources line
r("""- Project sources read at main `8977619e` (the cited main files are unchanged since `03acb7be`)
  and Math- `79cb314d`.""",
"""- Project sources read at main `8977619e` (the cited main files are unchanged from `03acb7be`
  through the current base `7588716d`) and Math- `79cb314d`.""")

# A8: In brief item 1
r("""1. **Folds.** When two zeros of the de Bruijn–Newman heat flow H_t merge, the antiderivative
   of H_t gains or loses exactly one persistence bar, and that bar follows the same cubic fold
   law as in this project (bar length ∝ gap³). This is ordinary fold geometry, not something
   special to zeta. We computed two actual collisions to about 40 digits. Both are at
   negative time (t ≈ −1.878 and t ≈ −0.385), and the fold law is accurate only close to the
   collision time. None of this says anything about RH.""",
"""1. **Folds.** When two zeros of the de Bruijn–Newman heat flow H_t merge at a nondegenerate
   double zero (H_zz ≠ 0), the antiderivative of H_t gains or loses exactly one persistence
   bar, and that bar follows the same cubic fold law as in this project (bar length ∝ gap³).
   This is ordinary fold geometry, not something special to zeta. We computed two actual
   collisions to about 40 digits. Both are at negative time (t ≈ −1.878 and t ≈ −0.385), and
   the fold law is accurate only close to the collision time. None of this says anything
   about RH.""")

# A4: In brief item 3
r("""   about RH. Zeta zeros (like eigenvalues of random unitary matrices, CUE) repel each other
   quadratically at short range, whereas the zeros in the project's Gaussian models repel
   linearly. With the same cubic fold, that one change moves the short-bar density from
   ℓ^(−1/3) to roughly ℓ⁰. Our numerics are decisive against −1/3 for zeta. They are
   consistent with 0""",
"""   about RH. Zeta zeros are conjectured (Montgomery; supported numerically by Odlyzko and by
   our gap slopes ≈ 3) to repel each other quadratically at short range, like eigenvalues of
   random unitary matrices (CUE), whereas the zeros in the project's Gaussian models repel
   linearly. With the same cubic fold, that one change moves the short-bar density from
   ℓ^(−1/3) to roughly ℓ⁰. Our numerics, at heights 10⁴–10⁶, are decisive against −1/3 for
   zeta. They are consistent with 0""")

# A9: committed logs for the trapezoid and tanh-sinh checks
r("""  at z ∈ {0, 10, 28.27, 40, 75, 150} (Gauss–Legendre), ≤ 10⁻⁴⁴ for z ≤ 250 (trapezoid rule)
  and about 10⁻⁴⁷ (tanh-sinh). The zeros of H_0 match 2·Im ρ_k for k = 1…41 to 30 digits.""",
"""  at z ∈ {0, 10, 28.27, 40, 75, 150} (Gauss–Legendre), ≤ 2·10⁻⁴⁴ for z ≤ 250 (trapezoid rule;
  `results/dbn/smoke_hflow.out`) and ≤ 1.3·10⁻⁵⁷ at z ∈ {0, 10, 28.27, 75} (tanh-sinh, 60 digits;
  `results/dbn/indep_ht_xi.out`). The zeros of H_0 match 2·Im ρ_k for k = 1…41 to 30 digits.""")

# A16 (i)
r("""  which is truncation-limited at about 1.5·10⁻⁴.""",
"""  which is truncation-limited at about 10⁻⁴.""")

# A16 (ii)
r("""  - De Bruijn's strip lemma: if all zeros of H_{t0} satisfy |Im z| ≤ y0, then Λ ≤ t0 + y0²/2.""",
"""  - De Bruijn's strip lemma: if all zeros of H_{t0} satisfy |Im z| ≤ y0, then Λ ≤ t0 + y0²/2
    [de Bruijn 1950; snippet].""")
r("""  - **Which results are kernel-generic.** De Bruijn's lemma and Newman's monotonicity hold for
    broad kernel classes. Ki–Kim–Lee""",
"""  - **Which results are kernel-generic.** De Bruijn's lemma and Newman's monotonicity hold for
    broad kernel classes [de Bruijn 1950; Newman 1976; snippet]. Ki–Kim–Lee""")

# A3 + A16 (iii) + Sturm tag: provenance lists
r("""  Montgomery 1973, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu 2020, and the
  historical table of Λ lower bounds including the −0.385 entry attributed to
  Norfolk–Ruttan–Varga 1992.
- *From memory, unchecked:* Riemann 1859, Jacobi (theta transformation), Titchmarsh, Mumford,""",
"""  Montgomery 1973, Rudnick–Sarnak 1996, Davenport–Heilbronn 1936, Newman–Wu 2020,
  Lagarias–Montague (arXiv:1106.4348, the integral of ξ; statement to be checked), and the
  historical table of Λ lower bounds including the −0.385 entry attributed to
  Norfolk–Ruttan–Varga 1992.
- *From memory, unchecked:* Riemann 1859, Hadamard and de la Vallée Poussin 1896, Jacobi (theta
  transformation), Sturm's zero-number property for the heat equation, Titchmarsh, Mumford,""")

# A2 (a): related open proposal in section 2
r("""  own README calls obsolete and wrong. **Do not cite it.**
""",
"""  own README calls obsolete and wrong. **Do not cite it.**
- **Related open proposal** (unclaimed when this packet was written).
  [main#307 comment 6116121839](https://github.com/d6g8k5htny-coder/main/issues/307#issuecomment-6116121839)
  (an OpenAI lane, 12 October 2026) proposes polynomial-first deterministic heat-flow fold
  controls with finite-interval H0 in `experiments/universality`. This packet does not take
  that scope; it touches only `experiments/zeta_fold_exploration/**`.
""")

# A2 (c): nondegeneracy in Remark 1
r("""   Here F''' = 2A, so k = |A|/6. Neither ζ nor the heat equation enters this relation.""",
"""   Here F''' = 2A, so k = |A|/6. Neither ζ nor the heat equation enters this relation.
   Proposition 1 needs H_zz ≠ 0 (A ≠ 0). A degenerate (triple) collision such as
   H = v₃ = x³ − 6xτ follows a different law, with critical-value gap r⁴/4 between adjacent
   zeros r apart, and is not covered.""")

# A3: Lagarias-Montague in section 4 item 3
r("""   average, β = (α+1)/m: w^{−1} for GUE-type repulsion (α = 2) and w^{−2/3} for the Gaussian
   case (α = 1).
""",
"""   average, β = (α+1)/m: w^{−1} for GUE-type repulsion (α = 2) and w^{−2/3} for the Gaussian
   case (α = 1).

   Prior work on the same antiderivative (up to scaling): Lagarias–Montague, arXiv:1106.4348,
   on the zeros of the integral of ξ [snippet; statement to be checked]. Its zeros are not the
   critical points of F used here, so the barcode question is distinct.
""")

# A8: Sturm / zero-number argument in 5.2
r("""- These counts confirm "first collision" because, as t decreases, real zeros can only merge
  and leave the axis (Proposition 1 gives no births in backward time), so the number of real
  zeros in a fixed window can only drop, except through zeros crossing the window edges.""",
"""- These counts confirm "first collision" because in s = −t, H solves the forward heat
  equation, and the number of sign changes on an interval is non-increasing apart from flux
  through the ends (Sturm / zero-number property [memory]). So real zeros in a fixed window
  cannot be created as t decreases, degenerate events included; Proposition 1 describes the
  nondegenerate case.""")

# A11 + A10 (iii): fold-law table
r("""| τ | pair 34/35: r²/(8τ) − 1 | pair 34/35: ℓ/(\\|A\\|r³/6) − 1 | pair 9/10: r²/(8τ) − 1 | pair 9/10: ℓ/(\\|A\\|r³/6) − 1 |""",
"""| τ | pair 34/35: r²/(8τ) − 1 | pair 34/35: lobe area/(\\|A\\|r³/6) − 1 | pair 9/10: r²/(8τ) − 1 | pair 9/10: lobe area/(\\|A\\|r³/6) − 1 |""")
r("""| 10⁻⁶ | −1.94·10⁻⁷ | +7.50·10⁻⁷ | — | — |""",
"""| 10⁻⁶ | −1.94·10⁻⁷ | +7.50·10⁻⁷ | −9.93·10⁻⁸ | +4.49·10⁻⁷ |""")
r("""| 10⁻¹² | −1.94·10⁻¹³ | +7.50·10⁻¹³ | −9.93·10⁻¹⁴ | +4.49·10⁻¹³ |
""",
"""| 10⁻¹² | −1.94·10⁻¹³ | +7.50·10⁻¹³ | −9.93·10⁻¹⁴ | +4.49·10⁻¹³ |

The lobe area |∫_{z₋}^{z₊} H_t| is the critical-value gap of F_t. Its bar pairing is verified only
near the collision (Proposition 1(c); persistence check at τ = 0.01), not at t = 0. A gap need
not be a finite bar: on [−3/2, 3/2], x³/3 − x has critical-value gap 4/3, but its only finite
H0 bar, born at the right end, has length 7/24.
""")

# A10 (ii), (i)
r("""- Over τ ≤ 10⁻⁴, both errors scale as τ^{1.000000}.""",
"""- Over τ ≤ 10⁻⁴, both errors scale as τ^{1.000000} to τ^{1.000001}.""")
r("""- Below t*, Im z / √(2(t* − t)) = 1.00097, 1.0000097 and 1.0000001 at t* − t = 10⁻², 10⁻⁴
  and 10⁻⁶.""",
"""- Below t*, Im z / √(2(t* − t)) = 1.00097, 1.0000097 and 1.0000001 at t* − t = 10⁻², 10⁻⁴
  and 10⁻⁶ (pair 34/35; pair 9/10: 1.00050, 1.0000050, 1.00000005).""")

# A12
r("""| ζ, t ∈ [10⁵, 10⁶] | −0.020 [−0.041, +0.001] | — | excludes −1/3 by ≈ 0.31; the upper end only just reaches 0 | — |""",
"""| ζ, t ∈ [10⁵, 10⁶] | −0.020 [−0.041, +0.001] | — | lower end 0.29 above −1/3; the upper end only just reaches 0 | — |""")
r("""**Reading of these numbers.** The ζ data are **decisive against −1/3** (a difference of about
0.33, far larger than any error source).""",
"""**Reading of these numbers.** The ζ data are **decisive against −1/3** (the point estimate is
0.31 above −1/3, far larger than any error source).""")

# A13
r("""density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.786, 0.911] with 2000-lobe
blocks; the two windows differ,""",
"""density were exactly flat. For ζ, t ∈ [10⁴, 10⁵] it is 0.844 [0.769, 0.910] with 20000-lobe
blocks (only 7 clusters; [0.786, 0.911] with 2000-lobe blocks); the two windows differ,""")

# A7
r("""  each side) close gaps sit in the lowest decile at 8.0–8.1%. So roughly half of the deficit
  below 10% is the definition and half is a real correlation
  (`results/stats/localamp_baseline.out`).""",
"""  each side) close gaps sit in the lowest decile at 8.0–8.1%. The far-neighbour proxy removes
  about 4 of the 6 points of deficit (10% → 4.0% becomes 10% → 8.0–8.1%). Roughly two-thirds of
  the deficit is the definition; the remaining ~2 points may be a real correlation
  (`results/stats/localamp_baseline.out`).""")

# A5: T-CUE question and what counts
r("""  F_U = ∫ Z_U on a fixed deterministic arc, with explicit end handling (bars born at the arc
  ends dropped), or on the full circle. Does the expected""",
"""  F_U = ∫ Z_U on the deterministic arc [0, 2π) (cut at angle 0), with bars born at the arc
  ends dropped (N even if a periodic Z_U is wanted). Does the expected""")
r("""  coefficient comparison**, recompute the CUE bars under exactly the T-CUE convention, or
  derive c_N for the cut-at-θ₁ convention used in `stats/cue.py`""",
"""  coefficient comparison**, recompute the CUE bars cut at angle 0 under exactly this
  convention, or derive c_N for the cut-at-θ₁ convention used in `stats/cue.py`""")

# A15
r("""- **Why it matters.** It would be the first natural non-Gaussian law in which the project's
  transfer theorem is discharged with α ≠ 1.""",
"""- **Why it matters.** To our knowledge (main `8977619e`, Math- `79cb314d`; open drafts not
  checked), it would be the first natural non-Gaussian law in which the project's transfer
  theorem is discharged with α ≠ 1.""")

# A6
r("""about 1.6·10⁷ zeros""", """about 1.9·10⁷ zeros""")

# A2 (a) again in 6.3 Proposition 1 bullet
r("""  Follow-up question: in the project's own scale or heat-flow families, does a PDE likewise
  force folds to be transversal?""",
"""  Follow-up question: in the project's own scale or heat-flow families, does a PDE likewise
  force folds to be transversal? Deterministic polynomial heat-flow fold controls with
  finite-interval H0 are the scope of the open proposal in main#307 comment 6116121839 (§2),
  which this packet does not take.""")

# A11: 6.3 last bullet
r("""- **Collision–bar correspondence at t = 0** (optional, lowest): check ℓ(0) ≈ (8√2/3)|A||t_c|^{3/2}
  against the backward collision time t_c for close pairs at moderate heights, using
  Polymath15's effective approximation of H_t. Illustration only; known in substance, and
  accurate only for very close pairs (the low collisions are off by 33–128% at t = 0).""",
"""- **Collision–lobe-area correspondence at t = 0** (optional, lowest): check the lobe area
  ℓ(0) ≈ (8√2/3)|A||t_c|^{3/2} against the backward collision time t_c for close pairs at
  moderate heights, using Polymath15's effective approximation of H_t; whether that lobe is a
  finite H0 bar must be checked separately (Proposition 1(c) covers only small τ).
  Illustration only; known in substance, and accurate only for very close pairs (at t = 0 the
  two low collisions are off by 19% and 73% in this form; `results/dbn/fold_analyze.out`).""")

# A1: section 7 RH-program bullet
r("""- **Any RH program.** The note's proposed route ends at showing that the zeta heat flow has no
  positive-time transition. That statement is equivalent to Λ ≤ 0, which is RH; it is not
  merely tied to RH.""",
"""- **Any RH program.** The note's proposed route ends at showing that H_t has no real-zero
  merger for t > 0. RH (Λ ≤ 0) implies that statement, and it is equivalent to RH if a positive
  Λ forces a real collision (`CLAIMS.md` 6.2); it is not merely tied to RH.""")

# A16 (iv)
r("""  jets); ζ needs a Mellin transform over L, which the project disclaims; Epstein zetas violate
  the RH analogue, so theta structure plus a functional equation selects nothing; and a formal""",
"""  jets); ζ needs a Mellin transform over L, which the project disclaims; some Epstein zetas
  (Z_4; binary forms with class number > 1) have zeros off their symmetry line, so theta
  structure plus a functional equation selects nothing; and a formal""")

# A2 (b): information loss
r("""  from the argument principle (Turing's method) already does this, and better; it is how RH
  is verified numerically.""",
"""  from the argument principle (Turing's method) already does this, and better; it is how RH
  is verified numerically. Real-axis barcodes alone discard this information: H = 1 and
  H = z² + 1 give antiderivatives z and z³/3 + z, both with no real critical point and the
  same barcode (no finite bar), yet the second has zeros ±i.""")

# A14 + R3: Pillow
r("""Pillow is optional and only rasterizes PNG previews (`stats/make_plots.py --png`).""",
"""Pillow is optional, pinned separately in `requirements-plot.txt` (as `experiments/periodic_h0`
pins matplotlib), and only rasterizes PNG previews (`pip install -r requirements-plot.txt`,
then `stats/make_plots.py --png`).""")

# R1 + R5: section 8 run list
r("""- `dbn/track.py 270` (~3 min) and `dbn/track.py 103 -40 0.02 low10` (minutes); then
  `dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35` and""",
"""- `dbn/track.py 270` (~3 min) and `dbn/track.py 103 -40 0.02 low10` (minutes), then
  `dbn/collision_table.py` (writes `results/dbn/collision_events.csv` from the two event logs);
  then `dbn/fold_check.py -0.3845611166 223.2757801856 pair34_35` and""")
r("""  `dbn/fold_analyze.py pair34_35 pair9_10` (also writes `fold_tau_r_ell.csv`).
""",
"""  `dbn/fold_analyze.py pair34_35 pair9_10` (also writes `fold_tau_r_ell.csv`).
- `dbn/track.py 103 -40 <DT> dt<DT>` for DT = 0.01, 0.02, 0.025, 0.04, 0.05 (minutes each),
  concatenated into `track_low10_dt_sensitivity.out`.
""")

# A9: list the xi run
r("""- `dbn/indep_ht.py 60 c910` (~2 min), `dbn/indep_ht.py 60 c3435` (~7 min),""",
"""- `dbn/indep_ht.py 60 xi` (seconds), `dbn/indep_ht.py 60 c910` (~2 min),
  `dbn/indep_ht.py 60 c3435` (~7 min),""")

# R4: thread pinning
r("""  oversubscription (without it `cue.py 128` ran more than ten times slower here).""",
"""  oversubscription (without it `cue.py 128` ran more than ten times slower here); the same
  setting also makes the regenerated samples byte-identical.""")

# R2 (+ R1): reproduction status list
r("""`track.py 270`, `track.py 103 -40 0.02 low10`, `fold_check` for both pairs (CSV and JSON;
the printed Newton residual differs because the seed differs), `fold_analyze` (including
`fold_tau_r_ell.csv`),""",
"""`track.py 270`, `track.py 103 -40 0.02 low10`, `collision_table`, `fold_check` for both pairs
(CSV, JSON and stdout), `fold_analyze` (including `fold_tau_r_ell.csv`),""")

# R4: reproduction status tail
r("""[`MANIFEST.json`](MANIFEST.json). Regenerated in a scratch copy of this packet, `gauss.npz`,
`cue12.npz` and `cue128.npz` were byte-identical to the recorded hashes; `zeta_A.npz` differed
only at relative 10⁻¹⁰ (floating-point summation order), below the Riemann–Siegel error, with
identical counts in `zeta_A_info.json`. Regeneration is therefore not guaranteed to be
byte-identical.""",
"""[`MANIFEST.json`](MANIFEST.json). Regenerated in a scratch copy with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, the seven raw samples other than the two Z_grid
files (not regenerated) — `zeta_A`, `zeta_B`, `gauss`, `cue12`, `cue128`, `rs_cheb_coeffs` and
`ecdf_samples` — were byte-identical to the recorded hashes, and `summary.json`, the ECDF CSVs,
`matched_tail.json`, `local_slopes.csv` and both SVGs were reproduced. An earlier regeneration
without thread pinning gave `zeta_A.npz` values differing at relative 10⁻¹⁰ (BLAS summation
order), with identical counts, so byte-identity is expected only with thread pinning.""")

# A3: review slice B
r("""- **B.** `CLAIMS.md` and the citations, ideally by a lane with working arXiv/journal access,
  confirming every item marked [snippet] or [memory].""",
"""- **B.** `CLAIMS.md` and the citations, ideally by a lane with working arXiv/journal access,
  confirming every item marked [snippet] or [memory], including the statement of
  Lagarias–Montague (arXiv:1106.4348).""")

bad = []
for old, new in E:
    c = s.count(old)
    if c != 1:
        bad.append((c, old[:90]))
        continue
    s = s.replace(old, new)
if bad:
    for c, o in bad:
        print("COUNT", c, "::", o)
    sys.exit(1)
P.write_text(s)
print("applied", len(E), "edits")
