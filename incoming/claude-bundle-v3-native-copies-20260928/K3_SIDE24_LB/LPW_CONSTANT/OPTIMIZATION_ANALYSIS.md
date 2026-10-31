# OPTIMIZATION_ANALYSIS.md — exponent ledger and coefficient structure of the LPW local event

**Record:** LPW-OPT-20260913-v1.0. **Lane:** H1, lower-constant optimization work order,
ANALYTIC first phase (light compute only; the promotion rungs own the cores).
**Status labels:** every claim is marked **Theorem** / **Lemma** / **Conjecture** /
**Heuristic** / **Estimate**. Theorems and lemmas are proved in-line here and mirrored
executable-exactly in `OPTIMIZATION_falsifier.py` (both modes byte-identical).
**Frozen carriers consumed (untouched; sha256 verified at consumption time):**
LPW-CAND-20260912 v1.0 (candidate, event construction §§2–8);
LPW_CONSTANT v1 `lpw_constant.py` `e258322c…4d3035`;
LPW_CONSTANT v2 `lpw_constant_v2.py` `ca654e39…30e125` + report `a469c456…61ef62`
(certified pair c = 260/(2082000000000·2⁴⁰·10²¹) ≥ 1.13e-43, r₀ = 1/2224640);
B1_TAXONOMY v1.0 (witness placement §7, coverage theorem §4);
D3_PERCOLATION (D3-SPLIT, D3-COUNT, remote Θ(r³) raw count 2.9274·(1+κ));
C1_ALPHA_INTENSITY (chart integral 1.6053e-4 ⟹ C\*_env = 1.284 at r = 0.05);
W6_REPORT (WP modulus E_WP(r) = 3.5e-3·r^{3/2} on (0, 0.05]).

---

## Q1 — THE EXPONENT LEDGER

### 1.1 Setup: the scaled field and the coefficient–unit dictionary (Lemma L1)

Under the six pins (f(M) = b, f(S) = b − r³/6, ∇f(M) = ∇f(S) = 0), the scaled field
F_r(x, y) = (f(rx, ry) − b)/r³ on the fixed chart D = [−2, 1/2]×[−1, 1] decomposes
(candidate §§5–8) as

    F_r = F_J + O(K r)  in C²(D),   F_J(x,y) = g(x) + A(x²−1/4)y + Bxy² + Cy² + D₃y³,

- g(x) = x³/3 − x/4 − 1/12, **forced** by the pins (the two values and two gradient
  zeros determine the axial cubic up to O(Kr); verified coefficientwise);
- the low-order coefficients (constant, x, x², y, xy) are **forced** by the pins and the
  endpoint gradient equations to their g-values within O(Kr) (candidate §8);
- the four **free** scaled coefficients are the J-coordinates in these units:

  | scaled coeff | physical coordinate | physical unit | scaled window cost |
  |---|---|---|---|
  | A | f_xxy(0)/2 | 1 | width w ⟹ physical volume w |
  | B | f_xyy(0)/2 | 1 | width w ⟹ physical volume w |
  | C | f_yy(0)/(2r) = q/(2r) | **r** | scaled width w ⟹ physical volume **r·w** |
  | D₃ | f_yyy(0)/6 | 1 | width w ⟹ physical volume w |

**Lemma L1 (exact).** A Θ(1) scaled window in C is a Θ(r) physical window in q; all
other free directions have unit 1. The y²-direction is the *unique* direction whose
scaled unit shrinks with r. (Proof: F_r's monomial coefficients are c_{ij} r^{i+j−3}
with c_{ij} the physical Taylor coefficient; i+j = 3 gives unit 1, and among the
order-2 coefficients x² and xy are pinned, leaving only y².) ∎

### 1.2 The general ledger (correcting the prompt's display)

For an event E contained in the failure event, with conditional J-density floor m_E,
weight floor w_E = inf_E W_r, and normalizer cap Z_r ≤ C_Z r²:

    Palm mass(E) ≥ m_E · vol(E) · w_E / (C_Z r²)   ⟹   exponent  e = v + w − n,

v = volume exponent, w = weight exponent, n = normalizer exponent. For d thin
directions at scales r^{α_i}: v = Σ α_i (Gaussian density Θ(1) on the box: cost of a
restriction of width r^{α_i} is r^{α_i}). **LPW: e = 1 + 4 − 2 = 3.**
(The prompt's "r^{2 + αk − 4}" is a sign slip; the consistent formula is αd + 4 − 2 =
2 + αd, which for α = k = 1 gives 3, matching the prompt's own r³.)

### 1.3 The weight exponent w = 4 is exact and forced (Lemma L2)

**Lemma L2.** On any event with A, B, C, D₃ = Θ(1): f_xx(M) = −r(1 + O(r)) and
f_xy(M) = −Ar(1 + O(r)) are **forced by the pins alone** (no event choice): the
endpoint gradient zeros plus the value drop r³/6 force f_x(rx)/r² = x² − 1/4 + O(r),
whence the axial second derivative at M is g″(−1/2)·r = −r and the mixed derivative is
−Ar. The yy-entries are r(2C ∓ B)(1 + O(r)) via the exact scaling H_f(rz) = r H_{F_r}(z)
(candidate §8). Hence |det H(M)|, |det H(S)| are Θ(r²) with scaled determinants
det H_F(m) = −(2C − B) − A² and det H_F(s) = (2C + B) − A², and W_r = Θ(r⁴) exactly.
The r⁴ is not a choice: the axial factors come from the pins, the yy factors from the
typing window (Lemma L3). Degrading the scaled dets toward 0 (event touching the typing
boundary) only *worsens* the ledger (w = 4 + κ). ∎

### 1.4 The typing window — upper edge of C (Lemma L3)

**Lemma L3 (typing/determinant edge).** For W_r > 0 the event must meet the typing
support: det H_F(m) > 0 with H_xx(m) = −1, and det H_F(s) < 0 with H_xx(s) = +1.
Exactly:

    C < (B − A²)/2   (m-edge),     C < (A² − B)/2   (s-edge).

For (A, B) near (2, 0) the m-edge binds: C < −2. Verified in the falsifier (F-TYPING). ∎

### 1.5 The climb bound — lower edge of C (Lemma L4)

**Lemma L4 (path-clearance/geometry edge).** Write x = −1/2 + ξ. Then
g(x) = −ξ²/2 + ξ³/3 exactly (verified coefficientwise). On the chart D, with
λ := −(C + 2|B| + |D₃|) > 0, for ξ ∈ [−3/2, 0):

    F(x, y) ≤ −ξ²/2 + ξ³/3 + |A||ξ²−ξ||y| − λy²
            ≤ −ξ²/2 + ξ³/3 + A²ξ²(1−ξ)²/(4λ),

and the right side is **strictly negative for every ξ ∈ [−3/2, 0) whenever
λ ≥ 25A²/8**, including the corner ξ = −3/2 (where the discarded ξ³/3 = −9/8 makes it
strict); the maximizing |y\*| = |A||ξ²−ξ|/(2λ) ≤ 3/10 < 1 stays inside the chart. The
right side ξ ∈ (0, 1] is dominated (25A²/8 > 3A²/2). Hence, if
C < −(25A²/8 + 2|B| + |D₃|) with A, B, D₃ bounded, the scaled field is negative
everywhere on D except at m itself: **no point of the chart has F > 0 — the above-b
target is unreachable inside the chart and the path witness dies.**
(Exact rational verification: 2000-point grid max −1121467/4750229335704 < 0, corner
exactly −9/8; F-CLIMB.) ∎

### 1.6 Theorem EXP-1 — α = 1 is forced, and the cubic is optimal within local witness events

**Theorem EXP-1.** Let E be an event of the six-pin law such that
(i) E ⊂ {D_f(M) ≠ S} is certified by a C²-robust path witness inside the scaled chart
(local event; any path exiting rD is A.rem's object by definition), and
(ii) E ⊂ TYP up to null sets (needed for W_r > 0).
Then, with A, B, D₃ confined to r-independent bounded intervals (required by the weight
floor and the clearance budget):

1. **C is confined to an r-independent window**
   −(25A²/8 + 2|B| + |D₃|) ≲ C < min((B−A²)/2, (A²−B)/2) (lower edge: Lemma L4,
   geometry; upper edge: Lemma L3, typing). The window is nonempty (the witness
   C = −5 ∈ (−12.5, −2)).
2. Therefore the physical q-window has width Θ(r) and no more: **α = 1 is forced**
   for the unique thin direction. A window of width r^α with α < 1 either includes
   C → −∞ (witness dies by L4: E ⊄ failure, inadmissible) or slides |C| → ∞
   (probability O(e^{−θ/r²}), worse than any power).
3. The volume exponent is v = 1 and cannot be 0 for an admissible event
   (v = 0 ⟺ Θ(1) physical q-window ⟺ unbounded scaled C ⟺ (i) or (ii) fails).
4. With w = 4 (Lemma L2) and n = 2 (normalizer, a property of the law: Z_r ≤ 4B₃r²,
   candidate §9), **e = 3 is the minimum exponent among admissible local witness
   events**, and it is achieved (LPW's E_r).

*Proof.* L1 gives the unit dictionary; L2 the weight; L3–L4 the two edges; the
inadmissibility dichotomy in (2) is exhaustive for α < 1; (3) restates that a
probability-Θ(1) admissible event would contradict L3+L4. ∎

**Answer to "which forces α = 1":** both, on opposite edges — the *typing/determinant*
side caps C above (Lemma L3) and the *event-geometry* side floors C below (Lemma L4).
Either alone makes the window Θ(1) scaled = Θ(r) physical. The determinant weight r⁴ is
then automatic (Lemma L2); it does not by itself bound C below (very negative C
*increases* |det|) — the lower edge is genuinely the path-clearance geometry.

### 1.7 The complete exponent table

| family | thin dirs | scales | v | w | n | e = v+w−n | status |
|---|---|---|---|---|---|---|---|
| LPW E_r (δ-box around F_*) | 1 (C) | r¹ | 1 | 4 | 2 | **3** | admissible, optimal (Thm EXP-1) |
| extra-thinned variant (A-window r^β) | 2 | r¹, r^β | 1+β | 4 | 2 | 3+β | admissible, strictly worse |
| widened C (α < 1) | 1 | r^α | — | — | — | **inadmissible** (witness dies, L4) |
| degenerate-det edge (C at typing boundary) | 1 | r¹ | 1 | 4+κ | 2 | 3+κ | admissible, strictly worse |
| slid window (\|C\| → ∞ as r → 0) | — | — | super-poly small | — | — | negligible (e^{−θ/r²}) |
| unrestricted C (v = 0) | 0 | — | — | — | — | **inadmissible** (⊄ failure or ⊄ TYP) |
| remote window saddles (A.rem, D3) | height window (s,b), width ℓ = r³/6 | — | 3 | 0 (RN → 1 at d ≥ 2r) | 0 | **3** | different mechanism, same exponent |
| WP / wrong-partner (W6) | — | — | — | — | — | ≤ 3/2 | larger mass as r → 0; upper-lane channel, irrelevant to the lower-bound coefficient |

**Conclusion (Theorem EXP-1 + table): no counter-family with exponent < 3 exists among
admissible local witness events; the cubic is the optimum of the local family.** The
remote mechanism independently sits at 3 through the height window, not a jet width.
*Boundary of the claim (labeled):* "no mechanism of any kind beats r³" is the upper
campaign's theorem-in-progress (C1/D3 raw carriers are both Θ(r³); B-classes are Stage C
open items) — not claimed here. No candidate counter-family was found; none is reported
for adversarial review.

---

## Q2 — THE COEFFICIENT QUESTION

### 2.1 Placement (frozen) and the lower-bound statement

**Theorem CONT-1.** The LPW local event G_r = E_r ∩ {M₄ ≤ K} satisfies
G_r ⊂ A.loc ⊂ A ⊂ F_r = {D_f(M) ≠ S}, exactly, on Reg ∩ TYP.
*Proof.* B1_TAXONOMY §7 (frozen): on G_r the path r·γ_\* ⊂ rD joins M to z ∈ rD with
f(z) > b and min f∘γ > s — exactly A.loc's definition (here executable-verified with
the explicit quantifiers k = 12, j = 2: 1/12 < 1/6 − 99/1280 = 343/3840 and
R(−2) = 9/16 ≥ 1/2; F-DISJ). A.loc ⊂ A by construction; A ⊂ F_r by T1 (τ_M > s). ∎

**Consequence (Theorem).** For all 0 < r ≤ r₀,
liminf_{r→0} (1 − q(r, 6/5))/r³ ≥ liminf P_r(G_r)/r³ ≥ c_v2 = 260/(2082000000000·2⁴⁰·10²¹)
≥ 1.13e-43 > 0 (v2's certified pair, re-certified by H1's v3). The LPW event is a
**certified lower bound on the failure coefficient**, not the coefficient itself.

### 2.2 Disjointness from the remote mechanism (Theorem DISJ-1)

**Theorem DISJ-1.** G_r ∩ A.rem = ∅, exactly.
*Proof.* A.rem := A ∖ A.loc (B1 §3, frozen, definitional). G_r ⊂ A.loc (Theorem
CONT-1). ∎  Supplementary geometric content: on G_r the M-lineage and a point z with
f(z) > b lie in *one* component of O^s ∩ rD, so all three D3-SPLIT alternatives fail
(E₁: the witness needs no counted saddle at all; E₃: the in-chart level-s connection is
the path itself). Note on scales: the witness path reaches distance ≈ 2.61r from S
(point r·(−2, 3/4)) — *outside* the 2r disk yet *inside* rD; the loc/rem split is by
path localizability in rD (measurable), not by the 2r distance, and D3's stitch assigns
rD ∪ B(pair, 2r) wholesale to the chart lane. Hence no double counting anywhere:
P(A) = P(A.loc) + P(A.rem) (disjoint), and D3-COUNT's raw-count split by saddle
location (rD vs T² ∖ rD) is a disjoint upper envelope over the *same* P(A) that the
LPW event lower-bounds. ∎

### 2.3 Add, interfere, or contained? — the precise answer

- **G_r and the remote A.rem: ADD (disjoint events).** Theorem DISJ-1. Neither contains
  the other; they are disjoint contributions to P(A), which is one of four disjoint
  classes in the exact identity 1 − q = P(A) + P(B1) + P(B2) + P(B4) (B1-COVER).
- **G_r and A.loc: CONTAINED, strictly.** G_r is one explicit Θ(r³)-positive tube inside
  the local preemption event. A.loc ∖ G_r is the near-field remainder *between* the tube
  and the remote zone — the third family the question asks about. It exists and is
  potentially much larger than the tube (the tube is a δ⁴-thin slice of an Θ(1)-sized
  witness region in (A, B, C, D₃) space; see §2.4, lever F-OPT-5). Its current certified
  upper envelope is C1's raw chart integral 1.6053e-4 = 1.284·r³ at r = 0.05 (rung
  evidence; window saddles in rD, qualification dropped, polarity-safe).
- **Interference: none.** The events are nested or disjoint (G_r ⊂ A.loc; A.loc ⊔ A.rem;
  A ⊔ B-classes). There is no overlap to subtract anywhere in the accounting.
- **Other channels:** W6's WP/wrong-partner modulus E_WP(r) = 3.5e-3·r^{3/2} (certified
  on (0, 0.05]) is a B-side channel at a *larger* order — negligible for the
  r³-coefficient as r → 0; B-class r³ content is Stage C's open upper-bound problem and
  does not touch the lower bound (the LPW event lives in A.loc).

So: **LPW's explicit local event is ONE positive contribution — a certified lower bound
on the actual coefficient, with the true coefficient above it and bounded (at the rung,
evidence grade) by the raw carriers** (local 1.284 + remote 2.9274·(1+κ)-class, plus the
D3 envelopes 19.6·r³ spine / 3.9·r³ evidence for the remote annuli).

### 2.4 The optimization question: how large a certified coefficient could the best local event give?

Current certified: c = 1.135776e-43 = 260·m·δ⁴/B₃ with m = 1e-21, δ = 1/1024,
B₃ = 2082000000000. The slack per factor (certified vs actual), ranked:

| lever | certified now | actual/sharp | gain | grade/cost |
|---|---|---|---|---|
| F-OPT-1: density floor on the *small* box E_r (exact quadratic-form max at the 16 vertices; QF convex) | m ≥ 1e-21 (global JBOX floor via \|j\|+\|μ\| triangle bound) | **Estimate:** floor ~ 1.117e-3 (QF_max ≈ 8.7280) | ×1.1e18 | LIGHT (minutes; exact moments already certified; covariance inverse certified) |
| F-OPT-2: widen δ toward the clearance budget (16δ ≤ 1/16 − 8Kr₀; as r₀ ↓ the remainder share → 0) | δ = 1/1024 | δ → 1/256 | ×256 exactly (δ⁴) | LIGHT (exact rational arithmetic) |
| F-OPT-3: weight floor refinement (F_* dets 6·14 = 84 vs robust 65; re-split the C² budget) | 65 | ≤ 84 | ×≤ 1.3 | LIGHT |
| F-OPT-4: two-sided normalizer (replace C_Z = 4B₃ by Z_r/r²) | C_Z = 8.328e12 (B₃ chain) | **Estimate:** Z_r/r² ≈ 3.2245 (D3 probe, r = 0.05, QMC evidence) | ×2.6e12 | **HEAVY — promotion-rung scope (G.7 two-sided normalizer); not light compute** |
| F-OPT-5: witness-region event (replace the δ-ball by the full Θ(1) witness region in coefficient space, certified region volume + density integral) | tube slice | true local coefficient | orders | **MODERATE–HEAVY** (finite-dimensional geometry + certified integration) |

**Estimate (Heuristic, from the falsifier's display lines):** levers 1+2 (light compute,
within existing certified objects) give c ≈ 3.25e-23; adding lever 4 (promotion-rung
scope) gives c ≈ 8.4e-11. Lever 5 would approach the true local coefficient.

**Conjecture (evidence-labeled, not proved):** the true local coefficient
lim (P_r(A.loc))/r³ is Θ(0.01–1). Evidence: historical C020 C\* ≈ 0.97 at (1.2, 0.05)
(numerical, non-authoritative); C1's raw chart envelope 1.284 (upper, polarity-safe,
qualification dropped); D3's corridor kill (on-axis needle dead to e^{−10^286} scale —
consistent with the local witness living exactly on the conditioned ridge inside rD).

### 2.5 Follow-up items (beyond light compute; handed to the promotion rungs)

1. **F-OPT-4**: certified two-sided normalizer Z_r = Θ(r²) (G.7-scope) — the single
   largest certified-coefficient lever (×2.6e12).
2. **F-OPT-5**: certified volume/density integral over the full witness region
   {(A,B,C,D₃): a clearing path exists} — finite-dimensional but needs certified
   integration machinery.
3. Certified recombination: any coefficient upgrade changes the optimal (c, r₀) split
   (δ vs 8Kr₀ budget) — re-run the v3-style assembly with the new floors.

---

## Executable falsifier and hashes

`OPTIMIZATION_falsifier.py` (this directory): exact-rational verification of every
Lemma/Theorem above (F-LEDGER, F-WITNESS, F-PATH, F-TYPING, F-CLIMB, F-DISJ, F-COEF)
plus labeled Estimate display lines for §2.4. Fail-closed ck() → SystemExit(1); no bare
asserts; deterministic; `python` and `python -O` byte-identical
(`opt_normal.txt` ≡ `opt_O.txt`). Kill semantics: any admissible local witness event
exhibited with volume exponent < 1, weight exponent < 4 on a Θ(1)-det region, or a
clearing path with C below the L4 threshold, kills Theorem EXP-1; any field on G_r
failing A.loc kills CONT-1/DISJ-1.

Run receipt (this delivery): both modes exit 0, output byte-identical; final line
`OPTIMIZATION-FALSIFIER PASS`. sha256 values: see `OPTIMIZATION_MANIFEST.sha256`.

END_BODY

Body sha256 (bytes through the line preceding END_BODY): fb7b5d1462924f93f0f3af9d0fa354fc378908943d77fb514e580e2d2377c381
