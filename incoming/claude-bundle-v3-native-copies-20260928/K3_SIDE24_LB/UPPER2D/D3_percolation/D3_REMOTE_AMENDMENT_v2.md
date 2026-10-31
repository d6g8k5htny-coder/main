# D3 REMOTE AMENDMENT v2 — κ_far and the remote bracket certified at the R2 rung floor

Stage E repair R2/V1, amendment to the frozen D3_PERCOLATION.md (body sha256
`8e7fef6b…`, UNTOUCHED). Supersedes the v1 amendment (body `63d91cdd…`,
kept as historical). Finding repaired: v1 computed the relative κ pieces
against the QMC probe Z_r = 8.06118054e-3; at the R2-certified rung floor
    Z_lo = 7.7592917375327855e-3
(`H3_closure/H3_RUNG_FLOOR.md`, file sha256 `6347275d86c5…`, pinned and
parsed fail-closed by the engine; interval-enclosed two-sided
Z_{0.05} ∈ [7.7592917375327855e-3, 1.1468646473404396e-2], no Monte Carlo in
that certificate), the v1 assembly 0.65284336 becomes 0.677284905, breaching
the v1 round-up 0.66 — and the annulus crude spine I_ann carries the same
1/Z_r (its frozen display divided by the same QMC probe), so only the
floor-consistent bracket below is genuinely Monte-Carlo-free.
Engine: `d3_amend_v2.py` (fail-closed, deterministic, both modes
byte-identical: `ta2_normal.txt` ≡ `ta2_opt.txt`). Falsifier:
`d3_falsifier.py` (F5 operative gate + F6 mutation suite).

BEGIN_FROZEN_BODY

## 1. The per-piece Z-scaling audit (verbatim from the engine)

Every quantity in the remote bracket is written with its Z_r-dependence
explicit (Z-free numerators exact-kernel, mpmath dps=100, axis-worst station
(d=5, θ=0°), r = 0.05, v = window mid):

    κ_cross = N_cross/Z_r,   N_cross = Bcross/m_sad = 5.04277749e-3
              (Bcross the frozen τ-form 1-Lipschitz bound) — scales 1/Z exactly
    κ_pair  = N_pair/Z_r,    N_pair = ||h||₂·√χ²(p9||p6) = 1.953515362e-5
              (χ² = 1.944e-6 closed form) — scales 1/Z exactly
    κ_y     = 0.0243696      (W2/Bures coupling vs the closed form m_sad) —
              Z-INDEPENDENT: no Z_r anywhere in the piece
    R_pg    = 0.99999994, R_wm = 1.0002592 — Z-INDEPENDENT exact ratios
    I_ann   = Num_ann/Z_r,   Num_ann = 1.714838453e-5 (frozen D4 grid, cap
              form, recomputed WITHOUT the 1/Z_r) — scales 1/Z exactly;
              consistency ck: Num_ann/Z_probe/r³ = 17.01823642 vs the frozen
              display 17.0182368 (within the display's rounding) ✓

Assembled at the certified floor (endpoint variation of N_cross displayed,
±9e-8, absorbed in the round-up):

    κ_cross = 0.64990178, κ_pair = 0.0025176, κ_y = 0.0243696
    κ_far assembled = 0.677284905  →  **κ_far ≤ 0.68 CERTIFIED** (round-UP)
    [diagnostic display at the QMC probe: 0.652843363 — matches v1]

## 2. The certified floor-consistent bracket (assembly-consumable)

    B_remote = I_ann(Z_lo) + I_far ≤ 17.6804 + 2.5282637·(1+0.68)
             = 17.6804 + 4.247483056 = **21.9279·r³**
    (upper bounds rounded UP: I_ann(Z_lo)/r³ = 17.68036065 → 17.6804;
     I_far = 4.247483056; sum 21.92788306 → 21.9279)
    exact consumable form for H5/D1-v2.2:
        B_remote = 17.6804 + 2.5282637·(1+κ_far),  κ_far ≤ 0.68
    full remote-class statement:
        P_r(A.rem) ≤ I_hole + B_remote ≤ 1.284 + 21.9279 = 23.2119·r³
        [I_hole CONSUMES C1's chart envelope, as frozen].

## 3. Grade statement (repaired)

With the R2 interval consumed (pinned artifact `6347275d86c5…`), the
G.7-scope two-sided normalizer dependency AT THE RUNG r = 0.05 is
**DISCHARGED**: every 1/Z_r in the bracket divides the certified floor
Z_lo; the R2 certificate is interval-enclosed and Monte-Carlo-free; the
QMC probe in the engine is a diagnostic display only, consistency-ck'd
inside the R2 interval (ck: Z_lo < Z_probe < Z_hi). "Monte-Carlo-free" is
claimed ONLY against this certified denominator. **D3-LEMMA-RN-UNIF remains
NAMED** for the ZONE UNIFORMITY — a different item, untouched by R2; the v1
disposition stands (missing: the rigidity-decoupling lemma for τ(y) uniform
over {d ≥ 5}, and the certified interval-box Riemann sum for the annulus
crude-spine integral; closed at the rung: all station κ pieces, the exact
far-zone main term (576−25π)J(ℓ), and the monotone-decay exact-kernel
evidence).

## 4. Reconciliation table (each line's denominator)

    19.5465    κ=0 remote POINT DISPLAY (RN at QMC-central; not an upper
               bound) — rejected by the operative gate
    20.9       frozen THM display (same κ=0 reading, incl. hole) — superseded
    21.2153    v1 bracket (κ pieces + I_ann at the QMC probe denominator)
    21.2658    mixed form (floor κ's, probe-denominator I_ann)
               — both SUPERSEDED-BY-FLOOR, both rejected by the gate
    21.9279    v2 CERTIFIED, floor-consistent, Monte-Carlo-free

## 5. Falsifier v2 (operative gate + mutations)

F5: re-runs `d3_amend_v2.py` byte-identically and applies the OPERATIVE gate
    [21.27, 22.5]: accepts 21.9279; rejects 19.5465 (κ=0), 21.2153
    (QMC-denominator), 21.2658 (mixed denominators) — all below the edge.
    Gate-edge note: the R2-round prescription specified [21.24, 22.5], but
    the mixed form 21.2658 > 21.24, so that edge cannot reject it; 21.27 is
    the minimal edge satisfying all four required discriminations.
F6 mutation suite: MUT-V2-1 gate self-test (9 sub-floor/out-of-range
readings rejected, 4 certified readings accepted); MUT-V2-2 transcript
tamper through the v2 parse path (κ=0, QMC-denominator, mixed, and inflated
tampered transcripts all rejected); MUT-V2-3 QMC-denominator injection at
the floor pin (the engine's pinned floor literal mutated to the QMC probe
value → SystemExit with the R2 floor-drift failure before any computation);
MUT-V2-4 one-hex mutation of the R2-artifact hash pin → SystemExit with the
hash-pin failure. Both modes byte-identical (`tf_normal.txt` ≡ `tf_opt.txt`).

END_FROZEN_BODY
