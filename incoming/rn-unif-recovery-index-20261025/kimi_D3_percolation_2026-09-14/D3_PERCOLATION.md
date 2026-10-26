# D3 — PERCOLATION LANE: THE REMOTE PREEMPTION CLASS A.rem (OBL-B1-PERC)

U2D-UPPER Stage D, D3 lane. Frozen inputs (bodies, sha256, first 8 hex):
B1 taxonomy `7d7ddbec…`, C1 α-intensity `ca2c02b8…`, C2 B-classes `2aba099d…`,
coordinator foundation `LEAD_INTENSITY_DERIVATION.md` `729644f3…`; reference
inputs at their grades: DER-027a (lower-campaign variance channel,
Var(f | pins) ≥ 0.9795957 for d ≥ 3, monotone envelope, 7-pin family) and
DER-009 (selection lemma). Engine: `d3_perc.py` (fail-closed, deterministic,
both modes byte-identical); falsifier: `d3_falsifier.py` (discrete cover
mirror + engine re-run). Transcripts: `t_normal.txt` ≡ `t_opt.txt`,
`tf_normal.txt` ≡ `tf_opt.txt`.

BEGIN_FROZEN_BODY

## 1. The exact event form of A.rem

The taxonomy's measurable forms (B1 §2, frozen): with `O^u = {f > u}`,
`Conn(x, q, u)` the event that x and q lie in one component of `O^u`, and
`Conn_{rD}` the same inside the chart rectangle `rD = r·[-2, 1/2]×[-1, 1]`
(the taxonomy's local rectangle; D = [-2,1/2]×[-1,1], area 5, so |rD| = 5r²):

    A     : exists q ∈ 𝒬, k, j : Conn(M, q, s + 1/k) ∧ f(q) ≥ b + 1/j
            (M's level-s⁺ lineage reaches a point strictly above b = 6/5;
             s = b - ℓ, ℓ = r³/6; equivalently by T1(c), on Reg ∩ TYP:
             M's lineage dies at a window saddle S' ≠ S, f(S') ∈ (s, b),
             merging into a strictly elder component, birth > b)
    A.loc : the same with Conn replaced by Conn_{rD} (a window-saddle witness
            whose path M → q is localizable inside rD)
    A.rem := A ∖ A.loc                                    (taxonomy, frozen)

A.rem is measurable (countable Boolean combinations of open-set connectivity
events, exactly as B1 §2). B2d's torus audit confirmed that all remote/wrap
preemption in the discrete ensemble concentrates in A.rem (no third remote
class). The C021-corrected mechanism picture: the α carrier threads the
summit neighborhood and rides the conditioned ridge network out to a genuine
elder maximum — the remote contribution is structurally real (the discrete
ensemble: 453 A.rem pairs in the falsifier's audit; B2d's wrap ensemble:
3107/8801 of A-failures are remote).

### Lemma D3-SPLIT (proved here; discrete mirror certified in d3_falsifier.py F1)

On Reg ∩ TYP, write S' for M's death saddle (T1(c): index-1, f(S') ∈ (s,b),
one sector side containing M's component, the other a strictly elder
component whose highest maximum — the *elder terminal* — we call M₂,
f(M₂) > b). Then

    A.rem ⊆ E₁ ∪ E₂ ∪ E₃,

    E₁: S' ∉ rD                          (remote death saddle),
    E₂: S' ∈ rD, M₂ ∉ rD                 (local saddle, remote elder),
    E₃: S', M₂ ∈ rD but M, M₂ in distinct components of O^s ∩ rD
                                         (chart hole: the only level-s⁺
                                          route exits the chart).

*Proof.* On A.rem, T1(c) gives S' and M₂. If S' ∉ rD: E₁. If S' ∈ rD and
M₂ ∉ rD: E₂. If both are in rD: M and M₂ lie in one component of O^s (the
merge path through S'), so either they lie in one component of
O^s ∩ rD — but then that component is open in rD, path-connected, and a
compactness step gives a path with min f > s and a rational q near M₂ with
f(q) ≥ b + 1/j, i.e. A.loc, contradicting A.rem — or they do not, which is
E₃. ∎

### Corollary D3-COUNT (the assembly-primary split, no double counting)

Let N_w(𝒵) = #{S' ∈ Crit₁(T²) ∖ {S} : S' ∈ 𝒵, f(S') ∈ (s,b)} (raw window
saddles; no adjacency/elder qualification — dropping it is polarity-safe).
On Reg ∩ TYP, A ⊆ {N_w(T²) ≥ 1} (T1(c)), hence exactly

    P_r(A) ≤ E_{P_r}[N_w(T²)] = E_{P_r}[N_w(rD)] + E_{P_r}[N_w(T² ∖ rD)],

the two terms being the chart lane's (C1) and this lane's disjoint objects.
For the A.rem-specific statement:

    P_r(A.rem) ≤ E_{P_r}[N_w(T² ∖ rD)] + E_{P_r}[N_w(rD)]
               =: I_rem + I_hole,

with I_hole consuming C1's chart envelope (E₂ ∪ E₃ ⊆ {N_w(rD) ≥ 1}); the
hole term carries the extra Conn_{rD}-failure/elder-remote condition and is
priced strictly below the full chart integral once the branch/AO control
lands (flagged for the assembly). The DER-009 selection lemma is NOT needed
in this lane: Markov on counts is exact and no selection is performed.

*Discrete mirror (certified, F1):* on every A.rem pair of the falsifier's
ensembles (40 seeds close pairs + 6 seeds all pairs on the 40×40 PL torus
model + carved synthetics), exactly one of E₁/E₂/E₃ holds with the chart box
of B2d's instrument; the death saddle is always a raw window saddle
(PL-saddle, key(S) < key(S') < key(M)). Ensemble census: 453 A.rem pairs:
E₁ = 201, E₂ = 252, E₃ = 0; carved non-vacuity witnesses T-D3-REM (E₁) and
T-D3-RELDER (E₂) realize as designed. E₃ is not realized in the ensembles
(the union bound covers it if it occurs).

## 2. The probability mechanism

A.rem needs a level-s⁺ connection from M's component to a genuinely remote
elder structure. Three exact-kernel facts drive the bound:

(a) *Variance reversion (the conditioned field returns to unconditioned).*
Under the six pins, `Var(f(y)|pins) = 1 - c(y)ᵀ Σ_PP^{-1} c(y)` with kernel
entries c(y) ∝ poly(d)·e^{-d²/2}, d = dist(y, {M,S}). Probe table (§4):
0.190 (d=1.5 axis) / 0.657 (transverse) → 0.9787 (d=3 axis), 0.9988 (d=3
transverse) → 1 − 4.1e-8 (d=5 axis). DER-027a's lower-campaign floor
(Var ≥ 0.9795957 for d ≥ 3, 7-pin family, monotone envelope) is displayed
against our 6-pin values as a consistency check only — same reversion scale,
different pin family.

(b) *Subcritical superlevel structure at b = 6/5.* At positive levels the
Bargmann–Fock excursion sets are subcritical (clusters of {f ≥ s} small;
connectivity decaying with distance/level). The precise upgrade input:

    [CONSUMES: PERC-DECAY] there exist c_perc, C_perc > 0 with
    P( a fixed level-s⁺ cluster of the near-unconditioned field reaches
       distance d ) ≤ C_perc e^{-c_perc d}   uniformly in r ≤ r₀,

needed ONLY for the o(r³) upgrade of the A.rem bound (it multiplies the
qualification/AO factor), NOT for the O(r³) theorem: the raw remote count is
already Θ(r³) because the window is thin (ℓ = r³/6) — see (c).

(c) *The thin window.* The witness saddles counted have heights in
(s, b) = (b − r³/6, b): the unconditioned saddle density at height b is
ρ_sad⁰(b) = 0.030493491569 (§3 closed form), so the raw remote count is
(Area)·(ℓ)·ρ_sad⁰(b)·(1+o(1)) = Θ(r³) — no percolation input needed for the
O(r³) grade.

## 3. The exact intensity under P_r and the RN factor at distance

Per the coordinator's (L1)/(L2), with the exact height integral retained, for
any zone 𝒵:

    E_{P_r}[N_w(𝒵)] = ∫_𝒵 (1/Z_r) p_grad(y) ∫_{b-ℓ}^{b} φ(v; μ_t(y), s_t(y))
                       g(y, v) dv dy,
    g(y,v) = E_{Q_r}[ W_r · |det H_y| · 1{y saddle} | ∇f(y)=0, f(y)=v ],
    W_r = |det H_M det H_S|·1{M max}·1{S saddle},  Z_r = E_{Q_r} W_r.

### 3.1 The RN factorization (the weighted law's approach to the unweighted)

Under the 9-pin law (six pins + ∇f(y)=0, f(y)=v), regress the y-Hessian on
the pair Hessians: H_y = μ_y + B(H_pair − μ_pair) + η, η independent of
H_pair, B = Δᵀ Σ_pair^{-1}, Δ = Cov(H_pair, H_y | 9 pins). The saddle-typed
determinant is the negative part u(x) = (−det x)₊ — 1-Lipschitz in det — so
with ||det x − det x'|| ≤ (||x||+||x'||)||x−x'||:

    |g(y,v) − Z_r^{yv}·m_y^{yv}(v)| ≤ B_cross(y,v)
      := √3 (E W_r⁴)^{1/4} (E q²)^{1/4} √τ(y),   q = ||H_y||+||H_y⁰||,
      τ(y) = tr(Δᵀ Σ_pair^{-1} Δ)   [the pair→y variance transfer],

all factors exact Wick moments (E[det^k], k ≤ 8, by Stein recursion on the
9-pin Gram; certified). Here Z_r^{yv} (resp. m_y^{yv}) is the pair (resp. y)
typed moment under its 9-pin marginal. Therefore, exactly,

    ρ^{P_r}(y) = ρ⁰(y) · [p_grad/p_grad⁰](y) · [wm/wm⁰](y)
                      · [m_y^{yv}/m_sad](y) · [Z_r^{yv}/Z_r] · (1 ± κ_cross),

with κ_cross = B_cross/(Z_r m_sad). Every bracket → 1 as d → ∞ at the
kernel rate e^{-d²/2}: the RN factor's correlation with the remote jet decays
through τ(y) (displayed: 1.08 at d=1.5 axis → 3.5e-4 at d=5 axis, 2.0e-6
transverse) and the y-law reversion (Var(f|pins) table, §2(a)).

### 3.2 The unconditioned plateau (exact closed forms, new this lane)

At the exact side-24 law, H | f=v decomposes as U = (h₁₁+h₂₂)/2 ~ N(−a₂v, s_u²),
V = (h₁₁−h₂₂)/2 ~ N(0, s_u²), W = h₁₂ ~ N(0, a₂²), independent, with
det = U² − V² − W² and s_u² = (a₄−a₂²)/2 = a₂² up to 2.8e-120 (certified
Poisson correction; the finite torus is never planar). With T = V²+W² ~ χ²₂,
E[(T−c)₊] = 2e^{−c/2} and E[(c−T)₊] = c − 2 + 2e^{−c/2}, giving the closed
forms

    m_sad(v) = √2 e^{−v²/4},
    m_max(v) = (v²−1)Φ(v) + vφ(v) + √2 e^{−v²/4} Φ(v/√2),   m_min(v) = m_max(−v),
    ρ_sad⁰(v) = φ(v)/(2πa₂) · √2 e^{−v²/4},
    J(ℓ) := ∫_{b−ℓ}^{b} ρ_sad⁰(v) dv
           = √2/(2π)^{3/2} · √(π/3) · [erf(√3 b/2) − erf(√3(b−ℓ)/2)],

cross-checked against C1's frozen F5 anchors (nested-quadrature,
independent code path) to all 12 digits: m_sad(b) = 0.986663322476,
m_max(b) = 1.413625600767, m_min(b) = 0.013037721709,
ρ_sad⁰(b) = 0.030493491569, ρ_max⁰(b) = 0.043689047070; Isserlis identity
E[det²|f=v] = v⁴+2v²+7 (= 11.9536 at v=b, matching C1). (Note: C1's
parenthetical "P_sad(v=0) = 1/√2" is P(det>0) = 1/√2 in our notation, and
its "E|det|(0) = √2" reads m_sad(0) = √2; the total is E|det|(0) = 2√2−1 —
notation note only; all working-point anchors agree.) By stationarity,

    E_uncond[N_w(𝒵)] = |𝒵| · J(ℓ)   exactly;   I_uncond(T²∖rD)
    = (576 − 5r²)·J(ℓ) = 2.92737·r³·(1+o(1))   (ladder-certified, r↓0).

### 3.3 The crude polarity-safe envelope (theorem spine)

Dropping all typing indicators (upper direction) and using Cauchy–Schwarz
twice with exact Wick moments,

    g(y,v) ≤ (E det H_M⁴ · E det H_S⁴)^{1/4} · √(E det H_y⁴)  =: g_env(y,v),

so with the EXACT needle window mass wm(y) = Φ((b−μ_t)/s_t) − Φ((s−μ_t)/s_t),

    ρ^{P_r}(y) ≤ ρ_spine(y) := (p_grad(y)/Z_r) · wm(y) · G_cap(y),
    G_cap(y) = max{g_env at v ∈ {b, mid, s}} + 20·spread₃

(3-point control of g_env across the window; the Wick moments are
polynomials of degree ≤ 8 in v of the affine 9-pin means; ℓ = 2.1e-5, so
interior overshoot beyond the 3-point range is second-order small; the
factor-20 margin is displayed). Ledger at fixed d as r → 0: p_grad = O(1),
wm = Θ(ℓ) = Θ(r³), (E det⁴)^{1/4}-pair factors = Θ(r²), y-block O(1), so
ρ_spine = Θ(r³)·[O(1)/Z_r·r²-cancelling] — displayed on the r-ladder
(r = 0.05 vs 0.025: exponents +0.002…+0.021). POLARITY: the 1/Z_r consumes
the campaign-wide two-sided normalizer input Z_r = Θ(r²) [CONSUMES:
coordinator G.7-scope], verbatim the C2 §5.0 caveat: with only the certified
LPW route Z_r ≥ c_Z r⁵ the annulus envelope degrades by r^{-3} and the crude
spine goes vacuous; the far-zone refined route (§3.1) needs Z_r only inside
the κ_cross denominator — same consumption.

## 4. Probe computations (evidence, labeled; transcripts t_normal ≡ t_opt)

All per-station factors exact-kernel (mpmath dps=100, certified inversions,
tails < 1e-60); the QMC ratios fixed-seed antithetic (N = 2e6 pair / 1e6
nine-pin), SEs displayed. Z_r(0.05) = 8.06118054e-3 ± 7.9e-6 (C1 F6:
8.059372673e-3 — agree). Headline remote-probe table (r = 0.05; θ measured
from the M–S axis):

    d     θ    Var(f|pins)    pgrad/pgrad0  wm/wm0   τ_cross    κ_cross(rel,cert)  g/(Z_r m_sad)    Z_r^yv/Z_r
    1.5   0    0.1903856029   1.3623        3.369    1.081      25.17              1.3771±.0029     0.9998
    1.5  90    0.6574525665   1.1986        2.419    1.034        5.32              0.0978±.0004     0.2118
    2     0    0.5663100819   0.7592        3.475    0.895       24.32              1.3624±.0033     1.0002
    2    90    0.9084218199   1.1216        1.276    0.867       12.89              0.3070±.0013     0.4586
    3     0    0.9787472613   0.9717        1.236    0.720       24.61              1.2417±.0036     1.0001
    3    90    0.9987659022   1.0043        1.016    0.174       12.57              0.8033±.0029     0.8947
    5     0    0.9999999590   1.0000        1.000    3.5e-4       0.626             1.0046±.0033     1.0000
    5    90    0.9999999996   1.0000        1.000    2.0e-6       0.047             1.0027±.0033     0.9999

The RN ratio g/(Z_r m_sad) sits in [0.098, 1.38] over all probes and reverts
to 1.00 ± 0.01 at d = 5 — the weighted law's approach to the unweighted law,
quantified. The certified relative cross bound κ_cross is tight at d = 5
(0.63 axis / 0.047 transverse, driven by the tiny τ) and vacuous at d ≤ 3
(the Lipschitz constant is lossy where the coupling τ = O(1)); the sharp
values at d ≤ 3 are the QMC displays (evidence).

Remote-zone envelope (D4 net: d ∈ {0.1,0.15,0.2,0.3,0.4,0.5,0.75,1,1.5,2,3,4,5},
θ ∈ {0,45,90,135,180}°, full circle by the exact reflection symmetry about
the M–S axis):

    * crude spine (polarity-safe cap form): ρ_spine/r³ per unit area rises
      from ≤ 0.01 (inner ring, needle-suppressed; the on-axis corridor is
      dead to e^{-10²⁸⁶}-scale at d ≤ 0.2 because the conditioned-mean ridge
      rides ABOVE the window there) to the plateau 0.55 (d ≥ 4);
      zone integral over {0.1 ≤ d ≤ 5}: 17.02·r³;
    * typed-y envelope (y-saddle indicator retained via anchor-certified GH
      quadrature, evidence): plateau 0.0178, corridor dead; zone integral
      1.41·r³;
    * r-ladder: ρ_env/r³ exponents +0.002…+0.021 at fixed d ∈ {1,2,3,5}
      (Θ(r³) certified within ±0.6, displayed within ±0.03); relative
      κ_cross r-stable within ±25% (the absolute bound ∝ r² like the main
      term) — the remote term is Θ(r³) with r-independent RN correction.

Historical consistency displays (labeled, never proof): C020's local
α-constant C* = 0.973 vs our remote raw plateau coefficient 2.9274 (the
remote raw count is ~3× the historical LOCAL α-constant — the AO/percolation
qualification is what removes it, cf. PERC-DECAY); C1/C012 anchor rel diffs
(6.4e-4, 9.6e-3 — ours agree to 12 digits, independent code path); C021
corridor ceiling 2.6e-8 (β corridor) vs our corridor window-saddle kill
(e^{-10²⁸⁶} scale at d ≤ 0.2, needle off-window: consistent with the
corridor carrying no window saddles); B2d discrete A.rem fraction 3107/8801
(PL model, structural display).

## 5. The theorem statement the assembly consumes

**D3-THM (assembly-consumable).** On Reg ∩ TYP, at the exact side-24 LPW law,
for 0 < r ≤ 0.1 (probe ladder r ∈ {0.05, 0.025, 0.0125}; torus side 24 ≫ all
scales):

    P_r(A.rem) ≤ I_hole(r) + I_ann(r) + I_far(r)  =  O(r³),

    I_hole ≤ E_{P_r}[N_qual(rD)] ≤ C*_env·r³ = 1.284·r³
        [CONSUMES C1's chart envelope; carries the Conn_{rD}-failure
         condition, priced lower once branch/AO control lands],
    I_ann(r) = E_{P_r}[N_w(rD^c ∩ {2r ≤ d ≤ 5})] ≤ 17.02·r³
        (crude spine, polarity-safe form: exact-kernel stations + exact
         window mass + 3-point G_cap; the typed-y evidence envelope gives
         1.41·r³; the QMC-true qualified count is far below both),
    I_far(r) = E_{P_r}[N_w({d ≥ 5})] ≤ (576 − 25π)·J(ℓ)·(1 + κ_far(5))
        = 2.5283·r³·(1 + κ_far),  κ_far(5) ≤ 0.63 + ε_QMC
        (certified cross bound 0.63 axis-worst + QMC RN 1.00±0.01),
    ─────────────────────────────────────────────────────────────
    P_r(A.rem) ≤ 20.9·r³  (r = 0.05 display; spine form),
    P_r(A.rem) ≤ 6.5·r³   (typed-y evidence envelope form).

The assembly-primary form (no double counting with C1):
P_r(A) ≤ E_{P_r}[N_w(rD)] + [I_ann + I_far], the bracket = this lane's
19.6·r³ spine / 3.9·r³ evidence envelope.

**Validity domain and named blockers/consumption points:**
  (i) [CONSUMES: coordinator G.7-scope] the two-sided normalizer
      Z_r = Θ(r²); with only the certified LPW c_Z r⁵ the 1/Z_r-bearing
      spine degrades by r^{-3} (verbatim C2 §5.0's polarity caveat);
  (ii) D3-LEMMA-RN-UNIF (named, foundations countersignature flagged): the
      sup over each zone of the bracketed factors in §3.1 is attained on the
      displayed probe net up to the displayed smoothness margins (the
      factors are exact-kernel, smooth in y with e^{-d²/2} tails; the
      score-IBP step through the non-smooth typing indicators uses the
      1-Lipschitz negative-part representation — the smoothing formality is
      the same grade as B1's Lemma R);
  (iii) [CONSUMES C1 chart lane] the near zone rD ∪ B(pair, 2r) (the disk
      d < 2r around the pair) is the chart lane's region; my zone starts at
      d = 2r = 0.1 where the crude envelope is already ≤ 0.01·r³ per unit
      area (displayed), so the stitch is benign;
  (iv) [CONSUMES: PERC-DECAY] for the o(r³) upgrade only (§2(b)).
**Grade summary:** the event form, the cover lemma D3-SPLIT, the count split
D3-COUNT, the unconditioned plateau closed forms and J(ℓ), the spine's
polarity-safe form (modulo (i) and the zone-uniformity step (ii)), and all
certificates are theorem-shaped with fail-closed verification; the probe
ratios (QMC), the GH typed-y envelope, and the zone-integral trapezoid
assemblies are deterministic evidence with SEs/refinement certificates.

## 6. Certificates and falsifiers

`d3_perc.py`: fail-closed ck()→SystemExit(1) throughout (library hashes vs H2
MANIFEST pre-import; inversion residuals; K1(0)=1; spectral/image agreement;
certified tails < 1e-60; Poisson identities; C1-anchor windows at 2e-9/2e-11;
GH anchor rel diff < 3e-4; Z_r drift window; exponent windows; GL refinement
< 2%; cap-domination checks; PASS digest line). No bare asserts; `python3`
and `python3 -O` outputs byte-identical (`t_normal.txt` ≡ `t_opt.txt`);
deterministic (dps=100, fixed MT19937 seeds, fixed loop orders).
`d3_falsifier.py`: F1 discrete cover mirror (453 ensemble A.rem pairs +
carved E₁/E₂ synthetics; killer-aware persistence cross-checked against
B1F.persistence; B1's falsifier hash-pinned 818df798…); F2 engine re-run
(byte-identity vs frozen transcripts + headline windows). Both modes
byte-identical (`tf_normal.txt` ≡ `tf_opt.txt`).

END_FROZEN_BODY
