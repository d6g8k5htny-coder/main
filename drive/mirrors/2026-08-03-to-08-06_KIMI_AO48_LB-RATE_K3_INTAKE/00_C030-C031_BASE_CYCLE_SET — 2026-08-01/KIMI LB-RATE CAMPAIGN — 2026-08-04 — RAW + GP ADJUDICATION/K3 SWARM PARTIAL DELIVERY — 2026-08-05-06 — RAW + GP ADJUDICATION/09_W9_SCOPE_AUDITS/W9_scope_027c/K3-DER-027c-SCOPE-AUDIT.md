# K3-DER-027c-SCOPE-AUDIT — W9: DER-027c scope audit + r-continuation report

**Program:** SIDE24/q0 — K3 SWARM Phase 4, workstream W9.
**Mandate:** DER-027c SCOPE AUDIT (supersedes WO-063 Task 4c's frame; DER-027c remains
the starting record). Items: (1) reproduce the deterministic conditional-mean result at
r = 0.025; (2) distinguish deterministic mean topology from noisy-field saddle counts;
(3) attempt interval continuation in r (the r-uniform enclosure); (4) preserve
exact-rung status and name the nonuniform obstruction with rates.

## Inputs (echo)

| input | sha256 |
|---|---|
| WO-063 work order (starting-record frame) | `e699bf44b7b84fca88faa2b509c1686ea9f8e500aee0dbe618c859b5ec5ad2e2` |
| KIMI-DER-027c.md (starting record) | `258f55df9baabf91a04d192f7b5b9bf993d34a25819b5721aea2392e66c93b05` |
| verify_ridge_two_sided_v1.py (starting certificate) | `d5f4a4ca84c7bbf09a4e26d57f6b299dde068b1c19d6df98f042326210c539a4` |
| lb1_sup_over_zone_certificate.py (object conventions) | `af19344df3524d03c8d54e188bddb30f41aa5a980414a5f588e84be995775c5e` |
| lb2_cert.py (RKHS/Kantorovich scheme) | `bf76df4189033525526d6fd4a73ca15ef62095de119d114548e033d3e137175d` |

**Object (unchanged from DER-027c):** the 9-pin conditional mean
m₉(x) = E[f(x) | 1-jet pins at M = (−r/2,0) value b = 6/5 (EXACT), S = (+r/2,0) value
b − ℓ, ℓ = r³/6 (EXACT), Y = r·(−1.26, 0.24) value v* = clip(μ_t, [b−ℓ, b])] of the
periodized side-24 2D Bargmann–Fock field, exact periodized kernel on the spectral
lattice (π/12)ℤ², masses e^{−|k|²/2}, truncation |k| ≤ 30 with certified tail < 1e-60.

**Certificate program:** `verify_ridge_scope_continuation_v1.py` — fail-closed ck()
(any failed check raises SystemExit; no asserts; deterministic), 73 checks, receipts
only. Normal and `python3 -O` transcripts **byte-identical** (cmp verified):
sha256 `3dd113e32922321e60cec4ae36e08d13a2cc85b00cf617b7035f647db66ef99a`.

**Precision labels:** EXACT (rational/integer data: b = 6/5, ℓ = r³/6, pin geometry,
rungs) vs decimal-dps-120 (mpmath computations with displayed envelopes) vs
float64+[5e-7] (grid engine with certified allowance EPS_F = 5e-7, whose analytic
worst case 4.46e-7 it exceeds).

---

## Verdicts (labels; program output contains receipts only)

| mandate item | verdict |
|---|---|
| (1) reproduce DER-027c at r = 0.025 | **PROVED [EXACT]** |
| (2) mean-topology vs noisy-field separation | **PROVED [scope table below]** |
| (3) r-uniform continuation | **PROVED at house grade [UNIFORM-R] over r ∈ [1e-6, 0.05]; the fully interval-rigorous tier is CLOSED-NEGATIVE (measured obstruction)** |
| (4) exact-rung status + named obstruction | **PROVED [EXACT]: certified rungs to r = 1.455e-12 at dps 120; obstruction named with measured rates; margins do not degrade** |
| r → 0 closure (uniformity at the singular point) | **OPEN — named route: exact limit object (C012/C013/C026), dependency not re-proved** |

---

## Item (1) — reproduction of the DER-027c census at r = 0.025 [EXACT]

The certificate rebuilds the 9-pin mean at the certified rung with the Cholesky-whitened
engine (whitened Gram ≡ I; forward-substitution solves) and cross-checks it against the
raw engine of DER-027c: **deviation 2.6e-109** (gate < 1e-60). Every DER-027c quote is
re-derived and gated (tolerances shown):

- P* = (1.30560936309, 0.685758739695), m(P*) = 1.9311903575579 (gate vs
  1.93119035755789484 at 1e-12), d(P*) = 1.4747477269211 (gate 1e-9), definiteness floor
  min|eig H| = 2.5874019 (gate vs 2.587401874 at 1e-6), uniqueness radius ρ₂ = 0.1061.
- Q* = (−1.10400474713, 0.879685522551), m(Q*) = 1.6665982193618 (gate vs
  1.66659821936183025), d(Q*) = 1.41162073528202, floor 1.9589439 (gate 1.958943924),
  ρ₂ = 0.08032.
- (b − v*)/ℓ = 0.499715906773 (gate 1e-9); ‖m₉‖²_H = 13.219736462363 (gate 1e-9).
- 7-point census in B₂.₅ typed exactly as DER-027c: M max at b = 6/5 exactly; S saddle at
  b − ℓ; Y saddle at v*; minima at −0.58051424651105 and −0.3317785491593; the adaptive
  second-order exclusion (float64, EPS_F = 5e-7) leaves 167 survivor cells, every one
  inside a certified uniqueness ball.
- Zero above-b saddles, torus-global: annulus 2.5 < d ≤ 3 max certified upper bound
  0.948134 < b; torus exterior d > 3 max certified upper bound 0.416956 < b.
- Kernel exactness: spectral-vs-planar cross-representation agreement < 1e-60 at all
  probe jets; 1D lattice tails < 1e-60 for orders 0,2,4,6; Poisson image bound < 1e-25
  torus-globally through order 3.

**Conclusion of item (1):** the deterministic conditional-mean result of DER-027c is
reproduced at the certificate standard, with two independent engines agreeing to 1e-109.

## Item (2) — which conclusions cover which object

| conclusion | object covered | object NOT covered |
|---|---|---|
| 7-point census, types, values, uniqueness balls (r = 0.025) | m₉ (deterministic analytic function) | noisy field f̃ = m₉ + ξ |
| zero above-b saddles, torus-global | m₉ | f̃ |
| ridge maxima paths P*(r), Q*(r), all rungs/blocks below | m₉(·; r) family | f̃(·; r) |
| conditioning/obstruction statements | the pin-Gram of the m₉ construction | — |

Receipts separating the objects (S2): the conditional variance at the ridge maxima is
certified v(P*) = 0.174043537433, v(Q*) = 0.126943975593, both in (0, 1) — so the noisy
conditioned field f̃ = m₉ + ξ is a genuinely different object (sd 0.417/0.356;
(m − b)/sd = 1.753/1.310). Expected critical-point counts of f̃ above b are Kac–Rice
objects owned by LB-1's Lemma H-SUP and the C029 B3(iii) mean-ridge channel; **no such
count is computed or claimed here**, and nothing in this certificate's mean-topology
census constrains them (LB-1 [F] certifies an O(1) expected count near P* — a statement
about f̃, not about m₉).

## Item (3) — continuation in r

### Tier UNIFORM-R (house grade) — PROVED over r ∈ [1e-6, 0.05]

Per block I = [r_hi − w, r_hi], the certificate certifies, for **both** ridge maxima and
**every** r ∈ I, existence/uniqueness in an explicit tube and a uniform eigenvalue
floor, via r-dependent Kantorovich bounds:

- exact midpoint certification: Newton root, pointwise Kantorovich (β, η, α, ρ₂);
- exact r-derivatives of the mean as RKHS elements: ‖m′‖_H, ‖m″‖_H by the section
  recursion c⁽ⁿ⁾ = Σ_pp⁻¹w⁽ⁿ⁾ with mixed section Grams (validated against finite
  differences at orders 1,2,3: rel dev 2.8e-13 / 5.6e-10 / 4.4e-6);
- uniform brackets from Hilbert-space Taylor: sup_I |∇m| ≤ res₀ + √2·(‖m′‖h +
  ‖m″‖h²/2 + C3·h³/6); inf_I λmin ≥ λmin₀ − √8·(same); sup_I ‖m‖_H likewise, with
  h = w/2 and C3 = 1.25 × (9-point exact mesh max of ‖m‴‖_H) — the **house named
  analyticity formality** (C027 standard: derived-on-grid cap; every other ingredient
  exact). Measured C3 ≈ 830–860 across the whole chain; ‖m′‖_H ≈ 13.9, ‖m″‖_H ≈ 74.9
  at r = 0.025;
- uniform gates: α_I = β_Iγ_Iη_I < 1/2 and λmin-inf > 0, both displayed per block;
- chain gates: the exact root at each shared rung lies inside the next block's
  uniqueness tube (gate displayed; all passed).

**Outcome (receipts S4):** 22 blocks, adaptive widths 0.002 → 0.0025 (largest failing
width 0.004 halved once at the top), uniform α ≤ 0.341 throughout; certified-uniform
coverage **r ∈ [1e-6, 0.05]** for both P*(r) and Q*(r); chain-end margins
λmin-inf = (2.5654, 1.9428), tube radii ρ₂ = (0.1051, 0.0796). Continuity of each root
path on the covered range follows by the implicit function theorem applied at each
certified rung (uniform nondegeneracy) — standard formality, named here.

### Tier INTERVAL (direct interval arithmetic) — CLOSED-NEGATIVE, obstruction measured

- S5.1: direct `iv` LU inversion of the 9-pin Gram over r ∈ [0.0499, 0.0501] aborts
  (ZeroDivisionError): interval pivots straddle zero even on a 2e-4 block at r = 0.05.
- S5.2: interval Cholesky viability search (widths 1e-2 halved 21 times to 4.8e-9) finds
  **no viable block at any rung** {0.05, 0.025, 0.0125, 0.00625, 0.003125}: the smallest
  radicand (pivot² ~ r⁴, measured pivot ~ r²) is overtaken by elimination-amplified
  interval widths (~1/pivot² ~ r⁻⁴). Labeled analysis (not a measurement): viability
  needs |I| ≲ r⁸, so direct interval continuation is exponentially costly as r → 0.

## Item (4) — exact-rung status and the named nonuniform obstruction

**Exact-rung ladder (receipts S3, all gates per rung):** Newton + Kantorovich at both
ridge maxima certifies **36 rungs** r = 0.05/2^k, k = 0 … 35, i.e. down to
**r = 1.455e-12 at dps 120**, with v*-window, eigenvalue-floor, ρ₂ and envelope gates
displayed. First failing rung r = 7.276e-13 (gates: v*-window, Kantorovich-Q) — the
μ_t/clip precision floor at fixed dps; this floor recedes with dps (whitened pin
reproduction residual stayed < 1e-66 at every rung; the failing ingredient is the raw
6-pin μ_t Schur computation at cond ~ r⁻⁶).

**Measured rates (log-log slopes over the ladder):**

- margins **converge to positive limits** (no degradation): m(P*) → 1.918715,
  m(Q*) → 1.658100, λmin floors → (2.56542, 1.94283), gaps → (0.718715, 0.458100);
  first-order drift |dm(P*)/dr| → ~0.56, |dλmin/dr| → ~1.06 (finite limits);
- conditioning: Cholesky pivot ~ r^2.00, cond(Σ_pp) ~ r^6.00 (measured slopes).

**Named obstruction.** The only obstruction to r-uniform certification is **pin-Gram
coalescence conditioning** — numerical, not topological: as r → 0 the nine pin
functionals coalesce into the 3-jet at the origin, κ(Σ_pp) ~ r⁻⁶, so (i) pointwise
exact certification at fixed precision has a dps-floor (measured: r = 1.455e-12 at dps
120; recedes with dps), and (ii) interval enclosures of any Gram factorization lose
viability at block widths ≲ r⁸ (S5). No Kantorovich margin, eigenvalue floor, value
gap, or uniqueness radius degenerates as r → 0 (measured limits above). The point
r = 0 itself is singular for the pin family; closing uniformity at r = 0 routes through
the exact limit object (the C012/C013/C026 rung, where (b − μ_t)/ℓ → −0.4999290
already bounds the family) — a named dependency, not re-proved here.

## Exact valid-scope table

| statement | valid scope | tier |
|---|---|---|
| census (7 pts, types, values), uniqueness balls | r = 0.025 exactly | EXACT |
| zero above-b saddles of m₉ | r = 0.025 exactly, torus-global | EXACT |
| P*, Q* Kantorovich-certified (existence, uniqueness, floors, values) | 36 rungs, r ∈ {0.05·2⁻ᵏ}, smallest 1.455e-12 | EXACT |
| P*(r), Q*(r) uniform: existence, uniqueness in tube, λmin ≥ displayed inf | every r ∈ [1e-6, 0.05] | UNIFORM-R (house named cap) |
| margin limits as r → 0 | asymptotic statement from measured ladder | EXACT (measured) |
| interval-enclosure nonviability | rungs {0.05 … 0.003125}, widths ≥ 4.8e-9 | INTERVAL (measured) |
| noisy-field f̃ critical counts | **not covered** (owner: LB-1 H-SUP, C029 B3(iii)) | — |
| r = 0 limit topology | **not covered** (owner: C012/C013/C026) | — |
| rungs r > 0.05 | **not covered** (not attempted) | — |

## Hypotheses

- H1: m₉ is built from the exact periodized kernel (spectral lattice, |k| ≤ 30, tail
  < 1e-60); cross-representation agreement < 1e-60 gates this.
- H2: Gaussian conditioning gives Var(∂ᵃf | pins) ≤ Var(∂ᵃf) = (2a₁−1)!!(2a₂−1)!!;
  used for all RKHS sup bounds (exact for Gaussian fields).
- H3: the C3 mesh cap (factor 1.25 on a 9-point exact mesh of ‖m‴‖_H per block) is the
  house named analyticity formality; all other UNIFORM-R ingredients are exact.
- H4: IFT continuity of root paths between certified rungs (standard; named).
- H5: float64 grid arithmetic within EPS_F = 5e-7 (analytic worst case 4.46e-7).

## Falsifier

Re-run `verify_ridge_scope_continuation_v1.py`: any failed check aborts with FAIL
(SystemExit). Four mutation receipts under `mutations/` demonstrate sensitivity:
m1 (DER-027c quote digit) → FAIL at the quote gate; m2 (chain gate tightened) → FAIL at
chain gate P; m3 (ladder eigenfloor gate) → FAIL at top-rung certification; m4 (object
separation window) → FAIL at the v(P*) window.

## Artifacts

| file | sha256 |
|---|---|
| verify_ridge_scope_continuation_v1.py | `4c1a8523812bf600bde41b486ee1e6cf45d71ec0fe707d89598f7f8c03f9ac97` |
| t_normal.txt (= t_O.txt, byte-identical) | `3dd113e32922321e60cec4ae36e08d13a2cc85b00cf617b7035f647db66ef99a` |
| mutations/m1_quote_gate.py (+ receipt) | `9ea7477ff6f64679c0add79bb39ed13135ff1738894a456f0d148d2d27c0c7b5` (+ `27ba7a8ea36936efd1608460639b66319c72bb5353baddcc2a7c3bf8f5fd1768`) |
| mutations/m2_chain_gate.py (+ receipt) | `81be03c57f960cb55e38be2e00700bd104dd30fc2308f0f409bfc14b03f5e2a2` (+ `891edab0aa3e1651e27de863c280648fcd4e66d8487f33739c6d1c52e7adfffe`) |
| mutations/m3_ladder_gate.py (+ receipt) | `d7de20dc0dc0f0f6ee1e1736df14f2335a5f73fb87a37d407d8959a35413d629` (+ `ac1c604ce5efbfb2fa7ac915119cee1a4c610ddf08e23e5498fbc76449e64af3`) |
| mutations/m4_scope_gate.py (+ receipt) | `fb18b709ff5a2732c0a2ba53a1f961e452533a080f06a1b2234f73be88b19008` (+ `68d348e1c33eac7d01bcbd617b83e28154e6b2daf0f21e89a2da8a252c788af3`) |
