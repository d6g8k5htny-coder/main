# Museum triage vs the public math ledger

**Object:** GROK-HEAVY-MUSEUM-TRIAGE-20260926-v1
**Scientific effect:** NONE. Does not accept Theorem A. Does not import TOE claims.

User insight: if anything survives ten years it is three separate exhibits, not `C = R + K + B` in one shot.

## What is actually on public GitHub now

The live front door (`main`, `Math-`) and the profile bio already implement exhibit 3: Gaussian random fields + persistent homology + reproducible checks. Public STATUS scoped ACCEPT is D2 remainder, D3 SIDE24 coefficients, D4 P15 Theorem F, D6 token hygiene. D1 Theorem A and D5 pin stay AMEND.

That is the program this session has been reviewing.

## Exhibit 1 — helicity barrier

Claimed law (2025 README only):

    |Δζ4| = 0.1843 - 0.2051 C_B + 0.022 C_B²,  βc ≈ 0.5

GitHub search: `filename:helicity_barrier.py` = 0. No ACE / PSP data files on default branches. Formula lives in `main/history/2025/README.original.md` with a ✅ Validated checkmark. That checkmark is **not** a Math- ACCEPT.

External literature is real: McIntyre et al., Phys. Rev. X 15, 031008 (2025) report a helicity-barrier signature in solar-wind turbulence near β ≲ 0.5 and σ_c ≳ 0.4. That paper does not state the quadratic above. The quadratic is a claimed regression, unpublished as a standalone derivation on this account.

Do not draft a heliophysics paper this session. Do not treat the barrier as a persistence theorem.

## Exhibit 2 — gauge complexity

    K(G) = λ · r(G) · ||f||²,   K(R|G) = μ Σ d(R_i) C_2(R_i)

`filename:gauge_theory.py` = 0. No computed `K(SM)` vs `K(SU5)` vs `K(SO10)` on default. Norm `||f||` and first-principles λ, μ are unspecified in the historical README. Absent as a proof object.

## What the insight correctly de-prioritizes

- `C(n) = n·K + exp(α(n-3)²)` — minimum at 3 is inserted, not derived from `K(R|G)`.
- Riemann zeros as a corollary of `δC=0` — not a standalone proof.
- The whole-museum TOE.

Those claims are already absent from public STATUS. Keep them out.

## What this does *not* change about D1

The fold lock `ℓ=(κ/6)r^3`, SIDE24 coefficient enclosure, and P15 Theorem F are the solid exhibits inside program 3. Helicity data cannot close A3 floor, cap implication, Condition (ND), or D5 pin. Different buildings.

Do not merge main #163 as acceptance.
