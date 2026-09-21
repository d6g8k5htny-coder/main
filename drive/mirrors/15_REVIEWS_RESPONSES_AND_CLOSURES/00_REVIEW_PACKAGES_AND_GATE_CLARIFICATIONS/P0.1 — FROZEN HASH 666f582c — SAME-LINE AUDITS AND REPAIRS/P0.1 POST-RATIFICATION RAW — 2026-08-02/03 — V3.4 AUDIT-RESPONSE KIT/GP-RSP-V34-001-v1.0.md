# GP-RSP-V34-001-v1.0 — V3.4 exact-display audit-response index

Status: PRE-STAGED / NO AUDIT VERDICT INFERRED

Source relay: `rp_c_rp_s_facewise_closure.md`, Drive ID `1sl4moxxKEOdeWV-TqV7WtE7komxsujOb`, 35,310 B / SHA-256 `40faad08824e4e77520143078ed606c8681532cc77ae9c9d42551baa5277b400` in the current raw recovery copy.

Full V3.4 archive: Drive ID `13QS9QQHxSiuLPIPSh9o5plS5HkClwSmz`; 12,465,983 B / SHA-256 `ba2a296b6a9c438eaf9f30aeaa40d4162195b256a9b2516259cd30a2468dd480`.

## Hybrid §6 envelope

`D(r,s)=[r(r+s²)]²(r²+s³)`.

Noncollinear charts:

`|det H_M|+|det H_S| ≤ C r(r+s²)P(J)θ^{-N}`,

`|det H_y| ≤ C(r²+s³)P(J)θ^{-N}`.

Axial density-weighted replacement:

`K_r(y,u) ≤ C s^{-7} D(r,s) Θ^{-N} exp(-c/Θ²)`,

and, after the legitimate value integration on non-endpoint collar charts,

`∫_h^b K_r(y,u)du ≤ C r^{-4} D(r,s) Θ^{-N} exp(-c/Θ²)`.

This is deliberately hybrid: it does not assert the withdrawn all-face pathwise bound on the exact axis.

## Endpoint displays

EP.1 on the axial cap `ρ≤2r`:

`|det H_M|+|det H_y| ≤ C d P(J)`,

`|det H_S| ≤ C r P(J)`,

`|det H_M det H_S det H_y| ≤ C r d² P(J)`.

EP.2 after value marginalization, radial volume, Palm normalization, the `O(r²)` angular-cap area, and any fixed regression loss `r^{-N}`:

`Z_r^{-1} ∫_{ρ≤2r} ∫_0^{δ_end r} d²[d^{-1}r^{-3-N}e^{-c/r²}] dd dω`

`≤ C r^{-1-N}e^{-c/r²}=O(r³)`.

## §6X withdrawal

The live relay explicitly marks former Lemma 6.1 false at `σ=0`, records the perpendicular remainder failure, and routes the counterexamples to `quartic_remainder_audit_note.md` plus `verify_quartic_endpoint_counterexample.py`. The relay omits the withdrawn argument block itself. Therefore this kit can prove live-text nonuse and counterexample coverage, but **not byte-complete preservation of the withdrawn block** without opening the authoritative V3.4 archive/original draft.

## §§8–9 ledgers

Collar endpoint oblique ledger: `r^{-2} d^{-3} r^{-1} (r⁵d) d²dd = r²dd`, which integrates over `0<d<ε₀r` to `O(r³)`.

Axial endpoint ledger before the angular-cap area: `r^{-2}(d^{-3}r^{-4})(rd²)d²dd = r^{-5}d dd`; EP.2 absorbs the remaining polynomial loss.

Singular-near angular axis:

`∫_0^{ρ₀} ρ(s²+ρ²)^{-m/2}e^{-c/(s²+ρ²)}dρ ≤ C_m` uniformly in `s`.

The remaining radial integral expands into

`r⁷s^{-5}, 2r⁶s^{-3}, r⁵s^{-2}, r⁵s^{-1}, 2r⁴, r³s²`,

with respective orders

`O(r³), O(r⁴), O(r⁴), O(r⁵ log(1/r)), O(r⁴), O(r³)`.

Thus both collar and singular-near ledgers close at `O(r³)` within the V3.4 draft, while package promotion remains HOLD pending independent audit.

