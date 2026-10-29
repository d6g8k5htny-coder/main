# Formal glossary — project terms → standard mathematics → Lean

A project term has to be given a formal translation before a statement using
it can be written in Lean. This table records, descriptively, how the pinned
sources use each term and which standard object the formaliser would have to
define or import. These are **proposed correspondences, not adopted
equivalences**: per the lane's
[reconnaissance memo](../docs/reconnaissance/2026-09-27-formal-verification.md),
elder selection needs its own point-process, pairing and conditioning
definitions, a lifetime asymptotic coefficient is not automatically a
Hermite-expansion coefficient, and **a missing formal translation is an
obligation to resolve, not evidence of novelty or ill-definition**. Novelty
claims are a separate matter for the
[novelty reconnaissance](../docs/RECON_NOVELTY_20260925.md). Nothing here
accepts or rejects any result.

The Math- pilot ships its own source-grounded glossary for the GP-FOR-192
companions; this file covers the SIDE24, RN-counting and P15 vocabulary that
the pilot does not. Merge rather than duplicate if the two overlap later.

Sources read for these entries: the SIDE24 note
([`sources/side24_v1/PROOF.md`](sources/side24_v1/PROOF.md)), the D2 remainder
note (`Math-` `frontiers/three_fronts_20260924/LIFETIME_REMAINDER.md`), the D4
remote-window proof (`frontiers/remote_window_20260924/PROOF.md`), the D6
full-price proof (`frontiers/full_price_20260924/PROOF.md`) and
[STATUS.md](../STATUS.md). Where an entry says "read the source", the standard
mapping still has to be extracted by whoever formalises that object.

Lean column: `—` means no Lean object yet; `core` means statable in core Lean;
`Mathlib` means the standard object exists in Mathlib and a Mathlib lane is
needed to state it.

## Gaussian persistence (D1, D2, D3)

| Project term | Standard object | Lean |
|---|---|---|
| **Periodized field**, `K_L`, `K_24` | Centred, variance-one stationary Gaussian random field on the flat torus `X = ℝ^d/(LZ^d)` whose covariance is the normalised theta-function periodisation of the Gaussian kernel `exp(−|z|²/2)`. `SIDE24` is the case `L = 24`. | Mathlib: `MeasureTheory` Gaussian measures, `AddCircle`; the covariance kernel itself is statable in core as a series but not usable |
| **Reference / CONTACT covariance**, `K_∞` | The unperiodised kernel `exp(−|z|²/2)` used only to define the reference coefficient `c_{d,ref}`; no infinite-volume theorem is claimed | Mathlib: `Real.exp` |
| **Pins** `M = −ru/2`, `S = ru/2` | Two conditioning points on the axis `u` at separation `r`, with prescribed heights `b`, `b − k r³` and zero gradients | — |
| **Continuous Gaussian regression** `Q`, `Q_r` | Conditional law of a Gaussian field given finitely many linear observations (conditional Gaussian measure) | Mathlib (partial) |
| **Endpoint weight** `W`, **FULL normalizer** `Z = E_Q W` | Kac–Rice Jacobian weight `|det H_M| · |det H_S|` times index indicators, and its expectation under the regression law | — |
| **Candidate** pair, **candidate density** `ν_cand(ℓ)` | Ordered (maximum, index-`(d−1)` saddle) critical-point pair with height gap `ℓ`; its density is the intensity of the corresponding marked critical-point-pair process (first-moment / Kac–Rice density) | — |
| **Elder** pair, **elder density** `ν_eld(ℓ)` | Critical-point pair actually paired by the elder rule of superlevel-set `H₀` persistent homology (a finite `H₀` bar of length `ℓ`); density = intensity of finite `H₀` persistence pairs | — |
| **Lifetime** `ℓ` | Persistence (birth minus death height) of an `H₀` bar | core (`ℝ` needs Mathlib) |
| **Lifetime coefficient** `c = c_{d,L}` | The constant in the leading term `c ℓ^{−1/3}` of the small-lifetime density asymptotics; defined by equation (15.2) of the parent note | — |
| **Reference coefficient** `c_{d,ref}` | `Γ(7/6)·(3/2)^{1/3}·D_{d−1} / (2√3·π^{d−1}·√π)`, display (1) of the SIDE24 note | Mathlib: `Real.Gamma`, `Real.pi`, `Real.sqrt`, rpow |
| **Cone moment** `D_m` | `E[det(A)² · 1{A ≺ 0}]` for a specified Gaussian symmetric `m×m` matrix `A`; a moment of the squared determinant over the negative-definite cone. `D_1 = 4/3`, `D_2 = 29/6 − √6` | Arithmetic parts: core (`cone_moment_m1_thirds`, `cone_moment_m2_algebra`); the integrals: Mathlib |
| **Image ledger**, **omitted periodic image** | Deterministic bound on the difference between derivatives of the periodised and unperiodised kernels at `0`, i.e. on the Poisson-summation image terms `n ≠ 0` | Arithmetic parts: core (`image_constant_value`, `covariance_relative_bound`, `exp_taylor_partial_sum_gt_ten`) |
| **Outward arithmetic** | Rational interval arithmetic with outward rounding to a fixed grid (`10^-80`), containment unconditional | — (would be Mathlib `Set.Icc` over `ℚ`) |
| **Matrix cap**, **marked cylinder**, **elder selection** (D1) | Read the parent source (`imports/lifetime_parent_20260925/`) with its reading rule; the D1 chain is ACCEPT — scoped at Layer 0 (reconciled in Math- #126, see `STATUS.md`) and has no agreed standard mapping yet | — |
| **RN / JETMOD / 24-jet obligations** | Historical downstream proof obligations tracked in `frontiers/downstream_gate_20260925/GRAPH.json`; not a single mathematical object | — |

## RN counting (D4, D5)

| Project term | Standard object | Lean |
|---|---|---|
| **Remote critical point** | Critical point `x` with `dist_X(x, 0) ≥ ρ`, i.e. outside a fixed neighbourhood of the two pins | — |
| **Height window** (between-pin) | The height interval `(b − k r³, b)` between the pinned saddle and maximum | — |
| **RN count** `N_{r,j}(E)` | Number of critical points of Morse index `j` in a Borel set `E` with height in the window; its expectation is a Kac–Rice integral | — |
| **Three determinant factors** | The Kac–Rice integrand factors `|det H|` at the two pins and at the counted point, under the regression law | — |
| **Contact kernel** | The positive mean kernel identified in the D4 proof for the counted point's conditional Hessian; read the source (section on the contact mean kernel) | — |
| **Fixed-remote** vs **shrinking** `ρ`, `η` | Whether the spatial exclusion radius `ρ` and witness separation `η` are held fixed as `r → 0` or allowed to shrink with `r` | — |
| **Pin neighbourhood**, **microdisk**, **collar**, **annulus** | Regions at distance `≲ r²`, `≪ r`, between scales, and at fixed scaled radius from the pins respectively (D5 decomposition); read the D5 sources | — |
| **Witness collision** | Two counted critical points whose mutual separation tends to zero (factorial-moment / second-moment regime) | — |

## P15 combinatorics (D6)

| Project term | Standard object | Lean |
|---|---|---|
| **Realized family** | The decreasing (downward-closed) set family `D` on disjoint coordinate blocks `X_i` with `|X_i| = a_i d_i + 1`, `|U ∩ X_i| ≤ a_i`, whose occupied-block support contains no edge of the clutter `H` | Mathlib: `Finset`, downsets; statable |
| **Clutter** `H` | Finite antichain of subsets (a Sperner family / simple hypergraph) of block indices | Mathlib: `Finset` antichain |
| **Palette** `K`, `K_H(d)` | Number of colour classes; `K_H(d)` the least `K` admitting classes `P_i` with `|P_i| ≥ d_i` and empty intersection over every `H`-edge | Mathlib: statable |
| **Capacity / demand** `d_i` | Per-block integer parameter; `d_i ≥ 2` is an essential hypothesis (demand one has an exact counterexample) | core |
| **Price** `c_v`, **transformed price** `φ(p) = min(1, −log(1−p))` | A weight per coordinate bounded by the hazard transform of its probability | Mathlib: `Real.log` |
| **Cover cost** `covercost_c(O_K(D))` | Minimum total generator price of a family covering the sets not partitionable into `K` members of `D` | Mathlib: statable |
| **Global hazard** `−log μ_p(D)` | Negative log of the product-measure probability of `D` | Mathlib: `Real.log`, finite product measure |
| **Sharp factor** `ρ* = 1/h*`, `h* = 3 − log(3e − 2)` | The constant in Theorem F; published enclosure `0.84547981724898672067 < ρ* < 0.84547981724898672068 < 6/7` | Mathlib: `Real.log`, `Real.exp`; a good `norm_num`/`Real.log` bound target |

## Governance / process terms

| Project term | Meaning here |
|---|---|
| **Layer 0** | Existing provenance, scope, review and CI system (hashes, `STATUS.md`, `PROOF_INDEX.md`, source-bound review, public-intake guard) |
| **Layer 1** | This Lean layer: statements and proofs checked by the Lean kernel and bound to Layer 0 bytes |
| **Author-side** | Produced by the author (human or model) and not yet checked by a distinct reviewer; applies equally to AI-generated Lean text |
| **Alignment review** | A distinct reviewer's check that a Lean statement says what the informal statement says at the stated scope; it does not re-prove the theorem (the kernel did) |
| **Informal anchor** | A verbatim quote from the pinned source bytes that a Lean target is attached to; verified as a substring by the gate |
| **`proved`** | Manifest-level label: proof text supplied and bound by hash. The strongest label a committed manifest may carry |
| **`kernel-checked`** | Receipt-level label only: a trusted `--run-lean` execution built the package, `leanchecker` passed, and every target's transitive axioms are within `propext`, `Classical.choice`, `Quot.sound` (here: none). Not a scientific status |
| **Alignment record** | JSON record validated by `--alignment`: accepted disposition, exact manifest and scope digests, all targets, distinct author/reviewer provider–family–agent, immutable evidence reference. Validation is structural; authenticating that the review happened is a controller's job |
| **Cross-check lane** | An independent AI prover producing a proof of the same Lean statement; extra evidence, same kernel, still author-side; requires actual availability, pinned identity and recorded execution |
