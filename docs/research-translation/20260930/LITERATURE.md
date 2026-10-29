# Literature comparison for the planar short-lifetime intensity

This note checks selected theorem statements and their applicability to the [manuscript](MANUSCRIPT.md), as of 30 September 2026. It is a bounded comparison, not exhaustive novelty clearance or a full proof audit of the external papers. The explicit arXiv versions and the precise statements inspected are recorded in [APPENDIX_SOURCES.json](APPENDIX_SOURCES.json), together with hashes of the HTML/PDF representations retrieved for this read. Those hashes establish representation identity, not mathematical validity. Project theorem identities remain the pinned Math- sources in [CROSSWALK.md](CROSSWALK.md) and [APPENDICES.md](APPENDICES.md).

## The comparison coordinates

The manuscript fixes `d=2`, `L=24`, the exact variance-one periodized Gaussian covariance, ordinary **superlevel H₀**, and exclusion of the essential global-maximum bar. Let `D_f=Σ_i δ_(b_i,s_i)` be the finite-bar diagram, with `b_i>s_i`. Its lifetime intensity measure is

```text
μ(A)=576^(−1) E D_f{(b,s): b−s∈A}.
```

It is useful to distinguish three expectation operations:

| Target | Definition, when well-defined | Interpretation |
|---|---|---|
| Per-area intensity | `E D_f /576` | Counts per realization per spatial area; the manuscript's target before lifetime pushforward |
| Normalized expected count | `E D_f / E N_f` | Weights realizations according to their number of bars |
| Expected normalized diagram | `E[D_f/N_f]` | Each nonempty realization has equal total weight; an empty-diagram convention must be specified |

Here `N_f=D_f(ℝ²)`. The last two expressions generally differ because expectation and division by a random count do not commute. None is the probability that a random diagram merely intersects a set. This distinction is mathematical bookkeeping for the present experiment: bins aggregate counts over independent realizations and divide by area and bin width.

A two-dimensional diagram density and a one-dimensional lifetime density also have different measures. If `ρ(b,s)` is a per-area birth/death density, then the lifetime density is `ν(ℓ)=∫_ℝ ρ(b,b−ℓ)db`, whenever this disintegration is justified. Changing from superlevels of `f` to sublevels of `−f` sends `(b,s)` to `(−b,−s)` and preserves positive lifetime `b−s`. These coordinate identities do not establish that another paper's model satisfies the manuscript's assumptions.

## L1. Kac–Rice: the actual external input

**Armentano, Azaïs and León, arXiv:2304.07424v3.** Theorem 2.2 gives Gaussian Kac–Rice under almost-sure `C¹` paths and positive-definite point covariance, with finite expectation on compact domains. Theorem 7.1 adds a nonnegative mark under its lower-semicontinuity and conditional-law continuity hypotheses; Remark 8 covers the latter for joint Gaussian fields. [Versioned primary text, §§2 and 7](https://arxiv.org/html/2304.07424v3).

For the manuscript, the field to which the counting formula applies is `(∇f(x),∇f(y))` on compact pair domains separated from the diagonal. The pin-rank argument verifies nonsingularity there. E2 uses continuous cylinder marks as the weighted base case, then establishes equality of finite measures and extends to the Borel global elder mark. The external theorem is therefore an input to E2, not a direct theorem about an arbitrary Borel elder indicator. It does not produce the selection probability, contact exponent or coefficient. [Exact E2 replacement, §§9.1–9.4](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_section9_borel_repair_20260925/REPAIR.md).

## L2. Expected diagram densities: which filtrations are covered

**Chazal and Divol, arXiv:1802.10457v2.** Theorem 3.1 assumes a density for a fixed-size point sample on a compact real-analytic manifold and filtering assumptions K1–K5, including compatibility and subanalytic nonvanishing gradients. Theorem 3.3 treats the variant with vertices entering at zero: its H₀ expected measure has a density on the vertical birth-zero line, while positive-degree diagrams have planar densities. Theorem 6.1 separately establishes a density for H₀ sublevel persistence of Brownian motion on `[0,1]`. [Versioned primary text, Theorems 3.1, 3.3 and 6.1](https://arxiv.org/html/1802.10457v2).

These are precise density-existence results, but none of these statements is a theorem for a smooth Gaussian field's superlevel filtration on a two-dimensional torus. The manuscript therefore uses its own repaired marked-pair argument for existence and its own contact estimates for the short-lifetime behavior. In particular, replacing the field experiment by Vietoris–Rips H₀ on sampled points would change the filtration and concentrate births at zero; it would not test the stated maximum–saddle law.

## L3. Gaussian Euler integrals: a signed aggregate of bar lengths

**Bobrowski and Borman, arXiv:1003.5175v2.** Definition 6.1 assigns a barcode the alternating sum of its lengths across homological degrees. Proposition 6.2 relates the corresponding truncated persistence quantity to an Euler integral. Under the Gaussian kinematic formula conditions, Theorem 6.4 evaluates its expectation at a deterministic truncation level. [Versioned primary text, §6](https://arxiv.org/html/1003.5175v2).

This provides a genuine Gaussian-field persistence result and should be credited as such. Its observable combines degrees with signs and truncates the filtration in height. The present target counts positive finite H₀ lifetimes and resolves their density near zero. A signed integrated identity cannot by itself identify that density: different distributions of bar lengths and higher-degree contributions can have the same signed total. No Gaussian kinematic formula is consumed in the manuscript's D1/D2 derivation, and this comparison does not independently verify that formula's hypotheses for an alternative computation on the torus.

## L4. Maximum–saddle pairs and the missing pairing probability

**Feldbrugge, van Engelen, van de Weygaert, Pranav and Vegter, arXiv:1908.01619v1.** Section 5, equations (58)–(60), describes expected superlevel persistence diagrams per area/volume and expresses planar H₀ persistence using a maximum–saddle pair density multiplied by a pairing probability `G₀(b,d,r)`, integrated over separation. The surrounding text identifies `G₀` with the event that the saddle kills the component born at the specified maximum. Sections 4.4–4.5 develop and numerically validate a fitting approach for the related Betti-curve probability. [Versioned primary text, §§4.4–5](https://arxiv.org/html/1908.01619v1).

This is a close conceptual antecedent: local critical-point type does not determine the persistence partner. The manuscript's exact global elder mark and separating cap address that same distinction for its specified finite-torus model. The displayed equations are not substituted into the manuscript's normalization ledger, and the fitting approach is not a source for its coefficient. The checked passage supplies neither the compact `1−p_r=O(r³)` argument nor the stated unrestricted `cℓ^(−1/3)+O(1)` theorem.

## L5. Estimation from independent diagrams

**Wu, Kim and Rinaldo, arXiv:2310.11982v2.** Section 2.2 studies kernel estimators from independent identically distributed diagrams. Assumption 2.2 requires a deterministic almost-sure bound on a total-persistence functional. Assumption 2.3 bounds a weighted intensity and the normalized density. Under these conditions, Theorem 2.4 controls fluctuations around the estimator's expectation; intensity control is weighted and away from the diagonal, while the normalized estimator has a different uniform bound. [Versioned primary text, Assumptions 2.2–2.3 and Theorem 2.4](https://arxiv.org/html/2310.11982v2).

The paper's separation of intensity and normalization informs the experiment's estimand. Its theorem does not automatically validate the present histogram or its error bars: the Gaussian amplitude is unbounded, the required deterministic total-persistence bound has not been checked, and the target here is a lifetime marginal. A finite expected moment from D2 is weaker than an almost-sure deterministic bound. Applying these statistical guarantees would require an explicit localization/truncation and estimator analysis, including its changed target and bias. Bars within one realization are not the independent sample units in that theorem.

## L6. A prior near-diagonal asymptotic for a different Gaussian process

**Baryshnikov, “Brownian motions, persistent homology and chirality,” 2025.** For standard Brownian motion with positive drift `m` on the positive time ray, Proposition 3.3 gives, on `0<b<d`, the H₀ sublevel diagram intensity density

```text
ρ(b,d)=4m² e^(−2mΔ)(1+e^(−2mΔ))/(1−e^(−2mΔ))³,
Δ=d−b,
```

and hence `ρ(b,d)∼1/(mΔ³)` near the diagonal. [Published primary article, §3 and Proposition 3.3](https://link.springer.com/article/10.1007/s41468-025-00224-w).

This is prior exact near-diagonal persistence analysis and prevents a broad claim that the manuscript introduces the first such law for Gaussian functions. Brownian paths are nonsmooth, the domain is one-dimensional and noncompact, and the displayed density is in two diagram coordinates, not an all-birth lifetime count per spatial area. Its exponent cannot be compared to `−1/3` without those qualifications. The manuscript's smooth contact-jet mechanism and finite torus are separate hypotheses, not harmless changes of notation.

## L7. Stability does not alone control a density at zero lifetime

**Cohen-Steiner, Edelsbrunner and Harer, SoCG 2005 text.** The Main Theorem, on PDF page 3, states bottleneck stability for continuous tame functions on the same triangulable space: the diagram distance is at most their uniform function distance. [Primary paper, Main Theorem](https://math.uchicago.edu/~shmuel/AAT-readings/Data%20Analysis%20/Edelsbrunner,%20Harer,%20Stability.pdf).

The experiment now supplies a [deterministic approximation construction](../../../experiments/periodic_h0/APPROXIMATION.md): a continuous PL representative on the same torus, derivative-based uniform error bounds and endpoint-safe lifetime-bin inequalities. As a direct deduction from a matching of cost `ε`, matched finite lifetimes differ by at most `2ε`, and bars of lifetime at most `2ε` may match the diagonal. These results do not by themselves establish convergence of a lifetime density near zero, a histogram bias bound, or interchange of the grid and `ℓ↓0` limits. Applying the construction to the numerical pilot still requires evaluated, certified nodal and derivative error bounds and arithmetic enclosures; those have not been supplied.

## Manuscript positioning supported by this read

A defensible description is: the manuscript assembles a specified **smooth finite-torus H₀ lifetime intensity**, a source-reviewed global pairing argument, its singular contact asymptotic and bounded unrestricted remainder, and an arithmetic enclosure of the corresponding coefficient. The scope is fixed by the exact field, all-birth finite-bar count, per-area normalization and lifetime limit. The comparison above does not establish priority for the exponent or certify a first result in any broader class.

The closest comparison questions for a human reader are now concrete. For the pairing literature, compare the exact global event and determinant-weighted conditional law. For diagram-density literature, compare the filtration and whether the measure is planar, a marginal or a normalized distribution. For statistical literature, check the actual random-measure hypotheses rather than borrowing rates from a similar-looking plot. For stability, require a quantified common-space approximation and an argument controlling diagonal mass.

This read verified the cited statements and nearby assumptions, not every external proof, later citation, journal revision or possible competing result. The broad survey and Gaussian-field simulation papers listed in the research guide remain useful context, but are not silently promoted to theorem dependencies here. The unseen V3 manuscript remains outside this comparison. No new external theorem is used to enlarge D1, D2 or SIDE24, and no live peer candidate is treated as accepted.
