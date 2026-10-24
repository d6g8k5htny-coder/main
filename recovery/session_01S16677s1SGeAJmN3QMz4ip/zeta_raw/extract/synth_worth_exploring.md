Ranked by value to the project's own goals (separating geometry from measure, and universality admission), not by RH relevance. None of these touches RH.

1. Theorem target T-CUE: short-bar law for fixed-N CUE.
- **Question.** Let U be Haar on U(N), N ≥ 2, with Z_U(θ) = Π_j 2 sin((θ−θ_j)/2), and let F_U = ∫Z_U on a fixed arc, dropping bars born at the arc ends. Does the expected short-bar density satisfy ν_N(ℓ) → c_N ∈ (0,∞) as ℓ ↓ 0? Here c_N = (1/3)∫ b0 κ^{-1} dλ, in the notation of Math- reviews/collision_mechanism_20260925 §D (D1), with α = 2, m = 3, β = 1. What is c_N explicitly (N = 2 in closed form), and what is the next-order term?
- **Why it matters.** It would be the first natural, non-Gaussian law in which the project's own transfer theorem is discharged with α ≠ 1. The fold is the same (m = 3); the contact law is different (r² from the Vandermonde |Δ|²); the exponent becomes 0.
  - This is exactly the "altered pair-count tilt" boundary prediction in experiments/universality/PROTOCOL.md:59-64.
  - It would show that CONTACT (pair intensity ∝ r) is a necessary admission input, not a consequence of the fold.
  - That goes beyond the toy control cubic_2 and the product-model phase diagram (§D2).
- **Feasibility.** Moderate, finite-dimensional.
  - The Weyl density and the Palm conditioning at a coincidence are explicit.
  - The heuristic mark law says a third eigenvalue near the pair has density ∝ |θ3−θ0|⁴, and κ ∝ |θ3−θ0|. So P(κ < ε) ≍ ε⁵, i.e. q = 4 > 0, and E[κ^{-1}] < ∞.
  - Elder selection near contact works as in Proposition 1(c).
  - The main work is §D hypotheses 3–4: uniform comparison and domination when a third eigenvalue approaches. Do N = 2 exactly first.
- **What counts as a result.** A Math- frontiers proof candidate with explicit, positive c_N, a stdlib checker (exact N = 2), and a coefficient-level comparison of predicted c_12 and c_128 with the existing CUE data.
- **Risk.**
  - Uniformity may fail near triple coincidences. The fallback is the cumulative statement (D2), which cannot be differentiated.
  - The N → ∞ (sine-kernel) limit needs amplitude normalization: Keating–Snaith log-normal spread of order ½ log N, from memory.
  - It says nothing about ζ without further conjectures.
- **Numerical grounding** (cutoff 0.01; cutoff 0.003 in parentheses):
  - CUE N=12: s = 0.980 [0.954, 1.007] (1.019 [0.973, 1.070]).
  - CUE N=128: 0.935 [0.920, 0.951] (0.971 [0.942, 1.001]). Convergence is slower because small-amplitude marks are over-represented 3.3× in short bars; the matched-tail difference is −0.034 [−0.052, −0.016].
  - Gap CDF slopes are 2.99–3.00 and the fold slope is 3.03.

2. Frozen zeta short-bar comparison (Conjecture Z), with the link to negative moments.
- **Question.** On fresh windows not yet inspected, is the amplitude-normalized short-bar density of ∫Z flat (exponent 0), with a coefficient matching CUE at N ≈ log(T/2π)? Candidate windows: t in [1e6, 1e7] by the existing Riemann–Siegel pipeline (about 1.6e7 zeros), or published zero tables at large height. Also: how does the unnormalized coefficient c(T) drift? It may grow like a power of log T, by analogy with negative discrete moments of ζ'(ρ) (Gonek/Hejhal and Hughes–Keating–O'Connell 2000, from memory, where the pole at k = −3/2 comes from the same r² repulsion).
- **Why it matters.** A deterministic arithmetic function lies outside every admission class, yet its short-bar law is set by its pair statistics. That makes it an external stress test of the separation between geometry and measure. It also quantifies mark (amplitude) effects in a natural setting, tying to the §D2 small-mark boundary and the amplitude counterexample (fold_structural_universality/PROOF.md:658-683).
- **Feasibility.** High numerically: the pipeline exists, 1.74M zeros took a few minutes, and zero counts were validated against mpmath.nzeros. Theory is conditional only. Follow PROTOCOL.md:53-64: freeze the windows, cutoffs, estimator, CUE reference and decision rule before generating the new data. The current windows are already inspected, so they stay exploratory.
- **What counts as a result.** A frozen prediction file plus a run, giving a pass/fail/indeterminate verdict on "exponent 0" and "coefficient within the CUE(N) band" at the stated heights. It is not a theorem and not RH evidence.
- **Risk.** GUE small-gap behaviour is unproved. log log T barely moves at accessible heights. There are finite-height arithmetic corrections (Bogomolny et al., from memory). Amplitude drift may dominate the coefficient. There is also a risk of being read as RH-relevant, so the wording needs care.
- **Numerical grounding.**
  - ζ [1e5, 1e6]: s = 0.980 [0.959, 1.001], density −0.020 [−0.041, +0.001].
  - ζ [1e4, 1e5]: s = 1.049 [0.968, 1.145].
  - Gaussian: 0.650 [0.641, 0.659].
  - ζ short bars below 0.01×median: 100% adjacent close pairs, 3.8% low-amplitude. The mean log10 shortfall −2.70 splits into gap −0.87 and peak/local amplitude −1.90.
  - Matched tail: s(area) − s(gap³) = 0.0004 [−0.0088, 0.0097].
  - Gap slope: 2.930 [2.882, 2.983].

3. Proposition 1 (fold of a backward-heat solution) and the vineyard birth law: write up and review.
- **Question.** Proposition 1 as drafted, plus one follow-up: in the project's own scale or heat-flow families, does a PDE likewise force folds to be transversal ("transversality for free")?
- **Why it matters.** It is the correct, bounded version of the note's main mathematical claim. It shows which part of ℓ ∝ r³ is generic (k = |F'''|/12) and which is dynamics-specific (r² = 8τ).
- **Feasibility.** Essentially done.
  - The proof is in the README draft.
  - Collisions were verified down to τ = 1e−12; errors scale like τ^{1.000000} and match κ_r and κ_ℓ to a relative 6e−6 to 3.5e−5.
  - kappa_check.py confirms the coefficients on a synthetic heat polynomial.
  - Only a nonauthor read is needed.
- **What counts as a result.** A reviewed author-side proposition, scientific effect NONE.
- **Risk.** Low. The novelty is essentially nil, so it must not be presented as new or RH-related.

4. Sensitivity of the barcode to normalization and marks (deterministic).
- **Question.** For F_w = ∫wH with smooth w > 0, which barcode features are invariant? The lemma to prove: for close pairs, ℓ_w = w(x0)ℓ(1+O(r)). So the exponent is invariant, while the coefficient becomes a w^{-1}-weighted Palm average. Long-bar pairings are not invariant, and a sign flip swaps the roles. When does a weight with huge dynamic range (the Ξ envelope varies by a factor of 1e−100 over a window) destroy any finite-window law?
- **Why it matters.** It is the deterministic counterpart of the project's amplitude counterexample and of the §D2 small-mark boundary. It informs how PROTOCOL observables are normalized.
- **Feasibility.** High: an elementary lemma plus data already in hand. Pairs shared with the Hardy normalization: Ξ weight 80/113 and 82/106; Ξ's actual sign 0/113 and 0/106; a mild e^{0.01t} weight 112/113 and 104/106.
- **What counts as a result.** A short lemma and a table in the packet.
- **Risk.** Low value if it stays a remark.

5. Optional, lowest priority: the correspondence between collisions and bars at t = 0.
- **Question.** For close pairs at t = 0, check ℓ(0) ≈ (8√2/3)|A||t_c|^{3/2} against the backward collision time t_c, at moderate heights, using Polymath15's effective approximation of H_t.
- **Why it matters.** Illustration only: it restates the Lehmer-pair mechanism in vineyard language.
- **Feasibility.** Moderate.
- **Risk.** It is known in substance and is accurate only for very close pairs. The low collisions are off by 33–128% at t = 0.