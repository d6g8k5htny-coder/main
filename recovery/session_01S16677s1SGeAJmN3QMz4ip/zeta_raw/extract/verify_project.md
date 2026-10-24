# SUMMARY
The note's local mathematics matches what the project actually states, and its exponent test is already in the repo. Its geometry-versus-probability question is already the project's own stated research question. Neither repo has any prior material on zeta, Epstein, GUE/CUE, de Bruijn–Newman or Lehmer pairs; the only exception is a retired 2025 chapter that is wrong in places and must not be cited. The theta link exists only as Poisson summation of the periodized covariance at a fixed torus side L. It is proved in the periodized proof, numerically checked in history carriers, and formalized in an unreviewed Math- Lean scratch. I flagged the note's errors and imprecisions with fixes, and checked the main identities numerically with mpmath 1.4.1. I also mapped where an exploratory packet could go: no exploratory label exists, all recent PRs are unlabeled draft PRs, coordination happens on main#307, and five open drafts already edit experiments/universality/README.md.

# DETAILS
Bases: main worktree at 03acb7be (W = .../scratchpad/wt-guide) and Math- at 79cb314 (M = .../scratchpad/math-src). Both were read only; I edited neither. Paths below are relative to the repo root. My only files are in .../scratchpad/zeta/project-map/.

## A. What the project actually states (exact quotes, file:line)

**1. Fold and height law: constants and variables**
- `experiments/universality/fold_controls.py:28-31`: "Gap of x**3/3-h**2*x at critical points +/-h: 4*h**3/3. The actual separation is 2*h, hence gap=separation**3/6."
- `fold_structural_universality/PROOF.md`:
  - `:36`: "ell=F(M)-F(S)>0" (M is the maximum, S the merging critical point of index d-1).
  - `:191-193`: "For heights b,b-k r^3 and zero gradients, k>0, the exact target is v_r=(b-k r^3/2,-k r^2,0,12k,...)". Here r is the geodesic separation, with M=exp_x(-ru/2) and S=exp_x(ru/2) (`:175`).
  - `:244-248`: in the contact limit, `F_uuu=12k`. So ell = k r^3 with k = F_uuu/12.
- The note's ell ≈ |A| r^3/6 for F ≈ A(x^3/3 - 2τx) is the same law: F''' = 2A, so k = |A|/6. With A=1 it is fold_controls' separation^3/6 exactly.
- `docs/research-translation/20260930/PERSISTENCE_BRIDGE.md:94-99` has the exact (not approximate) identity: phi(c)-phi(a) = -(1/2)∫(x-a)(c-x)phi_xxx dx, "The kernel integral is r³/6".

**2. Lifetime-density exponent: dimensions and hypotheses**
- Structural theorem (conditional), `fold_structural_universality/PROOF.md`:
  - `:11`: "a conditional structural theorem with explicit field-admission interfaces".
  - `:29-30`: on a compact manifold without boundary, "of dimension d>=1".
  - `:60-65`: "Under COUNT, CONTACT, SELECT, ENVELOPE and FAR ... nu_bar(ell) ~ C ell^(-1/3) ... E N_bar((0,t])/V_X ~ (3/2) C t^(2/3)".
  - `:56-58`: C must be strictly positive.
  - `:69-70`: "The exponent is independent of d once these interfaces hold. It is not an unconditional dimension-independent admission result."
- Proved for one law (adapter), `periodized_gaussian_h0/PROOF.md:54-61`: "For every fixed d>=2,L>0 ... nu_bar(ell) ~ c_(d,L) ell^(-1/3)". This is existential, with "no ... usable lifetime window or numerical digits" (`:65-66`).
- Headline, `docs/research-translation/20260930/MANUSCRIPT.md:7` and `:42-47`: d=2, L=24, `ν(ℓ) = c₂,₂₄ ℓ^(−1/3) + O(1)`, with 0.07340691930603427103 < c₂,₂₄ < 0.07340691930603427104.
- One dimension, Math- `frontiers/d1_third_order_law_20261001/README.md:13-19` (circle, Theorem 1D): ν(ℓ) = C₀ℓ^{−1/3} + C₁ℓ^{1/4} + 2B₂ℓ^{1/3} + O(ℓ^{1/2}). Mechanism at `:39-40`: "a maximum and the adjacent minimum at separation t, gap ≈ f'''t³/12".
- The line version is Math- `frontiers/d1_line_20261004/README.md:13-30`, under spectral hypotheses (R1)-(R3).
- Both 1D results are author-side proof candidates with "Scientific effect: NONE"; per their READMEs they are merged and passed nonauthor reads.
- Density exponent −1/3 and cumulative exponent 2/3 are distinct (`PROTOCOL.md:17-18`).

**3. Contact (pair) intensity at short separation: proportional to r dr**
- `fold_structural_universality/PROOF.md:390-400` gives the four factors: polar volume r^(d-1) dr, height change r^3, observation density 12 r^(-(d+3)), determinant weight r^2. "Their product is exactly J r B_r dr db dk ... (d-1)+3-(d+3)+2=1."
- `:402-404`: "no exponent can be inferred while omitting the transformed contact-density order or the determinant-product order."
- `docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md:76-87`: "= 12 r ... A cubic fold supplies the gap exponent; the radial counting law supplies the rest. A change of sampling, determinant tilt or selection can change the persistence exponent without changing the fold."

**4. Periodized covariance, weights q_n and normalizer** (`periodized_gaussian_h0/PROOF.md`)
- `:35-36`: K_L(z)=Σ_j exp(-|z+Lj|^2/2)/S_(d,L), with S_(d,L)=Σ_j exp(-|Lj|^2/2).
- `:44-47`: q_n=exp(-ω^2|n|^2/2)/Z_(d,L), with Z_(d,L)=Σ_j exp(-ω^2|j|^2/2) and ω=2π/L. This matches the note's formula exactly.
- `:87-102` derives q_n "without assuming a truncated Poisson formula". The derivation forces Z_(d,L) = (L/√(2π))^d S_(d,L), which is Jacobi's Θ(u)=u^{-1/2}Θ(1/u) at u=2π/L².
- `:82-83`: "No joint unbounded-d, variable-L, cutoff or model-family uniformity is asserted."
- `MANUSCRIPT.md:11-22` gives the same law at L=24.

**5. The five admission interfaces**
- COUNT, CONTACT, SELECT, ENVELOPE and FAR are tabulated, with falsifiers, at `fold_structural_universality/PROOF.md:42-48`.
- The positive-coefficient requirement is at `:56-58`.
- The periodized adapter discharges them for its law (`periodized_gaussian_h0/PROOF.md:70-78`).

**6. Counterexamples separating geometry from probability (the note's key question)**
- Toy controls:
  - `fold_controls.py:3-4`: "If dI=C*r**alpha dr and ell=k*r**p, the density power is (alpha+1)/p-1. The cubic fold alone supplies p, not alpha."
  - `test_fold_controls.py:8-12` asserts density exponents −2/3, −1/3 and 0 for radial powers 0, 1 and 2.
  - The suite README (`:178-181`) says the controls give "-2/3, -1/3, 0 or +1".
  - The note's test (gap ~ r² dr gives ℓ^0) is literally case `cubic_2`. I re-ran it and the output is saved.
- Covariance-only failures, `fold_structural_universality/PROOF.md`:
  - `:640-656`: an atomic coefficient law with the same covariance gives an atomic lifetime measure.
  - `:658-683`: the small-amplitude mixture F=RH, P(S≤s)=s^α with α<2/3, keeps the covariance but E N_F((0,t]) ≥ c t^α. "The small-amplitude boundary is not controlled by covariance or positive moments."
- Order taxonomy, `:693-741`: e=(γ+1-a+w)/m. `:714-715`: "The regular cubic contract has m=3, a=d+3 and w=2, giving s=1 and e=-1/3. The cubic order by itself supplies only m."
- Elder-selection shortcut rejected: `PERSISTENCE_BRIDGE.md:404-493` shows a fold pair need not be the actual persistence pair.
- Program target, `docs/PERSISTENCE_UNIVERSALITY_PROGRAM.md:3-6`: "a reusable theorem connecting a local singularity, the probability law transverse to its discriminant, and the intensity of actual persistence bars". This is the note's §5 question.
- In flight on main, all open drafts not at base 03acb7be:
  - #334: covariance-matched amplitude crossover.
  - #336: quartic germ where "Flat parameter density therefore realizes density power -1/2, while gamma=4/3 realizes -1/3".
  - #340: fold-edge ℓ^(-1/3) versus tip ℓ^(1/4).
  - #330 and #331 (spectral Gaussian admission, higher-homology folds).
  - All five edit `experiments/universality/README.md`.

## B. Repo searches (word-boundary, case-sensitive; audits/ and *.json excluded)

**main (315 raw hits).** The real hits:
- **Poisson summation** of the periodized covariance:
  - `experiments/periodic_h0/GAUSSIAN_TAIL.md:44-50` ("Fourier/Poisson identity ... DLMF §1.8(iv)").
  - `periodized_gaussian_h0/PROOF.md:87`.
  - `formal/GLOSSARY.md:48` ("Poisson-summation image terms n ≠ 0").
  - `history/carrier20_recovery_20260928/public_carriers/KIMI-DER-027a_farfield_envelope.md:30,146` ("Poisson summation (exact identity)") and `KIMI-DER-027c.md:50`.
  - `verify_thm023_v1_1_v1.py:117-122` (mpmath "P2 Poisson agreement: spectral vs image < 1e-50").
  - `incoming/.../H3_closure/h3_band_floor.py.txt:68` ("Poisson–Hermite identity").
- **theta as a theta function:**
  - `formal/GLOSSARY.md:37` ("normalised theta-function periodisation").
  - `experiments/periodic_h0/GAUSSIAN_TAIL.md:123` ("two-sided theta remainder") and `CONFIRMATION_READINESS.md:21`.
  - `experiments/periodic_h0/spectral_shrinking_bins/PROOF.md:145,343`.
  - Not theta functions: "theta-polynomial" (`periodized_gaussian_h0/PROOF.md:13,386`) is the trigonometric θ_a(z)=Σ[1-cos ω(z_j-a_j)] (`:135`). All other theta hits are angles.
- **zeta, Riemann, GUE, Newman, Lehmer** appear only in the retired `history/2025/` material.
  - `history/2025/README.md:3` says it is "obsolete and wrong".
  - `body.original.txt:582-628` is "Arithmetic, Compression, and the Riemann Hypothesis". Its "Theorem 13.1 (Critical Line Optimality)" (`:609`) is the tautology that Σ(σ_k−1/2)² is minimized at σ_k=1/2.
  - `:941` says "Odlyzko (1987) computed 8×10⁶ zeros near the 10²⁰-th zero". From memory, the 1987 Math. Comp. paper used zeros near number 10¹²; this needs checking.
  - Its tables are internally inconsistent: §13.5 says "Gram (1903)" for height 10⁶, while appendix G gives 15 zeros.
  - Do not cite any of it as prior project work.
- Other hits are non-matches: zeta and Riemann as variable names, Riemann sums and Riemann–Lebesgue, and "Poisson" meaning point processes (`iid_short_bar_process`, PROGRAM `:234-245`). Epstein and CUE have no hits.

**Math- (1354 raw hits, mostly theta as an angle).** The real hits:
- Poisson summation is used as a standard step in many packets, for example `frontiers/c6_palm_route_20260929/PROOF.md:143`, `frontiers/c8_elder_failure_coefficient_20261001/NOTE.md:275` and `frontiers/side24_periodization_20260929/PROOF.md:21`.
- `reviews/side24_d2_certified_small_L_claude_20261005/NOTE.md:53-54` computes moments by image sums θ_L^{(j)}(0) and by a "Poisson-dual spectral sum".
- **Lean (unreviewed scratch, "credit 0; sci NONE"):** `reviews/w3_side24_source_proposal_20261008/W3Scratch/Side24Law.lean`.
  - `:2` imports `Mathlib.NumberTheory.ModularForms.JacobiTheta.TwoVariable`; the README lists `Mathlib.Analysis.SpecialFunctions.Gaussian.PoissonSummation` as well.
  - `:240-250` `theta_zero_eq_dual` proves θ_L(0)=(L²/(2π))^{-1/2}·Σ_k e^{-(2πk/L)²/2} via Mathlib's `Real.tsum_exp_neg_mul_int_sq`.
  - `Side24Poisson.lean:28-33` `theta_eq_cos_series` covers general t.
  - From memory, Mathlib proves ζ's functional equation through the same Jacobi-theta and Mellin machinery. If so, the formal "theta bridge" already exists as infrastructure; it is not new mathematics.
- No zeta-function, Riemann Hypothesis, Epstein, GUE/CUE, Newman, Lehmer or Jacobi-theta (beyond the Lean imports) material. Other hits are variable names, Riemann sums or Riemann–Lebesgue, and Jacobi rotations or Jacobi's determinant formula.

## C. The note checked against the project and classical facts

Correct:
- The H_t definition and the backward heat equation ∂_tH=-∂_z²H.
- The local collision form H≈A(x²−2τ) with zeros ±√(2τ) for τ>0, i.e. forward time.
- ell≈|A|r³/6.
- The Mellin identity for Re s>1.
- R₂~(π²/3)r².
- The r² → ℓ^0 pushforward.
- RH ⇔ Λ≤0, and Λ≥0 by Rodgers–Tao.

Numerical checks (saved in checks.out):
- H_0(z)=ξ(1/2+iz/2)/8 at z=0, 10 and 2γ₁.
- Z_{1,L}=(L/√(2π))S_{1,L}=Θ(2π/L²) for L=1/2, 1, 3, 24, agreeing to 40 digits.
- The 1D normalized image sum equals ϑ₃(πt/L,q)/ϑ₃(0,q) with q=e^{−2π²/L²}.

Errors and imprecisions to fix:
1. **"Real zeros can approach, collide."** Under forward heat time real zeros repel (z_k' = 2Σ1/(z_k−z_j)). Collisions occur only going backward. For t>Λ zeros are real and simple (Csordas–Smith–Varga 1994, from memory; my search did not confirm the paper itself). So the F_t vineyard has no diagonal events for t>Λ. Under RH, the fold events sit at t<0, accumulating at 0⁻.
   - Ki–Kim–Lee: for every t>0 all but finitely many zeros of H_t are real and simple. This is confirmed only via a conference abstract.
   - That result suggests "Λ>0 ⇒ a multiple real zero at some t∈(0,Λ]". Treat that as unverified until checked against KKL and CSV.
2. **Attribution.** Θ(u)=u^{-1/2}Θ(1/u) ⇒ ξ(s)=ξ(1−s) is Riemann 1859 (his second proof). Mumford's Tata Lectures is a textbook source, not the origin.
3. **"Same Gaussian-lattice construction up to normalization and scale."** The precise statement:
   - q_n = e^{−πu|n|²}/Θ(u)^d with u=2π/L².
   - K_L(z)=∏_i ϑ₃(πz_i/L,q)/ϑ₃(0,q).
   - The project's Fourier/image-sum identity is exactly Jacobi's transformation.
   - The project uses a single u (fixed L). ζ needs the Mellin integral over all u, i.e. all L, and the project disclaims variable-L results (`periodized_gaussian_h0/PROOF.md:82-83`).
4. **Epstein caution (from memory).** For the project's lattice Z^d, the Mellin transform of Θ(u)^d−1 gives Σ r_d(n)n^{-s}.
   - d=2 gives 4ζ(s)L(s,χ₋₄).
   - d=4 gives 8(1−4^{1−s})ζ(s)ζ(s−1). Its ζ-zeros sit at Re s=1/2 and 3/2 (under RH), not on its symmetry line Re s=1.
   - Davenport–Heilbronn (1936): some Epstein zetas have zeros in Re s>1.
   - So theta/lattice structure alone does not select a critical line. This strengthens the note's "not a proof bridge".
5. **The note's "central theorem".** It conflates the conditional structural theorem (d≥1, five interfaces) with the proved adapters: periodized Gaussian at fixed d≥2 and L; finite Gaussian laws at d=2; and the Math- Theorem 1D on the circle and line.
6. **Persistence of F_t=∫H_t is not intrinsic to the zeros.** Multiplying H by a positive weight keeps the critical points but changes lifetimes and possibly elder pairings. |H_0| decays roughly like e^{−π|z|/8}, so raw lifetimes on [T,2T] are dominated by the envelope. A local normalization (for example ξ over its envelope, or Hardy's Z) is a modelling choice. The project's amplitude counterexample (`fold_structural_universality/PROOF.md:658-683`) shows such choices can change exponents.
7. **Setting mismatch.** Project theorems concern expected measures of random laws on closed manifolds (`fold_structural_universality/PROOF.md:29-30`). ζ is deterministic and the proposal uses finite intervals, so this needs a spatially averaged empirical measure plus boundary handling. No project theorem covers that.

New but elementary observations (checked locally, not literature-searched):
- **(a) Exact lifetime between consecutive real zeros a<c.**
  ∫_a^c H_t = −½∫(x−a)(c−x)H_t'' = ½∫(x−a)(c−x)∂_tH_t.
  This is the PERSISTENCE_BRIDGE (3) identity combined with the heat equation. Both sides equal −6.31861166707933e−4 between 2γ₁ and 2γ₂.
- **(b) Vineyard birth law.** A collision at t₀ creates an F_t bar with ℓ ≈ (8√2/3)|A|(t−t₀)^{3/2}.
- **(c) Natural persistence object for H itself.** RH ⇒ Ξ is in the Laguerre–Pólya class, which gives the Laguerre inequality Ξ'²−ΞΞ''≥0 (Pólya; Csordas–Varga, from memory). That means no positive local minimum or negative local maximum, i.e. every finite superlevel H0 bar of Ξ dies at a level ≤0.
  - In the collision model the critical value −2Aτ changes sign at the collision.
  - This restates classical content; it is not progress.
- **(d) A Gaussian null model.** H_t(z)/H_t(0) is positive definite for every real t, because Φ(u)>0 termwise (2π>3) and Φ is even.
  - So a stationary Gaussian process with that covariance exists. The Math- line theorem's (R1)-(R3) appear to hold, which would give ℓ^{-1/3} with constants made from ξ's derivatives at 1/2.
  - Caveat: its zero density is constant, so it encodes nothing about ζ's zeros. It is only a covariance-matched null model.

Citations checked by web search (direct fetch was blocked, so arXiv and Wikipedia pages could not be opened):
- Polymath Λ≤0.22 (Res. Math. Sci. 2019).
- Platt–Trudgian Λ≤0.2 (Bull. LMS 53 (2021) 792–797).
- Saouter–Gourdon–Demichel, Math. Comp. 80 (2011).
- Csordas–Odlyzko–Smith–Varga, ETNA 1 (1993) 104–111, Λ≥−5.895e−9.
- CSV 1994, Λ≥−4.379e−6.
- Everything else above is from memory.

## D. Where an exploratory packet would fit

**main, `experiments/`:**
- `experiments/periodic_h0/` pins numpy==2.5.3 and gudhi==3.13.0 in `requirements.txt`. Its CI `.github/workflows/periodic-h0-pilot.yml` runs on Python 3.12 and pip-installs them.
- `experiments/universality/` is stdlib only: `python3 -B -E -S` and `-O` unittest discovery, exact Fraction controls, `<name>/PROOF.md` plus `REVIEW.md`, and a suite README.
  - Its CI `universality-controls.yml:5-8` filters on `experiments/universality/**`. Under -S, mpmath is not importable.
  - Discovery does not recurse into subdirectories without `__init__.py`.
- Any change under `docs/**` triggers `navigation.yml` (`tools/navigation_check.py --verify-public`).
- A new `experiments/<dir>/` has no CI unless a workflow is added.

**Admission rules in `experiments/universality/PROTOCOL.md`:**
- Exploratory execution is not confirmation; "Do not fill the gaps with fitted coefficients or a favorable pilot slope" (`:53-55`).
- Boundary predictions must be frozen and should include "altered pair-count tilt" (`:59-64`). Quadratic (GUE-type) repulsion is exactly such a tilt.
- README `:13-14`: running a model does not establish theorem inputs.
- PROGRAM `:5-8`: lemmas "earn priority only when they discharge a named interface, distinguish competing mechanisms, or enable a predeclared falsification test".

**Math-, `frontiers/<topic>_<YYYYMMDD>/`:**
- Contents: `README.md` ("Author-side ... Scientific effect: NONE. Nonauthor review required.", object id CL-...-vN), `PROOF.md` or `NOTE.md`, `SOURCES.json` (schema 1, `scientific_effect` NONE, files with bytes/sha256/git_blob, consumed/cited_only, mutants) and `RESULTS.json`.
- A stdlib checker whose output must be byte-identical under `-B -S` and `-B -O -S`.
- Mutants must exit 1, and an unknown mutant must exit 2.
- Each packet has its own `.github/workflows/<name>.yml` with path filters; see `soft-rejected-pairs.yml`.
- `AGENTS.md:13`: "Calculations here use the Python standard library only". An mpmath zeta-zero computation therefore does not fit Math- outside the Lean `formal/` lane.

**Board and labels:**
- main#307, "Agent Message Board (continued from #229)", is open with 1020 comments and no labels. It is for claims, pickups, routes, verdicts and releases, and is explicitly "not a new ownership, priority or scientific-status register". It points to task queue #275 (now titled "Closure evidence: adopt the useful parts of the 10/10 closure proposal (decision record)"; labels enhancement, math-review).
- No exploratory label exists. main has 25 labels, including math, mathematics, math-review, docs, results-for-review and do-not-merge. Math- has 17.
- Recent main PRs #328–#368 are unlabeled apart from dependabot; most math work is draft PRs. So the existing convention for non-status work is an unlabeled draft PR, stating "Scientific effect: NONE", with a scope claim on #307.
- Conflicts to avoid: open drafts #330, #331, #334, #336 and #340 all edit `experiments/universality/README.md`, and another job is editing `docs/RESEARCH_INDEX.md`.

Files: `checks.py` (mpmath checks), `checks.out` (results) and `fold_controls.out` (the project's own control output re-run).

# ARTIFACTS
[
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/project-map/checks.py",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/project-map/checks.out",
 "/tmp/claude-0/-home-user/77b2421d-c423-5f03-8502-d31f33b60ec1/scratchpad/zeta/project-map/fold_controls.out"
]