# Discovery / priority ledger — 2026-09-25

**Object:** DISCOVERY-PRIORITY-LEDGER-20260925-v1  
**Purpose:** preserve candidate discoveries, nearest prior art, novelty delta and proof/review state without turning a novelty search into theorem acceptance.  
**Scientific effect:** NONE. "Apparently novel" means no matching result was found in the scoped search; it is not a worldwide-first certificate.

**Amended 2026-09-27** at queue reconciliation, answering the nonauthor review in [PR #114 comment 5848575436](https://github.com/d6g8k5htny-coder/main/pull/114#issuecomment-5848575436): source bindings added per row, "current state" cells aligned with `STATUS.md` (2026-09-26), and the prior art that the scoped search missed recorded under each entry. The novelty deltas below are narrowed accordingly; none is widened.

## Candidate contributions

| ID | Candidate contribution | Novelty assessment | Current state |
|---|---|---|---|
| DP-01 | Near-diagonal H0 lifetime law for the 3D SIDE24 periodized Gaussian field: nu(ell) ~ c ell^(-1/3) | **Strongest apparent novelty** (delta narrowed; see prior art) | Coefficient D3 is ACCEPT-scoped (#65 closed); remainder D2 is ACCEPT-scoped at its O(1) scope (#67). The leading law **rests on DP-02, whose parent §§2–7 selection chain is AMEND** (#63 reopened). Not a landed theorem. |
| DP-02 | Quantitative cubic elder-selection defect / close-pair persistence-selection control: 1-p_r=O(r^3) | **Apparently distinctive quantitative theorem** | **AMEND.** `STATUS.md` D1: the quantitative §§2–7 selection chain is reopened for independent review with an additive §5 congruence erratum; the §8–§15 interfaces were accepted. |
| DP-03 | Finite-r Hermite-before-blow-up correction for coalescing maximum–saddle pins | **New technical discovery in this project** | Exact algebraic falsifier of old rows (bound below); the replacement continuum theorem is not thereby accepted. Math- PR9 AMEND; PR16 closed SUPERSEDED/CONSUMED (candidate imported at Math- `6e4085a`); PR19 merged 2026-09-25. |
| DP-04 | Three-critical-gradient ⇒ three-small-Hessian mechanism and fixed transverse O(k r^3) witness count | **High-potential original lemma/mechanism** | OpenAI author-side candidate; independent continuum review required. Source bound below. |
| DP-05 | Collision-order / pair-intensity universality: nu(ell) ~ ell^((alpha+1)/m - 1) | **Conjectural synthesis** (m=3 is the classical fold normal form; see prior art) | Not a theorem |
| DP-06 | P15 realized-family full-probability transformed-price budget with sharp factor 1/[3-log(3e-2)] | **Likely project-specific theorem** | **ACCEPT — scoped** (`STATUS.md` D6; #74 closed 2026-09-26; Math- `reviews/p15_full_price_nonauthor_20260926/REVIEW.md` records ACCEPT on all five steps at the realized-family, demand ≥ 2 scope). Demand one remains an obstruction. |

### Source bindings (step 2 of the protocol)

Each row binds the earliest recoverable exact source in the public Math- repository at commit `d6628da09384728992dcbe6e921cc28ba85aebb0` (an ancestor of Math- `main`; the pinned public reading checkout in `README.md`). Bytes and SHA-256 were read back through the GitHub API on 2026-09-27.

| ID | Math- path | Bytes | SHA-256 |
|---|---|---:|---|
| DP-01, DP-02 | `imports/lifetime_parent_20260925/UNIFORM_MATRIX_CAP_AND_LIFETIME.md` | 40261 | `9350ad6eaba6626b93c3dedeef9e2ff816e5cdf1c8318e85fb27499141c84bc7` |
| DP-02 (erratum) | `imports/lifetime_parent_20260925/ERRATUM_CONGRUENCE.md` at `5ed3b455b9a192487cedb32ad5dd8f2b90fbc1c1` | 1782 | `bad7ef609c4ad8c41ad6af562c1b6807921e19a9d556ed793ad1a0db6e202028` |
| DP-03 | `reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md` | 8054 | `8260e567e46cad8892442a5baf59cc0cbcbbeaf860edb1e234379e666e035bad` |
| DP-04 | `reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md` | 9902 | `f64c954245f17bcbc64a5ccf58a60689cf2a2fd85362f8321c0f64ff66584e36` |
| DP-05 | no source body; conjecture stated only in this ledger | — | — |
| DP-06 | `frontiers/full_price_20260924/PROOF.md` | 11352 | `87521901ca8e5405b4d1e47f1deb1cd0326affbd6f5967b53c4178590da993f9` |

A binding records where the statement lives; it is custody, not correctness.

## DP-01 — near-diagonal lifetime singularity

**Candidate claim.** For the exact 3D SIDE24 periodized Gaussian model and declared H0 maximum–saddle pairing convention,

    nu_{3,24}(ell) = c_{3,24} ell^(-1/3)(1+o(1)),  ell -> 0+.

Structural mechanism: cubic persistence splitting `ell ~ kappa r^3` combined with the close-pair Kac–Rice/contact radial law.

**Internal route.** `docs/RESEARCH_INDEX.md` routes the parent matrix-cap/lifetime proof, coefficient proof and quantitative remainder to review issues #63, #65, #67. Historical SIDE24 evidence remains provenance; use current source-bound review before citing the theorem.

**Nearest prior art found.**
- Bobrowski & Borman (2012), *Euler Integration of Gaussian Random Fields and Persistent Homology*: expected Euler-integral descriptor, not a near-diagonal lifetime density.
- Feldbrugge et al. (JCAP 2019, DOI 10.1088/1475-7516/2019/09/052): Betti numbers/persistence diagrams of 2D Gaussian fields and critical-point formalism; no located ell^(-1/3) near-diagonal lifetime law.
- Pranav, arXiv:2109.08721: statistical persistence-diagram/intensity analysis in 3D Gaussian fields; no located analytic near-diagonal power law.
- Cadiou et al., arXiv:2003.04413: coalescing critical events under changing smoothing scale; a different parameterized critical-event problem.
- Cammarota/Marinucci/Wigman critical-value/Kac–Rice work: critical point/extrema/saddle densities and covariance degeneracies, not a persistence-pair lifetime law.

**Prior art the scoped search missed (added 2026-09-27).**
- The close-pair radial law `r^alpha dr` is the short-range two-point function of critical points: Beliaev–Cammarota–Wigman, *Two point function for critical points of a random plane wave*, IMRN 2019(9) 2661, arXiv:1704.04943 (second factorial moment in a radius-r disc ∝ r^4, no repulsion, so unrestricted pair intensity ∝ r dr in d=2), and Azaïs–Delmas, arXiv:1911.02300, for isotropic fields in any d. DP-01 must state its α against these and explain why the #63 accounting (d−1)+3−(d+3)+2 = 1 gives α = 1 in every d.
- Chazal–Divol, *The density of expected persistence diagrams and its kernel based estimation*, JoCG 10(2) 2019, arXiv:1802.10457: density and near-diagonal behaviour of expected persistence diagrams in general. DP-01's delta is therefore "Gaussian sublevel-set law with explicit exponent and coefficient", not "near-diagonal intensity" as such.

**Novelty delta (narrowed).** Not "persistent homology of Gaussian fields", "Kac–Rice for critical points", "close-pair critical-point intensity" or "near-diagonal density of expected diagrams". The apparent novelty is the explicit analytic exponent −1/3 and coefficient for the H0 maximum–saddle lifetime law of the declared Gaussian model, together with the collision-to-persistence selection control (DP-02) that licenses it. That control is currently AMEND.

**Publication blockers.** Independent review of parent marked Kac–Rice/regression/normalizer/elder/domination (#63), coefficient (#65), remainder (#67); no reliance on falsified old D5 rows.

## DP-02 — cubic elder-selection defect

Candidate scoped statement: `1-p_r=O(r^3)` for the declared close maximum–saddle regime. The elder rule itself is classical and not ours. The scoped search did not locate a quantitative small-separation cubic failure probability connecting a local Gaussian critical pair to the actual elder-selected persistence pair.

**Novelty delta:** quantitative local-to-global persistence-selection control, not the elder rule.

## DP-03 — finite-r Hermite pins before singular blow-up

The 2026-09-25 exact six-pin cubic family shows

    f_x(ru,rv)/r^2
      = 6 kappa (u^2-1/4) + q u v + (c/2) v^2,

not the old `6 kappa u^2 + ...`. Corresponding midpoint corrections survive in the transverse derivative and cubic height coefficient.

**Internal evidence:** Math- PR16 source-bound oracle (closed SUPERSEDED/CONSUMED 2026-09-26; candidate imported at Math- `6e4085a`); PR9 returned AMEND REQUIRED; PR19 (OpenAI-authored finite-r Hermite repair candidate) merged 2026-09-25T21:27Z. The bound source is `reviews/d5_finite_r_hermite_repair_20260925/C6_REMAINDERS.md` (table above).

**Novelty delta:** finite-radius Hermite constraints and singular blow-up do not commute when a discarded pin residual has the same order as the blow-up denominator. Exact falsification is algebraic; replacement continuum theorem is not thereby accepted.

## DP-04 — three critical points force three small Hessians

On a fixed 2D scaled transverse chart away from axis/pins, the candidate mechanism is

    ||H_M|| + ||H_S|| + ||H_X|| <= C r M_3,

hence

    |det H_M det H_S det H_X| <= C r^6 M_3^6.

Combined with joint gradient-height density r^-6, spatial r^2, height kappa r^3 and full endpoint normalizer Z_r~r^2, this gives candidate count

    E N_j(rE) <= C kappa r^3 area(E).

**Internal source:** Math- `reviews/downstream_boundary_20260925/TRANSVERSE_BOUND_CANDIDATE.md` (bound in the table above; originally delivered via PR16), OpenAI-authored. Independent review interfaces: endpoint regression/normalizer; corrected joint density; deterministic Hessian suppression; conditional C3 moments; weighted marked Kac–Rice.

**Novelty delta:** scoped search did not locate this triple-Hessian suppression mechanism used to cancel the singular witness density in a persistence-pair problem. **Status: candidate, not accepted.**

## DP-05 — persistence-collision universality conjecture

If `ell ~ kappa r^m` and close-pair intensity is `r^alpha dr`, change of variables predicts

    nu(ell) ~ C ell^((alpha+1)/m - 1).

SIDE24 corresponds to m=3, alpha=1, giving -1/3. Treat as a conjectural synthesis until precise hypotheses are formulated and at least one non-SIDE24 model is proved.

**Prior art (added 2026-09-27).** The splitting `ell ~ kappa r^3` is the fold normal form: `x^3 − 3εx` has critical points at ±√ε, separation r = 2√ε and gap 4ε^{3/2} = r^3/2, so m = 3 is classical for any C^3 field and carries no novelty; the delta of DP-05 is the exponent α and the selection control only. A universality programme for random persistence already exists — Bobrowski–Skraba, arXiv:2207.03926 (experiments include Gaussian fields), Sci. Rep. 2023, and arXiv:2406.05553 — for a different statistic (the death/birth ratio); DP-05 must be stated against it.

## DP-06 — P15 full-price realized-family theorem

Within exact realized-family/palette hypotheses and demands >=2, the project derives a full independent-probability-range transformed-price budget with sharp factor

    rho_* = 1/[3-log(3e-2)] < 6/7.

The earlier 16/27 factor is stronger on its smaller domain; demand-one counterexamples remain.

Exact-expression and terminology searches found no matching theorem. This is lower-confidence novelty evidence because P15 terminology is project-specific. Review #74 closed 2026-09-26 with ACCEPT at the realized-family scope (`STATUS.md` D6); acceptance at scope is separate from novelty.

**Prior art (added 2026-09-27).** `frontiers/full_price_20260924/PROOF.md` §7 already records Gunby–He–Narayanan, arXiv:2112.08525, and Warnke, arXiv:2310.11662; carry them here. Any priority statement must also say why the cover-cost / −log μ_p shape is not a case of Talagrand's selector-process bounds (STOC 2010) or of Frankston–Kahn–Narayanan–Park, *Thresholds versus fractional expectation-thresholds*, Ann. Math. 194 (2021).

## Scoped literature reconnaissance

Search date: 2026-09-25, with the additions of 2026-09-27 noted per entry. Sources/records inspected include Bobrowski–Borman; Feldbrugge et al.; Pranav/Pranav et al.; Cadiou et al.; Cammarota/Marinucci/Wigman; general elder/persistence-pair literature; exact P15 constants. Queries targeted Gaussian random fields + persistence diagrams, small persistence/lifetime/near diagonal, maximum–saddle + Kac–Rice, ell^-1/3, critical-pair coalescence, and P15 constants.

**This does not prove worldwide priority.** It is not an exhaustive MathSciNet/Zentralblatt/dissertation/non-English/patent/unpublished-manuscript search. Require a second independent novelty audit before public "first" language.

## Priority-preservation protocol

1. Freeze exact statement/scope.
2. Bind earliest recoverable internal source/commit/Drive carrier and timestamp.
3. Preserve falsifiers and superseded versions.
4. Record nearest prior art and exact novelty delta.
5. Require distinct-lineage review of the current semantic digest.
6. Keep theorem status separate from novelty status.
7. For public release, use a timestamped preprint/repository release with reproducible source package once the chosen proof/review threshold is met.
8. Avoid public worldwide-first language until independent novelty review.

## Recommended paper decomposition

- **Paper A — Local collision theory / near-diagonal law:** DP-01 + DP-02 + corrected DP-03; DP-05 as outlook unless generalized.
- **Paper B — SIDE24 coefficient / finite-volume implementation:** coefficient, periodization/image correction, reproducibility, remainder.
- **Paper C / technical note — shrinking witness regions:** DP-04 plus axis/thin-belt/pin/intermediate/collision regions once independently closed.
- **P15 separate paper/note:** DP-06 belongs to a different combinatorial line.

Novelty and correctness are orthogonal.
