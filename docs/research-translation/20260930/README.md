# From the current results to a paper and a reproducible experiment

[Manuscript draft](MANUSCRIPT.md) · [Statement crosswalk](CROSSWALK.md) · [Refinement results](../../../experiments/periodic_h0/results/refinement8/RESULTS.md) · [Proof appendices](APPENDICES.md) · [Literature](LITERATURE.md) · [Research guide](../../RESEARCH_INDEX.md) · [Experiment protocol](EXPERIMENT.md) · [Exact sources](SOURCES.json)

**Research plan, 30 September 2026.** This is an editorial synthesis of specified sources and a proposal for the next scientific outputs. It does not supply a new proof, change a scientific register, or certify a publication-ready theorem. The source cut is Math- `fa2e9909d8cca40d38b7dc2eadda6766bd0788e1` and main `98a014f556116f39bf449fea859f5329ac322fc3`. Later developments require a successor assessment.

**Implementation update, 30 September 2026:** the [exposition draft](MANUSCRIPT.md) and [paragraph crosswalk](CROSSWALK.md) are now written, and the [fixed planar pilot](../../../experiments/periodic_h0/README.md) has run. Its coefficient comparison is **inconclusive at the tested resolutions** because the smallest bins drift under grid refinement. This advances the original plan without declaring a complete paper, new theorem, certified numerical window or human review. The source cut above remains the mathematical basis. The next iteration adds a [deterministic approximation argument](../../../experiments/periodic_h0/APPROXIMATION.md), [coupled refinement through 1024²](../../../experiments/periodic_h0/results/refinement8/RESULTS.md), [proof appendices](APPENDICES.md) and a [primary-source literature comparison](LITERATURE.md). [Confirmation readiness](../../../experiments/periodic_h0/CONFIRMATION_READINESS.md) keeps the evaluated-error and remainder gates visible.

The next useful external object is a readable account of short-lifetime **finite superlevel H₀ bars** for one precisely defined Gaussian field, accompanied by an experiment that measures the same quantity. Readers should be able to identify the model, theorem, evidence and limitations without reconstructing the federation. The detailed proofs and review history remain accessible behind that account.

## What the current record changes about the proposed research direction

An outside strategy note suggested a paper, standard persistence-library experiments and human review. Those are useful directions, but its proposed gap list mixed historical snapshots with current results. The corrections below matter before assigning new mathematical work.

| Proposed diagnosis | Source-backed correction | Consequence for the next work |
|---|---|---|
| The leading law must first be made conditional on a new blanket pairing hypothesis. | The parent explicitly defines the **global ordinary elder selector** and states compact and unrestricted leading laws. The current D1 reconciliation records acceptance at its stated consumption scope, with required erratum, Borel repair, W1 and embedding restriction. | Assemble that actual chain. Do not invent a new hypothesis or remove a required existing premise. An unresolved statement-alignment question belongs in a specific manuscript annotation. |
| D5 pin/intermediate bridges and higher-dimensional first moments are still missing everywhere. | The later fixed-dimensional packet gives P_d, C_d, I_d and G_d; it landed through [Math #141](https://github.com/d6g8k5htny-coder/Math-/pull/141) with source-bound continuum reviews. | Preserve fixed d and L, compact birth/gap marks with k bounded away from zero, the original determinant tilt and sufficiently small radius. Numerical constants and an all-height global count do not follow. |
| C6 only gives r³ log(1/r), and sharp r³ remains open. | The later Palm route in [Math #145](https://github.com/d6g8k5htny-coder/Math-/pull/145) gives E[(N)_q] ≤ C_q r³ for each fixed q≥2 and sharp second-factorial/event order r³, using its bound inputs. | Do not repeat the older logarithmic task. Higher factorial **lower** bounds, uniformity as k→0 or d→∞, and evaluated constants are separate questions. |
| No actual local-to-global elder identification has been delivered. | [Math #170](https://github.com/d6g8k5htny-coder/Math-/pull/170), merged in this source cut, contains the planar actual partner/failure law and compact rejected-candidate coefficient, with scoped A/B/C and composition reviews. | Preserve its planar and parameter restrictions and parent interfaces. Its actual partner law is different from the count of two additional saddle witnesses. Higher-dimensional and unrestricted refinements require their own review. |
| The remainder still lacks the listed second-provider read. | [Math #186](https://github.com/d6g8k5htny-coder/Math-/pull/186) records a full-depth R1–R6 review of D2, with repaired evidence controls. | Retain its existential O(1) meaning. It evaluates neither an error constant nor a usable lifetime cutoff. Model-provider diversity does not create independent human review. |

Read the [D1 reconciliation](https://github.com/d6g8k5htny-coder/Math-/blob/fa2e9909d8cca40d38b7dc2eadda6766bd0788e1/reviews/d1_chain_reconciliation_20260928/RECONCILIATION.md) with each consumed interface. Frozen source headers can retain their original candidate disposition while later exact-source reviews record acceptance; a merge alone establishes neither. The source manifest identifies the proof and review records separately.

## The first manuscript: one model, one quantity, one chain

A working title is **Short-lifetime H₀ persistence of a periodized Gaussian field**. Keep other combinatorial or cosmological programs as separate publications. The title is a working description, not a novelty claim or promise of impact.

Start with d=2, L=24 for the exposition and experiment. A d=3 statement may appear only with an explicitly checked dimension-specific dependency chain; a d=2 argument cannot silently certify it. The parent model is the centered, variance-one field on Rᵈ/(24 Zᵈ) with

```
K_24(z) = Σ[n∈Zᵈ] exp(-|z+24n|²/2) / Σ[n∈Zᵈ] exp(-|24n|²/2).
```

The unrestricted quantity is the expected number of finite ordinary superlevel H₀ bars per unit spatial volume and per unit lifetime. Exclude the essential global-maximum class. Distinguish this from all ordered typed maximum/saddle candidates, a compact birth/gap subpopulation, and a probability distribution obtained by normalizing by the number of bars.

The assembly target is the parent's leading law and D2's bounded remainder, with the exact SIDE24 coefficient expression. The numerical enclosures apply to that expression; their interpretation as a persistence intensity consumes the parent chain. Do not copy the outside note's combined “compact marks + unrestricted coefficient” formulation: c_(B,K) and c_(d,24) refer to different populations.

| Manuscript section | Existing material to assemble | Question the reader must be able to answer |
|---|---|---|
| Model and observable | Parent §1; reconciliation reading rule | What field, index convention, intensity measure and bar convention are used? |
| Local geometry and regression | Marked-cylinder cap; parent §§2–9 with repairs | Which deterministic cap implication and conditional Gaussian law are consumed? |
| Selection and leading law | Parent Theorems A/B/C and its Kac–Rice ledger | Where is the global elder event used, and which estimates remain compact-mark estimates? |
| Error term | D2 Theorem R and its exact-source reviews | What is existential, and why is no finite numerical accuracy window supplied? |
| Evaluated coefficient | SIDE24 PROOF and exact replay | Which expression is enclosed, under what parent interpretation? |
| Numerical comparison | The separate [protocol](EXPERIMENT.md) | Does the experiment use the same model, measure, units and essential-bar exclusion? |
| Limits and extensions | Scoped D5/C6/elder sources | Which stronger rate, uniformity, dimension or numerical certificate is still a separate obligation? |

Write the short exposition first, then attach full proofs or precise appendices. A page count is an editorial target, not a substitute for a complete argument. Keep a paragraph-by-paragraph statement crosswalk during assembly: source statement, exact version, hypotheses, repairs, review scope and unresolved alignment question. The successor [draft](MANUSCRIPT.md) and [crosswalk](CROSSWALK.md) now implement this assembly step; complete proof appendices and theorem-level literature comparison remain.

## Next mathematical work, after accounting for existing results

1. **Make the existential error usable.** Seek explicit constants and a justified lifetime window for the exact finite-torus observable. The evaluated SIDE24 coefficient alone does not make a finite sample or finite lifetime lie in the asymptotic regime. A negative result establishing an impractically small sufficient window is still useful.
2. **Complete the precise refined-selector obligations.** Inspect the active higher-dimensional and unrestricted successor PRs before claiming a scope. At this audit, [Math #175](https://github.com/d6g8k5htny-coder/Math-/pull/175) is a separate higher-dimensional actual-partner candidate. Do not treat an open PR as an analytic failure or take over its author's work. Compact candidate-minus-selected rates must not be exported to unrestricted marks or counted again as replacement bars.
3. **Connect computation to the continuum.** Establish or quantify spectral truncation and spatial interpolation errors sufficient for the observable under study. The experiment below records empirical refinement without representing it as a proof of these errors.
4. **Extend the scientific model only by new arguments.** Other covariances, higher homology and joint infinite-volume/small-lifetime limits are worthwhile extensions. A negligible SIDE24 coefficient correction is not an interchange-of-limits theorem. Prioritize an extension when its exact dependency and intended observable are clear.

The historical RN/24-jet certificate lane retains its own obligations. Qualitative count theorems do not complete a numerical annulus partition or evaluate the required weighted integral. Conversely, an unfinished historical certificate does not erase a separately source-reviewed qualitative theorem.

## Literature and external review

The manuscript should explain what it adds relative to related observables, without asserting “first” from a bounded search. The following primary-source starting points were checked for this plan; this is not an exhaustive novelty clearance.

- [Adler, Bobrowski, Borman, Subag and Weinberger (2010)](https://arxiv.org/abs/1003.1001) surveys persistence for random fields and complexes. Use it to place the random-field question, not to equate field models with random point-cloud complexes.
- [Bobrowski and Borman, Euler Integration of Gaussian Random Fields and Persistent Homology](https://arxiv.org/abs/1003.5175) connects an expected Euler integral with Gaussian geometry. That observable differs from a lifetime intensity near the diagonal.
- [Chazal and Divol, The density of expected persistence diagrams and its kernel based estimation](https://arxiv.org/abs/1802.10457) treats expected diagram densities under specified filtration assumptions. Cite the applicable theorem if consuming it; density existence alone supplies no short-lifetime exponent or coefficient.
- [Feldbrugge et al., Stochastic Homology of Gaussian vs. non-Gaussian Random Fields](https://arxiv.org/abs/1908.01619), especially §§4–5, uses Morse/graph connections, a fitted Betti-curve model and persistence-diagram integral expressions. Compare the pairing observable and normalization explicitly.
- [Wu, Kim and Rinaldo, On the estimation of persistence intensity functions and linear representations of persistence diagrams](https://arxiv.org/abs/2310.11982) distinguishes intensity and normalized density and studies estimators from independent diagrams. Its distinction is directly relevant to experimental reporting; its statistical guarantees need their own assumption check.

A human referee packet should contain the short manuscript, source/review crosswalk, one reproducible experiment, and a small list of exact unresolved questions. A useful first request is to check the observable and the local-to-global composition, then the coefficient normalization. Model reviews, including provider-distinct ones, remain attributed as model reviews. No human review, agreement, journal acceptance, DOI or submission has occurred merely because this plan exists.

## Delivery order and completion criteria

| Output | Completion means | Current disposition |
|---|---|---|
| Manuscript assembly | A specialist can follow every theorem to its complete source chain; missing alignments remain explicit | Exposition draft and paragraph crosswalk supplied; full paper and proof appendices remain |
| Planar experiment | Pinned generator and persistence code, deterministic controls, independent realization replicates and resolution/truncation comparisons | Executed fixed pilot with complete retained observations; grid-sensitive and inconclusive; no fit or certified comparison |
| Literature comparison | Checked theorem-level comparison, bibliography and carefully limited novelty language | Primary-source starting set supplied; full comparison remains |
| External assessment | Actual human reader, documented questions and responses, revisions attributed | Not arranged or completed |
| Extensions | Exact statement, author, source dependencies, falsifiers and scoped review | Follow existing peer PRs before opening duplicate work |

Read the draft and pilot through the links above. Their technical review does not complete the remaining publication, continuum-error or human-review obligations.
