Every number here is computed from `REGISTER.json` by the rendering script and re-asserted on each render; the
labels are the four of the lane and nothing else. A row at `none` says only that no Lean sentence for the claim
exists yet; it is not a verdict on the mathematics, and a row at `specified` is a precise question, not evidence.

### 5.1 Counts

{{COUNTS_TABLE}}

By claim kind:

{{KIND_TABLE}}

The 197 `specified` statement entries point at **150 distinct declarations** of the packet (141 `def … : Prop`,
9 interface `structure`s): `Field` 12, `Lifetime` 24, `Side24` 28, `RN` 28, `P15` 58. A declaration is listed
beneath every claim it serves, which is why there are more entries than declarations (`RN.FixedRemoteMeanMeasureA2`,
for instance, sits beneath the landing claim, the graph node, the region and the D5 component that consume Theorem A).
The modules' other 123 declarations are non-Prop definitions (`cRef`, `rhoStar`, `thetaSum`, `coverCost`, …) that the
Props are stated about; they are not register entries. The 59 `proved` entries are the theorems of the `main`
core-Lean package: 31 SIDE24 targets at origin/main `f6deeba7` (`Ledger` 17, `ConeMoments` 10, `Endpoints` 4), for
which the hosted `formal-verification.yml` run is the kernel evidence, and 28 P15 targets (`PriceBoundary` 6,
`RealizedCovers` 12, `FullPrice` 10) that exist only in this programme's uncommitted worktree; their sole receipt is a
local `--run-lean` whose `checked_commit` is the dirty tree's HEAD `f6deeba7`, which is not evidence about any commit.
Each of the 59 is a finite integer or cross-multiplied rational identity (`decide`, one conjunct by core
`Nat.pow_pos`), listed beneath a claim as its arithmetic skeleton; none states a research claim, which is why every
claim they sit under stays `specified` or `none`.

The **33 `specified` claims** are:

{{SPECIFIED_LIST}}

Existing Lean coverage outside the packet is inventoried as seven `existing_lean_package` rows, all `none` at
claim level: `main formal/` (59 theorems as above), Math- `formal/ResearchFormalCoreR1` (63 theorems; the only
alignment record on file, main `reviews/lean_probability_alignment_20261005`, covers 20 targets of Math- #272 at
`27016790` and is stale for the 63-target manifest), the three D2 companions (16 + 4 + 6 theorems), `CapFirstExit`
(152 declarations, kernel evidence via `cap-first-exit-lean.yml`, `alignment_review: NOT_CLAIMED`) and `W3Scratch`
(about 101 declarations, unregistered, built by no workflow, no kernel evidence). Their theorems are not Props stating
a claim of this register, and where their sources are not among the 23 pinned here no statement can be listed with
a verbatim anchor; the register's next step for each is to pin those sources and list the theorems as `proved`
statements in a later packet revision.

### 5.2 What stays at `none`, and why

Two different reasons put a row at `none`. The *mathematical* reason: the claim's principal display cannot be stated
without an object the sources leave informal, and the packet's rule is to make such objects fields of an interface
`structure`, never axioms; where even that would leave the claim itself unstated, no Prop was written. The
*inventory* reason: the bytes are not pinned in the packet (the 82 id-only PROOF_INDEX packets, the 12 `main`
experiment proofs, the 12 exact certificates, the 7 Lean packages, the graph's reading-rule and review nodes), or
there is no theorem to state (`hist.D0-RN-UNIF-historical-predicates`, whose carriers are absent by fingerprint;
`downstream-hard-gate`, an engineering control). Mapped onto the missing interfaces of
[docs/agents/LEAN_AUDIT_OWNER.md](agents/LEAN_AUDIT_OWNER.md) (paths below are relative to
`.lake/packages/mathlib/Mathlib/` in the pinned tree at `d13f23b7`, confirmed by `grep`):

| Interface | Claim families kept at `none`, or `specified` only over an interface | What the sources leave informal | Pinned Mathlib: has / lacks |
|---|---|---|---|
| **I1 source objects** | the four D1 reading-rule component nodes (`congruence-erratum`, `section9-replacement-v1_1`, `reconciliation-record`, `marked-cylinder-cap`: the last has pinned bytes but no Prop, and `CapFirstExit`'s 152 declarations are not registered); the D5 component rows, anchored to the reconciliation's transcription because `PUNCTURED_PIN_PROOF.md`, `COLLAR_PROOF.md` and `intermediate_window/PROOF.md` are not pinned; the eight D5 review-record nodes (reviews, not theorems) | identification of SP's frame, periodic lift, cap and frontier with the interface objects; the reading rule itself (E1, Section 9 v1.1, W1, `r < L/(4√2)`) is not transcribed into any hypothesis | has: `EuclideanSpace` (Analysis/InnerProductSpace/PiL2.lean:114), `ContDiff` (Analysis/Calculus/ContDiff/Defs.lean:1068), `Matrix.PosSemidef` (LinearAlgebra/Matrix/PosDef.lean:60); lacks: nothing structural, the gap is pinning and transcription |
| **I2 persistence pairing** | `SuperlevelBarInterface` (`Field`, `Lifetime`) takes the bar and the elder pairing as fields; `IsLifetimeDensityVersion` is characterised only by an integral identity; `RemainderBoundedR1`, `PlanarElderLowerE2`/`E4`/`E11` and the three parent Props are stated over it; the `main` experiment proofs (finite H0 lifetime, elder transfer, iid short-bar process, …) and the nine `periodic_h0` certificates are `none` | the H0 superlevel persistence module, born class, elder convention, tie rules; `elder_death_level_peak` in `CapFirstExit` is component-level only | lacks: persistent homology, barcodes, any elder rule (`grep -ri persisten\|barcode` finds only metaprogramming); has: `MeasureTheory.Measure.hausdorffMeasure` (MeasureTheory/Measure/Hausdorff.lean:535) |
| **I3 source data** | `PinnedLawInterface` (`RN`) imports the pin regime `0 < r < L`, `Z_r > 0`, the weight `W_r`, the contact kernel `Λ_j`; every D5/C6/RN row (`D5a`–`D5f`, dimension lift `P`/`C`/`I`/`G`, Palm `Q`/`Θ`, Fourier `F4`, remote collision `D`, annulus bridge, fixed annulus `A2`) is `specified` only over it; no inhabitant is constructed | the regression law `Q_r`, the typed weight, the normalizer floor `Z_r ≥ z_* r²` (5.5) — stated nowhere, the boundary of the "Probability formalizer" work offer | has: `MeasureTheory.Measure.withDensity`, `IsProbabilityMeasure`; lacks: nothing structural; an inhabitant (the Dirac-mass remark in the `PinnedLawInterface` docstring) is a proof-lane target |
| **I4 probability** | `GaussianFieldInterface` (`Field`) takes Gaussianity of finite linear combinations as a field (`ProbabilityTheory.gaussianReal`, Probability/Distributions/Gaussian/Real.lean:222, with `covQuadForm`); `rn-count-interface` stays `none` because (N6)–(N7), the Gaussian numerator interface, are not stated; `two-scale-s6-s21`, `axial-density`, `fixed-transverse-chart` are `none` (not pinned) | the field's construction; the Kac–Rice representations (parent Sections 9–10, contact kernel (13), the Stecconi weighted Kac–Rice input, the Palm route's marked Kac–Rice with Hölder insertion); `Q^W(G_r^c) ≤ C r³` | has: `gaussianReal`, `multivariateGaussian` (Gaussian/Multivariate.lean:166), `IsGaussian` (Gaussian/Basic.lean:45), `IsGaussianProcess` (Gaussian/IsGaussianProcess/Def.lean:30), `IsProjectiveMeasureFamily` / `IsProjectiveLimit` (MeasureTheory/Constructions/Projective.lean:45/115), `IsKolmogorovProcess` (Probability/Process/Kolmogorov.lean); lacks: the Kolmogorov extension theorem ("not in Mathlib yet", header of Probability/BrownianMotion/GaussianProjectiveFamily.lean), a continuous- or smooth-modification theorem (no grep match), Kac–Rice, area and coarea formulas (`grep -rn 'Kac\|coarea\|area formula'` finds only Kac–Moody algebras and Rice's theorem in computability), Palm measures, point processes; change of variables exists only for injective maps (`integral_image_eq_integral_abs_det_fderiv_smul`, MeasureTheory/Function/Jacobian.lean:1213, hypothesis `InjOn f s`) |
| **Analytic coefficient** | `side24-coefficient` is `specified` only over `PeriodicCoefficientInterface` (the cone moment `D_u` is a given field); the enclosure, the Section 5 Stirling / `Gamma(7/6)` remainder, the theta sums and `coefficient.py`'s interval arithmetic are not stated; the three SIDE24 follow-up packets (`CL-SIDE24-PFV`, `OA-SIDE24-REMAINDER`, `CL-SIDE24-PERIODIZATION-REMAINDER`) are `none` (not pinned); `cert:math-side24_v1-coefficient.py` is L2 evidence | the integration of the coefficient and a proof-producing enclosure | has: `Real.Gamma` (Analysis/SpecialFunctions/Gamma/Basic.lean:402), `Gamma_add_one` :410, `Gamma_one` :416, `Gamma_nat_eq_factorial` :422, `Gamma_pos_of_pos` :443, `Real.Gamma_one_half_eq` (Analysis/SpecialFunctions/Gaussian/GaussianIntegral.lean:307), `convexOn_log_Gamma` (Gamma/BohrMollerup.lean:111), `Gamma_strictMonoOn_Ici` :353, `Gamma_mul_Gamma_one_sub` (Gamma/Beta.lean:491); lacks: any numerical bound on `Gamma (7/6)`, interval arithmetic over ℝ (Tactic/NormNum has extensions for `Nat.log`, `Nat.sqrt`, `Real.sqrt`, none for `Real.exp`, `Real.log` or `Gamma`) |
| **Other source claims** | the 82 PROOF_INDEX packets listed by id only (statement *not extracted*; in the audit owner's words NOT ASSESSED, never inferred covered); `cumulative-transfer-correction` (pinned, no Prop); `math.rn-mesoscopic-reduction` and `math.rn-selector-region-crosswalk` (inventory nodes); `main:unmerged-codex-prs-330-350` (not on origin/main); the two P15 prize-reconnaissance packets (HOLD, listed by id only) | — | — |

Of the 15 graph nodes at `none`, four are D1 reading-rule components, eight are D5 review records (reviews, not
theorems), two have no Prop for a different reason each (`math.rn-mesoscopic-reduction`, whose source is not pinned,
and `math.rn-selector-region-crosswalk`, an inventory node) and one is `math.rn-count-interface` (its five abstract
Props exist, the Gaussian interface does not). Of the 89 PROOF_INDEX rows at `none`, 82 are id-only packets and
seven are inventoried with their statements but without a Prop for the headline: three whose bytes are not pinned
(`two-scale-s6-s21`, `axial-density`, `fixed-transverse-chart`) and four that are pinned (`OA-C6-FOURIER-20260929-v1`,
only (F4) stated and under the hypothesis `G_2`; `OA-C6-RARE-CLUSTER-20260929-v1`, only Theorem O stated;
`CL-D5-REMOTE-COLLISION-20260928-v1`, only Corollary D stated; `cumulative-transfer-correction`, no Prop).