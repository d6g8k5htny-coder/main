# Lean statement specifications (2026-10-10)

Scientific effect: **NONE**. Organizational-independence credit of this packet: 0 (Anthropic /
Claude, delegated AI work). Alignment status of every statement: `PENDING_INDEPENDENT_REVIEW`.
Formal progress of every declaration here: `specified`. The sibling proof packet
`specifications/lean_statement_proofs_20261011` holds theorems whose type is exactly one of these
Props (77 of the 141 Props, 78 theorems); the 64 Props without a theorem are classified in the Frontier section below; the register lists them under the claim rows as
statement entries of kind `theorem` with label `proved` (52 of them kernel-checked by the hosted run 38121713840: PR head 910809d6,
actual tested merge 5b198699… with base b2d70d77…, same tree as the head; the 8 of `P15Interpolation`, added after PR head 79731b96 (run 38124262242, actual tested merge 318ff194…), by run 38149888699:
PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96…, the head of PR #422, same tree as the head; the 13 of `Side24Analysis`, `Side24Interfaces` and `RNCluster`, added after PR head 12e072d6 (run 38150363063, actual tested merge 86a6e8c8…), by run 38180705952:
PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6…, the head of PR #423, same tree as the head; the 2 of `Side24Covariance` and `Side24ConeMoment`, added after PR head be014794 (run 38187538744, actual tested merge 79d3964c…), by run 38195810463:
PR head c6ce0256, actual tested merge 1320c5b2… with base be014794…, the head of PR #434, same tree as the head; the 3 of `Side24Images`, added after PR head c6ce0256 (run 38195810463, actual tested merge 1320c5b2…), by run 38277458954:
PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94…, Math- main at the merge of PR #440, the tested merge differing from the head only outside both packets),
which changes no
label of this packet and no claim-level label. No claim, premise, obligation or
disposition changes because this packet exists, builds, or is green in CI, or because a Prop of it is proved.

## What this packet is

A **statement-only** Lean 4 + Mathlib package. Each module declares, as `def … : Prop` and
interface `structure`s, the mathematical objects and results that the byte-pinned Layer 0 texts
under `sources/` describe. Two independent stages hold the modules to "statements only". The
source scan (`check.py`, no Lean) refuses `theorem`, `lemma`, `example`, `instance`, `axiom`,
`sorry` and the rest of its lexical list as whole tokens of comment-stripped code, refuses string
literals, guillemet names, `#` commands and attributes outright, requires every command keyword
at column 0 (so `open … in` and indented commands are refused), refuses `section` and `variable`
(no module needs them; a `variable (h : False)` would add a hypothesis the scan cannot see), requires an
explicit result type on every definition, refuses proof-shaped result types, `Sort` and `_root_`.
It is a fail-closed pre-Lean stage, not a Lean lexer. The authority is the environment audit of
`check.py --execute`: Lean's own `ConstantInfo` of every constant the five modules add refuses
any theorem, axiom, opaque constant, instance or definition whose type is a proposition beyond
what `structure` and `def` elaboration generate, refuses any transitive axiom outside `propext`,
`Classical.choice`, `Quot.sound`, and requires Lean's statement definitions (type syntactically
`∀ …, Prop`, read without reduction, so a `Set`-valued definition is auxiliary exactly as the
scan's `: Prop` rule has it), structures and auxiliary definitions to equal the register and the
scan exactly. A `def … : Prop` that
elaborates is a precise statement of what would have to be proved. It is not L5 of the #95
ladder; it is `specified`.

| file | role |
|---|---|
| `Specifications.lean` | root; imports exactly the five modules below, in this order |
| `Specifications/Field.lean` | shared objects of the periodized Gaussian lifetime track (namespace `UniversalLaw.Spec.Field`) |
| `Specifications/Lifetime.lean` | lifetime parent Theorems A/B/C, remainder, planar laws (`UniversalLaw.Spec.Lifetime`) |
| `Specifications/Side24.lean` | SIDE24 coefficient objects and enclosure statements (`UniversalLaw.Spec.Side24`) |
| `Specifications/RN.lean` | RN count interface, fixed-window and collision statements (`UniversalLaw.Spec.RN`) |
| `Specifications/P15.lean` | P15 realized covers, price boundary, price budget, full price (`UniversalLaw.Spec.P15`) |
| `REGISTER.json` | the statement register: claim rows, their Lean statements, formal-progress labels, what is not established, next step |
| `sources/` | byte-pinned Layer 0 copies (`sources/SOURCES.json`: repository, commit, path, blob, bytes, sha256) |
| `check.py`, `test_check.py`, `replay.sh` | the checker, its unit tests and negative controls, the local replay |
| `lakefile.toml`, `lean-toolchain`, `lake-manifest.json` | Lean `leanprover/lean4:v4.34.1`, Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`, lock identical to `formal/lake-manifest.json`; `autoImplicit = false` so an unbound name in a header is an error, never a silent implicit parameter |
| `.github/workflows/lean-specifications.yml` | CI: tests, source check, `lean-action` with the Mathlib cache, `check.py --execute`, artifact of `.lake/evidence/` |
| `../lean_statement_proofs_20261011/` | the sibling proof packet (path dependency on this packet, never edits it): theorems typed exactly as registered Props, its own gate `check.py` and workflow `.github/workflows/lean-specification-proofs.yml`; its theorems appear in `REGISTER.json` as `proved` statement entries |

## Declarations

150 registered declarations (141 `def … : Prop`, 9 interface `structure`), every one with formal progress `specified` in this package; the 123 auxiliary definitions and abbreviations of the modules are inventory only and are not register statements. Names are given without the module namespace `UniversalLaw.Spec.<Module>.`; the source column is the pinned source named by the docstring's `Source:` line. Anchors and "Not established" cells are copied verbatim from the Lean docstrings (`Anchor:` and `Does not claim:` lines) through REGISTER.json. The last column names the theorems of the sibling proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly the declaration (namespace `UniversalLaw.Proofs.<Family>.`): 77 of the 141 Props have one (78 theorems; `TriangleNeedsThree` is proved twice), each listed in the register under the claim rows as a statement entry of kind `theorem` with label `proved` (the label of a committed proof package; `kernel-checked` appears only in a run receipt: the hosted `lean-specification-proofs.yml` run 38121713840 receipted the 52 theorems of PR head 910809d6 and run 38116277597 the first 38; a pull_request run checks out the synthetic merge of the PR head into the base, so each receipt records as checked_commit that merge and not the head: run 38121713840 PR head 910809d6, actual tested merge 5b198699308e0585904c21d12bf540cf2475d2f0 with base b2d70d77876a10c365f5db5d3b6cafcf180af4e2; run 38116277597 PR head 00d2ac6e, actual tested merge c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c with the same base; run 38122386459 corroborates on PR head d9b8dd33, actual tested merge 866136cca66b46e6dbd3186b47c9340f7e93b4f6; each merge has the same tree as its head; the 8 theorems of `P15Interpolation`, added after PR head 79731b96 (run 38124262242, actual tested merge 318ff194eecc42c39a6b27c1d7bf03f30d6a34d9 with base b2d70d77876a10c365f5db5d3b6cafcf180af4e2, receipt theorem_count 52), are kernel-checked by the hosted run 38149888699 on PR head 0dcb0016 (actual tested merge 4363b6928c62da242952631c6b0357d096c3b252 with base 79731b961fac555f33be3cd90e379b246a5a80e3, the head of PR #422; receipt theorem_count 60, leanchecker PASS) and were first receipted by run 38149524759 on PR head 88c21058 (actual tested merge 3306233bbdeadb15d9b4b07b94cc52a80ed8caeb with the same base); same trees as the heads; the 13 theorems of `Side24Analysis`, `Side24Interfaces` and `RNCluster`, added after PR head 12e072d6 (run 38150363063, actual tested merge 86a6e8c876f7904cadda2d006a56f8792dc5c1af with base 79731b961fac555f33be3cd90e379b246a5a80e3; receipt theorem_count 60, corroborating the 60 theorems above on that head; artifact lean-specification-proofs-38150363063-1, sha256 bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690), are kernel-checked by the hosted run 38180705952 on PR head 48bfcbae (actual tested merge 9833d6d9f9facc335e1a56e243c3144adefe6225 with base 12e072d614dbe6e91b66d22a2ab55cf87860d693, the head of PR #423; receipt theorem_count 73, leanchecker PASS; artifact lean-specification-proofs-38180705952-1, sha256 87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd) and were first receipted by that run, which receipts all 73 theorems of that head; same tree as the head; the 2 theorems of `Side24Covariance` and `Side24ConeMoment`, added after PR head be014794 (run 38187538744, actual tested merge 79d3964cf70e2eadad5c9629eef8f4b47f218dd0 with base 12e072d614dbe6e91b66d22a2ab55cf87860d693; receipt theorem_count 73, corroborating the 73 theorems above on that head; artifact lean-specification-proofs-38187538744-1, sha256 60c9be0b49d45931d6c82f04d73fb7e6b6e97d539e7c10a376c3e99e302f2617), are kernel-checked by the hosted run 38195810463 on PR head c6ce0256 (actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169, the head of PR #434; receipt theorem_count 75, leanchecker PASS; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff) and were first receipted by that run, which receipts all 75 theorems of that head; same tree as the head; the 3 theorems of `Side24Images`, added after PR head c6ce0256 (run 38195810463, actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169; receipt theorem_count 75, naming the 75 theorems above on that head and not these; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff), are kernel-checked by the hosted run 38277458954 on PR head 9e305ea5 (actual tested merge 1376175ce5536a6ac74cb03706d18e65fdb695fa with base a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc, Math- main at the merge of PR #440; receipt theorem_count 78, leanchecker PASS; artifact lean-specification-proofs-38277458954-1, sha256 1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79) and were first receipted by that run, which receipts all 78 theorems of that head; the tested merge differs from the head only outside both packets (main's changes from PRs #428 and #440: 13 files under `companions/torus_filtration_20261011/` and `.github/workflows/torus-filtration.yml`; this packet and the proof packet are byte-identical between head and merge)). That label belongs to the theorem, not to the declaration here, which stays `specified`, and not to any claim row; a kernel-checked theorem is evidence about its Prop only, and alignment of every statement is `PENDING_INDEPENDENT_REVIEW`.

### `Specifications/Field.lean` (12 registered declarations)

| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; 52 kernel-checked by hosted run 38121713840: PR head 910809d6, actual tested merge 5b198699… with base b2d70d77…; the 8 of P15Interpolation by run 38149888699: PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96… (the head of PR #422); same trees as the heads; the 13 of Side24Analysis, Side24Interfaces and RNCluster by run 38180705952: PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6… (the head of PR #423), the merge having the same tree as PR head 48bfcbae; receipt theorem_count 73, artifact sha256 87309efa…; the 2 of Side24Covariance and Side24ConeMoment by run 38195810463: PR head c6ce0256, actual tested merge 1320c5b2… with base be014794… (the head of PR #434), the merge having the same tree as PR head c6ce0256; receipt theorem_count 75, artifact sha256 d8abd42b…; the 3 of Side24Images by run 38277458954: PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94… (Math- main at the merge of PR #440), the tested merge differing from PR head 9e305ea5 only outside both packets; receipt theorem_count 78, artifact sha256 1f32d60e…) |
|---|---|---|---|---|
| `IsLPeriodic` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsSmooth` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `GaussianFieldInterface` | structure | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsCriticalPoint` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `HessianHasInertia` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsNondegMax` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsIndexSaddle` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsCandidatePair` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsMorseDistinct` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `SuperlevelBarInterface` | structure | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsFiniteBar` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsLifetimeDensityVersion` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |

### `Specifications/Lifetime.lean` (24 registered declarations)

| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; 52 kernel-checked by hosted run 38121713840: PR head 910809d6, actual tested merge 5b198699… with base b2d70d77…; the 8 of P15Interpolation by run 38149888699: PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96… (the head of PR #422); same trees as the heads; the 13 of Side24Analysis, Side24Interfaces and RNCluster by run 38180705952: PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6… (the head of PR #423), the merge having the same tree as PR head 48bfcbae; receipt theorem_count 73, artifact sha256 87309efa…; the 2 of Side24Covariance and Side24ConeMoment by run 38195810463: PR head c6ce0256, actual tested merge 1320c5b2… with base be014794… (the head of PR #434), the merge having the same tree as PR head c6ce0256; receipt theorem_count 75, artifact sha256 d8abd42b…; the 3 of Side24Images by run 38277458954: PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94… (Math- main at the merge of PR #440), the tested merge differing from PR head 9e305ea5 only outside both packets; receipt theorem_count 78, artifact sha256 1f32d60e…) |
|---|---|---|---|---|
| `IsLPeriodic` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsSmooth` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `GaussianFieldInterface` | structure | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsCriticalPoint` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `HessianHasInertia` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsNondegMax` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsIndexSaddle` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsCandidatePair` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsMorseDistinct` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `SuperlevelBarInterface` | structure | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsFiniteBar` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime` | — |
| `IsLifetimeDensityVersion` | def_prop | `lifetime-parent` | `lifetime-remainder`, `math.uniform-matrix-cap-lifetime`, `D1-THM-B` | — |
| `IsWindowPair` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-B` | — |
| `IsWindowElderPair` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-B` | — |
| `IsUnrestrictedPair` | def_prop | `lifetime-parent` | `lifetime-remainder`, `math.uniform-matrix-cap-lifetime`, `D1-THM-C` | — |
| `PairLawInterface` | structure | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-A` | — |
| `ParentUniformSelectionA` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-A` | — |
| `ParentCompactWindowDensityB` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-B` | — |
| `IsUnrestrictedLeadingCoefficient` | def_prop | `lifetime-parent` | `lifetime-remainder`, `math.uniform-matrix-cap-lifetime`, `D1-THM-C` | — |
| `ParentUnrestrictedLeadingDensityC` | def_prop | `lifetime-parent` | `math.uniform-matrix-cap-lifetime`, `D1-THM-C` | — |
| `RemainderBoundedR1` | def_prop | `lifetime-remainder` | `lifetime-remainder`, `math.lifetime-remainder` | — |
| `PlanarElderLowerE2` | def_prop | `elder-lower-density-gap` | `OA-ELDER-LOWER-DENSITY-GAP-20260928-v1` | — |
| `PlanarTwoSidedRateE4` | def_prop | `elder-lower-density-gap` | `OA-ELDER-LOWER-DENSITY-GAP-20260928-v1` | — |
| `PlanarDensityGapE11` | def_prop | `elder-lower-density-gap` | `OA-ELDER-LOWER-DENSITY-GAP-20260928-v1` | — |

### `Specifications/Side24.lean` (28 registered declarations)

| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; 52 kernel-checked by hosted run 38121713840: PR head 910809d6, actual tested merge 5b198699… with base b2d70d77…; the 8 of P15Interpolation by run 38149888699: PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96… (the head of PR #422); same trees as the heads; the 13 of Side24Analysis, Side24Interfaces and RNCluster by run 38180705952: PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6… (the head of PR #423), the merge having the same tree as PR head 48bfcbae; receipt theorem_count 73, artifact sha256 87309efa…; the 2 of Side24Covariance and Side24ConeMoment by run 38195810463: PR head c6ce0256, actual tested merge 1320c5b2… with base be014794… (the head of PR #434), the merge having the same tree as PR head c6ce0256; receipt theorem_count 75, artifact sha256 d8abd42b…; the 3 of Side24Images by run 38277458954: PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94… (Math- main at the merge of PR #440), the tested merge differing from PR head 9e305ea5 only outside both packets; receipt theorem_count 78, artifact sha256 1f32d60e…) |
|---|---|---|---|---|
| `ThetaSummable` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.thetaSummable` |
| `IsUnitTuple` | def_prop | `side24-proof` | `side24-coefficient` | — |
| `DerivativePairingBound` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.derivativePairingBound` |
| `DerivativeBoundAtLeastOne` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.derivativeBoundAtLeastOne` |
| `DerivativeBoundAtZero` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.derivativeBoundAtZero` |
| `LatticeImageSumBound` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.latticeImageSumBound` |
| `SuccessiveTermRatio` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Elementary.successiveTermRatio` |
| `ExponentialBounds` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Elementary.exponentialBounds` |
| `ImageBound` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Images.imageBound` |
| `CovarianceOrdering` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Images.covarianceOrdering` |
| `EntrywiseCovarianceBound` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Images.entrywiseCovarianceBound` |
| `ReferenceCovarianceLowerBound` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Covariance.referenceCovarianceLowerBound` |
| `ReferenceOddAndPinBlocks` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.referenceOddAndPinBlocks` |
| `DensityComparison` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.densityComparison` |
| `IntegrandRatioBounds` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.integrandRatioBounds` |
| `ReferencePinDensityProduct` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Analysis.referencePinDensityProduct` |
| `ReferenceCoefficientIdentity` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Elementary.referenceCoefficientIdentity` |
| `ConeMomentInterface` | structure | `side24-proof` | `side24-coefficient` | — |
| `NegDefCone` | def_prop | `side24-proof` | `side24-coefficient` | — |
| `ConeMomentD1Identity` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Interfaces.coneMomentD1Identity` |
| `ConeMomentD2Identity` | def_prop | `side24-proof` | `side24-coefficient` | `Side24ConeMoment.coneMomentD2Identity` |
| `ConeMomentD2Algebra` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Elementary.coneMomentD2Algebra` |
| `TraceMoments` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Interfaces.traceMoments` |
| `RayleighSquareLaw` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Interfaces.rayleighSquareLaw` |
| `ElementaryConeIntegral` | def_prop | `side24-proof` | `side24-coefficient` | `Side24Elementary.elementaryConeIntegral` |
| `PeriodicCoefficientInterface` | structure | `side24-proof` | `side24-coefficient`, `math.side24-coefficient` | — |
| `CoefficientEnclosure` | def_prop | `side24-proof` | `side24-coefficient`, `math.side24-coefficient` | — |
| `ReferenceComparison` | def_prop | `side24-proof` | `side24-coefficient`, `math.side24-coefficient` | — |

### `Specifications/RN.lean` (28 registered declarations)

| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; 52 kernel-checked by hosted run 38121713840: PR head 910809d6, actual tested merge 5b198699… with base b2d70d77…; the 8 of P15Interpolation by run 38149888699: PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96… (the head of PR #422); same trees as the heads; the 13 of Side24Analysis, Side24Interfaces and RNCluster by run 38180705952: PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6… (the head of PR #423), the merge having the same tree as PR head 48bfcbae; receipt theorem_count 73, artifact sha256 87309efa…; the 2 of Side24Covariance and Side24ConeMoment by run 38195810463: PR head c6ce0256, actual tested merge 1320c5b2… with base be014794… (the head of PR #434), the merge having the same tree as PR head c6ce0256; receipt theorem_count 75, artifact sha256 d8abd42b…; the 3 of Side24Images by run 38277458954: PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94… (Math- main at the merge of PR #440), the tested merge differing from PR head 9e305ea5 only outside both packets; receipt theorem_count 78, artifact sha256 1f32d60e…) |
|---|---|---|---|---|
| `IsCriticalPoint` | def_prop | `lifetime-parent` | `rn-fixed-remote-window` | — |
| `HessianHasInertia` | def_prop | `lifetime-parent` | `rn-fixed-remote-window` | — |
| `HolderCountBoundN1` | def_prop | `rn-count-interface` | `rn-count-interface`, `math.rn-count-interface` | `RNProbability.holderCountBoundN1` |
| `HolderConsequenceN2` | def_prop | `rn-count-interface` | `rn-count-interface`, `math.rn-count-interface` | `RNProbability.holderConsequenceN2` |
| `ConditionalMeanIdentityN3` | def_prop | `rn-count-interface` | `rn-count-interface`, `math.rn-count-interface` | `RNProbability.conditionalMeanIdentityN3` |
| `UniformCounterexampleN4` | def_prop | `rn-count-interface` | `rn-count-interface`, `math.rn-count-interface` | `RNProbability.uniformCounterexampleN4` |
| `ExponentialTailLayerCakeN5` | def_prop | `rn-count-interface` | `rn-count-interface`, `math.rn-count-interface` | `RNProbability.exponentialTailLayerCakeN5` |
| `PinnedLawInterface` | structure | `rn-fixed-remote-window` | `rn-fixed-remote-window`, `math.rn-fixed-remote-window` | — |
| `FixedRemoteMeanMeasureA2` | def_prop | `rn-fixed-remote-window` | `rn-fixed-remote-window`, `math.rn-fixed-remote-window`, `math.rn-region.fixed-remote`, `math.d5-component.remote-window-proof` | — |
| `FixedRemoteCountBoundA3` | def_prop | `rn-fixed-remote-window` | `rn-fixed-remote-window`, `math.rn-fixed-remote-window`, `math.rn-region.fixed-remote`, `math.d5-component.remote-window-proof` | — |
| `FixedAnnulusWindowCandidateA2` | def_prop | `rn-fixed-annulus-window` | `rn-fixed-annulus-window`, `math.rn-fixed-annulus-window`, `math.rn-region.fixed-annulus-window` | — |
| `AnnulusBridgeAllHeight2` | def_prop | `rn-annulus-bridge` | `math.rn-region.mesoscopic-scaled-annulus`, `math.d5-component.annulus-proof` | — |
| `D5aPuncturedPinDisks` | def_prop | `d5-reconciliation` | `math.rn-region.pin-collision`, `math.d5-pin-neighborhood-first-moment`, `math.d5-component.punctured-pin-proof` | — |
| `D5bMicrodisk` | def_prop | `d5-reconciliation` | `math.rn-region.pin-collision`, `math.d5-pin-neighborhood-first-moment`, `math.d5-component.punctured-pin-proof` | — |
| `D5cCollarC1` | def_prop | `d5-reconciliation` | `math.d5-pin-neighborhood-first-moment`, `math.d5-component.collar-proof` | — |
| `D5dSummedPinNeighbourhoodC2` | def_prop | `d5-reconciliation` | `math.rn-region.mesoscopic-scaled-annulus`, `math.rn-region.pin-collision`, `math.d5-pin-neighborhood-first-moment` | — |
| `D5eIntermediateShellsI3I4` | def_prop | `d5-reconciliation` | `math.rn-region.intermediate-r-to-rho`, `math.d5-pin-neighborhood-first-moment`, `math.d5-component.intermediate-window-proof` | — |
| `D5fGlobalWindowFirstMomentI5` | def_prop | `d5-reconciliation` | `math.d5-pin-neighborhood-first-moment`, `math.d5-component.intermediate-window-proof` | — |
| `DimensionLiftPuncturedPinBallP` | def_prop | `d5-dimension-lift` | `CL-D5-DIMENSION-LIFT-20260929-v1` | — |
| `DimensionLiftCompactCollarC` | def_prop | `d5-dimension-lift` | `CL-D5-DIMENSION-LIFT-20260929-v1` | — |
| `DimensionLiftIntermediateShellsI` | def_prop | `d5-dimension-lift` | `CL-D5-DIMENSION-LIFT-20260929-v1` | — |
| `DimensionLiftGlobalWindowG` | def_prop | `d5-dimension-lift` | `CL-D5-DIMENSION-LIFT-20260929-v1` | — |
| `FourierCutoffPlanarSecondFactorialF4` | def_prop | `c6-fourier-cutoff` | `OA-C6-FOURIER-20260929-v1` | — |
| `PalmRouteFactorialMomentsQ` | def_prop | `c6-palm-route` | `CL-C6-PALM-20260929-v1` | — |
| `PalmRouteSecondFactorialTheta` | def_prop | `c6-palm-route` | `CL-C6-PALM-20260929-v1` | — |
| `RareClusterPoissonObstructionO` | def_prop | `c6-rare-cluster` | `OA-C6-RARE-CLUSTER-20260929-v1` | `RNCluster.rareClusterPoissonObstructionO` |
| `RemoteCollisionSecondFactorialD` | def_prop | `remote-collision` | `math.rn-region.witness-collision`, `CL-D5-REMOTE-COLLISION-20260928-v1` | — |
| `WitnessCollisionNearPinPairsTarget` | def_prop | `d5-reconciliation` | `math.rn-region.witness-collision` | — |

### `Specifications/P15.lean` (58 registered declarations)

| declaration | kind | source | registered under claim rows | proved by (proof packet, `proved`; 52 kernel-checked by hosted run 38121713840: PR head 910809d6, actual tested merge 5b198699… with base b2d70d77…; the 8 of P15Interpolation by run 38149888699: PR head 0dcb0016, actual tested merge 4363b692… with base 79731b96… (the head of PR #422); same trees as the heads; the 13 of Side24Analysis, Side24Interfaces and RNCluster by run 38180705952: PR head 48bfcbae, actual tested merge 9833d6d9… with base 12e072d6… (the head of PR #423), the merge having the same tree as PR head 48bfcbae; receipt theorem_count 73, artifact sha256 87309efa…; the 2 of Side24Covariance and Side24ConeMoment by run 38195810463: PR head c6ce0256, actual tested merge 1320c5b2… with base be014794… (the head of PR #434), the merge having the same tree as PR head c6ce0256; receipt theorem_count 75, artifact sha256 d8abd42b…; the 3 of Side24Images by run 38277458954: PR head 9e305ea5, actual tested merge 1376175c… with base a0323b94… (Math- main at the merge of PR #440), the tested merge differing from PR head 9e305ea5 only outside both packets; receipt theorem_count 78, artifact sha256 1f32d60e…) |
|---|---|---|---|---|
| `RealizedData` | structure | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsRealizedFamily` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsGood` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsKDecomposable` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsGeneratorCover` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsPaletteFeasible` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `PaletteNumberAttained` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Covers.paletteNumberAttained` |
| `IsMinimalForbidden` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `IsTransversal` | def_prop | `p15-realized-covers` | `p15-realized-covers` | — |
| `MinimalForbiddenCharacterisation` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.minimalForbiddenCharacterisation` |
| `LocalRestriction` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Covers.localRestriction` |
| `LocalDecomposability` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.localDecomposability` |
| `LocalObstructionIsFullBlock` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.localObstructionIsFullBlock` |
| `LocalPriceChain` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Hazard.localPriceChain` |
| `IndependentBlockHazard` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Hazard.independentBlockHazard` |
| `FullBlockCoverIff` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.fullBlockCoverIff` |
| `CoverCostHazardBound` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Budget.coverCostHazardBound` |
| `WholeGroundChromaticNumber` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.wholeGroundChromaticNumber` |
| `ElementaryPaletteFormula` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.elementaryPaletteFormula` |
| `TriangleNeedsThree` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Elementary.triangleNeedsThree`, `P15Structure.triangleNeedsThree` |
| `Benchmark816IsRealized` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.benchmark816IsRealized` |
| `Benchmark816Palette` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.benchmark816Palette` |
| `Benchmark816Cover` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Covers.benchmark816Cover` |
| `Benchmark816WholeGround` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.benchmark816WholeGround` |
| `Benchmark816MinimalForbiddenCounts` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.benchmark816MinimalForbiddenCounts` |
| `Benchmark816Price` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Structure.benchmark816Price` |
| `Benchmark816UnionBound` | def_prop | `p15-realized-covers` | `p15-realized-covers` | `P15Covers.benchmark816UnionBound` |
| `TwoCoordGeneratorPrices` | def_prop | `p15-price-boundary` | `p15-price-boundary`, `math.p15-price-boundary` | `P15Elementary.twoCoordGeneratorPrices` |
| `DemandOneRefutation` | def_prop | `p15-price-boundary` | `p15-price-boundary`, `math.p15-price-boundary` | `P15Elementary.demandOneRefutation` |
| `LogOnePlusBelowIdentity` | def_prop | `p15-price-boundary` | `p15-price-boundary` | `P15Elementary.logOnePlusBelowIdentity` |
| `LogRatioBound` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Elementary.logRatioBound` |
| `RatioExponentBound` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Elementary.ratioExponentBound` |
| `RestrictedLocalBudget` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Budget.restrictedLocalBudget` |
| `RestrictedTransformedPriceBudget` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Budget.restrictedTransformedPriceBudget` |
| `RhoTestSufficient` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Budget.rhoTestSufficient` |
| `CapTestAtTwoThirds` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Elementary.capTestAtTwoThirds` |
| `Benchmark816TransformedRange` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Budget.benchmark816TransformedRange` |
| `Benchmark816RationalPrice` | def_prop | `p15-price-budget` | `p15-price-budget-restricted` | `P15Structure.benchmark816RationalPrice` |
| `FullTransformedPriceBudget` | def_prop | `p15-full-price` | `p15-full-price`, `math.p15-full-price` | `P15Interpolation.fullTransformedPriceBudget` |
| `RhoStarEnclosure` | def_prop | `p15-full-price` | `p15-full-price`, `math.p15-full-price` | `P15Interpolation.rhoStarEnclosure` |
| `RhoStarBelowSixSevenths` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.rhoStarBelowSixSevenths` |
| `IsProperDownset` | def_prop | `p15-full-price` | `p15-full-price` | — |
| `HazardStarPositive` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.hazardStarPositive` |
| `CoordinatewiseConcavity` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.coordinatewiseConcavity` |
| `HazardProductBound` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.hazardProductBound` |
| `HazardPhiBound` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.hazardPhiBound` |
| `FullGroundPriceBound` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.fullGroundPriceBound` |
| `PhiAtPStar` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.phiAtPStar` |
| `FullGroundBudgetIff` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.fullGroundBudgetIff` |
| `CapacityHazardIsBinomial` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.capacityHazardIsBinomial` |
| `OddMajorityIdentity` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.oddMajorityIdentity` |
| `CapacityWorstCase` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.capacityWorstCase` |
| `HStarExceedsOne` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.hStarExceedsOne` |
| `LocalPriceBoundFull` | def_prop | `p15-full-price` | `p15-full-price` | `P15Interpolation.localPriceBoundFull` |
| `GlobalAssemblyHazard` | def_prop | `p15-full-price` | `p15-full-price` | `P15Hazard.globalAssemblyHazard` |
| `SharpnessOfRhoStar` | def_prop | `p15-full-price` | `p15-full-price`, `math.p15-full-price` | `P15Elementary.sharpnessOfRhoStar` |
| `DemandOneBoundary` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.demandOneBoundary` |
| `SixSeventhsCertificate` | def_prop | `p15-full-price` | `p15-full-price` | `P15Elementary.sixSeventhsCertificate` |

## Claims

171 claim rows: 10 `landing_claim`, 33 `graph_node`, 2 `open_obligation`, 95 `proof_index_result`, 7 `existing_lean_package`, 12 `main_experiment_proof`, 12 `exact_certificate`. Claim-level formal progress: 33 `specified` (a `def … : Prop` stating the claim itself is registered), 138 `none`; no research claim is `proved` or `kernel-checked`. 197 statement entries point at the 150 declarations of this package (all `specified`). 160 statement entries carry the label `proved` and are theorems of other packages, listed as the arithmetic skeleton or the Prop-level proof of the claim they sit under and never as the claim: 31 are the 31 Side24 arithmetic targets of main's `formal/` at origin/main `f6deeba7`; 41 are the 28 P15 arithmetic targets of main's `formal/` at `91cffbb5` (main PR #358, not yet on main; kernel evidence the hosted workspace-landing run 38096827443), each listed on every P15 row whose layer 0 source it anchors to; 88 are theorems of the proof packet `specifications/lean_statement_proofs_20261011` whose type is exactly a registered Prop of this package (78 theorems, 77 distinct Props, each listed on every row that registers its Prop; the hosted `lean-specification-proofs.yml` run 38121713840 receipted the 52 theorems of PR head 910809d6 as kernel-checked and run 38116277597 the first 38; a pull_request run checks out the synthetic merge of the PR head into the base, which the receipt records as checked_commit: run 38121713840 PR head 910809d6, actual tested merge 5b198699308e0585904c21d12bf540cf2475d2f0 with base b2d70d77876a10c365f5db5d3b6cafcf180af4e2; run 38116277597 PR head 00d2ac6e, actual tested merge c4c66bb8cb26f0cd9bc03d52bb25f1b3925b272c with the same base; run 38122386459 corroborates on PR head d9b8dd33, actual tested merge 866136cca66b46e6dbd3186b47c9340f7e93b4f6; same trees as the heads; the 8 theorems of `P15Interpolation`, added after PR head 79731b96 (run 38124262242, actual tested merge 318ff194eecc42c39a6b27c1d7bf03f30d6a34d9 with base b2d70d77876a10c365f5db5d3b6cafcf180af4e2, receipt theorem_count 52), are kernel-checked by the hosted run 38149888699 on PR head 0dcb0016 (actual tested merge 4363b6928c62da242952631c6b0357d096c3b252 with base 79731b961fac555f33be3cd90e379b246a5a80e3, the head of PR #422; receipt theorem_count 60, leanchecker PASS) and were first receipted by run 38149524759 on PR head 88c21058 (actual tested merge 3306233bbdeadb15d9b4b07b94cc52a80ed8caeb with the same base); same trees as the heads; the 13 theorems of `Side24Analysis`, `Side24Interfaces` and `RNCluster`, added after PR head 12e072d6 (run 38150363063, actual tested merge 86a6e8c876f7904cadda2d006a56f8792dc5c1af with base 79731b961fac555f33be3cd90e379b246a5a80e3; receipt theorem_count 60, corroborating the 60 theorems above on that head; artifact lean-specification-proofs-38150363063-1, sha256 bf0e0515e2035103b07d894e2a73eeb5b542885f6da16b98c228877a5e0c1690), are kernel-checked by the hosted run 38180705952 on PR head 48bfcbae (actual tested merge 9833d6d9f9facc335e1a56e243c3144adefe6225 with base 12e072d614dbe6e91b66d22a2ab55cf87860d693, the head of PR #423; receipt theorem_count 73, leanchecker PASS; artifact lean-specification-proofs-38180705952-1, sha256 87309efad9b804c53c543b60fdb466de1f797971417363cc2e8fa1e10eac26bd) and were first receipted by that run, which receipts all 73 theorems of that head; same tree as the head; the 2 theorems of `Side24Covariance` and `Side24ConeMoment`, added after PR head be014794 (run 38187538744, actual tested merge 79d3964cf70e2eadad5c9629eef8f4b47f218dd0 with base 12e072d614dbe6e91b66d22a2ab55cf87860d693; receipt theorem_count 73, corroborating the 73 theorems above on that head; artifact lean-specification-proofs-38187538744-1, sha256 60c9be0b49d45931d6c82f04d73fb7e6b6e97d539e7c10a376c3e99e302f2617), are kernel-checked by the hosted run 38195810463 on PR head c6ce0256 (actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169, the head of PR #434; receipt theorem_count 75, leanchecker PASS; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff) and were first receipted by that run, which receipts all 75 theorems of that head; same tree as the head; the 3 theorems of `Side24Images`, added after PR head c6ce0256 (run 38195810463, actual tested merge 1320c5b217e491d301ea140e5cc25b07ff5a21b7 with base be01479435f98f34b802258ee9ec3c3578f37169; receipt theorem_count 75, naming the 75 theorems above on that head and not these; artifact lean-specification-proofs-38195810463-1, sha256 d8abd42be0facd4248647c3c7eb67a264bd19b9b47c6128733b5b5fc93b0a6ff), are kernel-checked by the hosted run 38277458954 on PR head 9e305ea5 (actual tested merge 1376175ce5536a6ac74cb03706d18e65fdb695fa with base a0323b94ae54a78f3d2878dc9c358dc40cdfe1dc, Math- main at the merge of PR #440; receipt theorem_count 78, leanchecker PASS; artifact lean-specification-proofs-38277458954-1, sha256 1f32d60e08b0ff11b634c0413291686982f0b02e6809b9e96af2848350cbbf79) and were first receipted by that run, which receipts all 78 theorems of that head; the tested merge differs from the head only outside both packets (main's changes from PRs #428 and #440: 13 files under `companions/torus_filtration_20261011/` and `.github/workflows/torus-filtration.yml`; this packet and the proof packet are byte-identical between head and merge)). Exactly one Prop registered as stating a claim itself has such a theorem: `DemandOneRefutation` on rows `p15-price-boundary` and `math.p15-price-boundary`; their claim-level label stays `specified` (the register admits only none/specified at claim level, alignment of the Prop with the source paragraph is PENDING_INDEPENDENT_REVIEW, and a receipt is about the Prop only), as the rows' `not established` cells record. `disposition` is the verbatim LANDING_CLAIMS value or `—` (null). Nothing in these tables is a scientific status, a review verdict or an acceptance; alignment of every statement is `PENDING_INDEPENDENT_REVIEW`.

| claim_id | kind | disposition (LANDING_CLAIMS, verbatim) | formal_progress | layer 0 source | statements |
|---|---|---|---|---|---|
| `side24-coefficient` | landing_claim | REVIEWED_SCOPED | `specified` | `side24-proof` | 28 specified: `ThetaSummable`, `IsUnitTuple`, `DerivativePairingBound`, `DerivativeBoundAtLeastOne`, `DerivativeBoundAtZero`, `LatticeImageSumBound`, `SuccessiveTermRatio`, `ExponentialBounds`, `ImageBound`, `CovarianceOrdering`, `EntrywiseCovarianceBound`, `ReferenceCovarianceLowerBound`, `ReferenceOddAndPinBlocks`, `DensityComparison`, `IntegrandRatioBounds`, `ReferencePinDensityProduct`, `ReferenceCoefficientIdentity`, `ConeMomentInterface`, `NegDefCone`, `ConeMomentD1Identity`, `ConeMomentD2Identity`, `ConeMomentD2Algebra`, `TraceMoments`, `RayleighSquareLaw`, `ElementaryConeIntegral`, `PeriodicCoefficientInterface`, `CoefficientEnclosure`, `ReferenceComparison`; 27 proved (main, origin/main f6deeba7): `pairing_sum_six`, `pairing_sum_le_six`, `pairing_terms_six`, `sixth_moment_double_factorial`, `lattice_shell_constant`, `geometric_ratio_constants`, `period_exponent`, `image_constant_value`, `exp_taylor_partial_sum_gt_ten`, `covariance_relative_bound`, `odd_block_eigenvalues`, `odd_block_shifted_minors`, `smaller_eigenvalue_exceeds_third`, `exponent_bounds`, `exponent_values`, `ratio_constant_ordering`, `density_comparison_below_reported`, `conditional_third_derivative_variance`, `transverse_covariance_entries_thirds`, `cone_moment_m1_thirds`, `trace_and_traceless_variances`, `fourth_moment_of_trace`, `exponential_moment_of_trace`, `cone_moment_m2_algebra`, `cube_root_simplification`, `pin_determinant_and_joint_dimension`, `hessian_block_eigenvalues`; 22 proved (proof packet): `Side24Elementary.exponentialBounds`, `Side24Elementary.successiveTermRatio`, `Side24Elementary.coneMomentD2Algebra`, `Side24Elementary.referenceCoefficientIdentity`, `Side24Elementary.elementaryConeIntegral`, `Side24Analysis.thetaSummable`, `Side24Analysis.derivativePairingBound`, `Side24Analysis.derivativeBoundAtLeastOne`, `Side24Analysis.derivativeBoundAtZero`, `Side24Analysis.referenceOddAndPinBlocks`, `Side24Analysis.referencePinDensityProduct`, `Side24Analysis.integrandRatioBounds`, `Side24Analysis.latticeImageSumBound`, `Side24Analysis.densityComparison`, `Side24Interfaces.coneMomentD1Identity`, `Side24Interfaces.traceMoments`, `Side24Interfaces.rayleighSquareLaw`, `Side24Covariance.referenceCovarianceLowerBound`, `Side24ConeMoment.coneMomentD2Identity`, `Side24Images.imageBound`, `Side24Images.entrywiseCovarianceBound`, `Side24Images.covarianceOrdering` |
| `lifetime-remainder` | landing_claim | REVIEWED_SCOPED | `specified` | `lifetime-remainder` | 4 specified: `RemainderBoundedR1`, `IsUnrestrictedPair`, `IsUnrestrictedLeadingCoefficient`, `IsLifetimeDensityVersion` |
| `rn-count-interface` | landing_claim | HOLD_WITH_DOMAIN | `none` | `rn-count-interface` | 5 specified: `HolderCountBoundN1`, `HolderConsequenceN2`, `ConditionalMeanIdentityN3`, `UniformCounterexampleN4`, `ExponentialTailLayerCakeN5`; 5 proved (proof packet): `RNProbability.conditionalMeanIdentityN3`, `RNProbability.holderCountBoundN1`, `RNProbability.holderConsequenceN2`, `RNProbability.uniformCounterexampleN4`, `RNProbability.exponentialTailLayerCakeN5` |
| `rn-fixed-remote-window` | landing_claim | HOLD_WITH_DOMAIN | `specified` | `rn-fixed-remote-window` | 5 specified: `PinnedLawInterface`, `IsCriticalPoint`, `HessianHasInertia`, `FixedRemoteMeanMeasureA2`, `FixedRemoteCountBoundA3` |
| `rn-fixed-annulus-window` | landing_claim | REVIEWED_SCOPED | `specified` | `rn-fixed-annulus-window` | 1 specified: `FixedAnnulusWindowCandidateA2` |
| `p15-realized-covers` | landing_claim | HOLD_WITH_DOMAIN | `specified` | `p15-realized-covers` | 27 specified: `RealizedData`, `IsRealizedFamily`, `IsGood`, `IsKDecomposable`, `IsGeneratorCover`, `IsPaletteFeasible`, `PaletteNumberAttained`, `IsMinimalForbidden`, `IsTransversal`, `MinimalForbiddenCharacterisation`, `LocalRestriction`, `LocalDecomposability`, `LocalObstructionIsFullBlock`, `LocalPriceChain`, `IndependentBlockHazard`, `FullBlockCoverIff`, `CoverCostHazardBound`, `WholeGroundChromaticNumber`, `ElementaryPaletteFormula`, `TriangleNeedsThree`, `Benchmark816IsRealized`, `Benchmark816Palette`, `Benchmark816Cover`, `Benchmark816WholeGround`, `Benchmark816MinimalForbiddenCounts`, `Benchmark816Price`, `Benchmark816UnionBound`; 12 proved (main 91cffbb5, PR #358, not on main): `benchmark_block_and_ground_sizes`, `four_block_supports_fifteen`, `minimal_forbidder_counts`, `palette_816_from_two_label_classes`, `palette_815_fails_by_counting`, `whole_ground_chromatic_818`, `elementary_palette_formula_instances`, `macro_triangle_three_colors`, `ceil_block_over_capacity_instances`, `block_occupancy_one_percent`, `low_failure_union_bound_arithmetic`, `cover_price_numerator_and_denominator_positive`; 20 proved (proof packet): `P15Elementary.triangleNeedsThree`, `P15Structure.benchmark816IsRealized`, `P15Structure.benchmark816Price`, `P15Structure.elementaryPaletteFormula`, `P15Structure.triangleNeedsThree`, `P15Structure.benchmark816Palette`, `P15Structure.localDecomposability`, `P15Structure.localObstructionIsFullBlock`, `P15Structure.fullBlockCoverIff`, `P15Structure.wholeGroundChromaticNumber`, `P15Structure.benchmark816WholeGround`, `P15Structure.minimalForbiddenCharacterisation`, `P15Structure.benchmark816MinimalForbiddenCounts`, `P15Hazard.localPriceChain`, `P15Hazard.independentBlockHazard`, `P15Covers.benchmark816Cover`, `P15Covers.localRestriction`, `P15Covers.paletteNumberAttained`, `P15Covers.benchmark816UnionBound`, `P15Budget.coverCostHazardBound` |
| `p15-price-boundary` | landing_claim | EXACT_COUNTEREXAMPLE | `specified` | `p15-price-boundary` | 3 specified: `TwoCoordGeneratorPrices`, `DemandOneRefutation`, `LogOnePlusBelowIdentity`; 6 proved (main 91cffbb5, PR #358, not on main): `smallest_member_block_size`, `demand_one_good_sets`, `demand_one_obstruction`, `good_measure_three_quarters`, `generator_prices_ninths`, `minimum_generator_price_exceeds_third`; 3 proved (proof packet): `P15Elementary.twoCoordGeneratorPrices`, `P15Elementary.logOnePlusBelowIdentity`, `P15Elementary.demandOneRefutation` |
| `p15-price-budget-restricted` | landing_claim | HOLD_WITH_DOMAIN | `specified` | `p15-price-budget` | 8 specified: `LogRatioBound`, `RatioExponentBound`, `RestrictedLocalBudget`, `RestrictedTransformedPriceBudget`, `RhoTestSufficient`, `CapTestAtTwoThirds`, `Benchmark816TransformedRange`, `Benchmark816RationalPrice`; 3 proved (main 91cffbb5, PR #358, not on main): `t3_ratio_at_minimal_block`, `t3_bound_finite_instances`, `explicit_rational_price_above_p`; 8 proved (proof packet): `P15Elementary.logRatioBound`, `P15Elementary.capTestAtTwoThirds`, `P15Elementary.ratioExponentBound`, `P15Structure.benchmark816RationalPrice`, `P15Budget.restrictedLocalBudget`, `P15Budget.rhoTestSufficient`, `P15Budget.restrictedTransformedPriceBudget`, `P15Budget.benchmark816TransformedRange` |
| `p15-full-price` | landing_claim | REVIEWED_SCOPED | `specified` | `p15-full-price` | 20 specified: `FullTransformedPriceBudget`, `RhoStarEnclosure`, `RhoStarBelowSixSevenths`, `IsProperDownset`, `HazardStarPositive`, `CoordinatewiseConcavity`, `HazardProductBound`, `HazardPhiBound`, `FullGroundPriceBound`, `PhiAtPStar`, `FullGroundBudgetIff`, `CapacityHazardIsBinomial`, `OddMajorityIdentity`, `CapacityWorstCase`, `HStarExceedsOne`, `LocalPriceBoundFull`, `GlobalAssemblyHazard`, `SharpnessOfRhoStar`, `DemandOneBoundary`, `SixSeventhsCertificate`; 7 proved (main 91cffbb5, PR #358, not on main): `rho_star_endpoints_adjacent`, `rho_star_upper_endpoint_below_six_sevenths`, `sixteen_twentysevenths_below_six_sevenths`, `minimal_block_size_instances`, `e_rational_bounds_ordering`, `exp_eleven_sixths_taylor_certificate`, `h_star_rational_certificate_constants`; 19 proved (proof packet): `P15Elementary.rhoStarBelowSixSevenths`, `P15Elementary.sixSeventhsCertificate`, `P15Elementary.hStarExceedsOne`, `P15Elementary.phiAtPStar`, `P15Elementary.hazardStarPositive`, `P15Elementary.oddMajorityIdentity`, `P15Elementary.capacityWorstCase`, `P15Elementary.demandOneBoundary`, `P15Elementary.capacityHazardIsBinomial`, `P15Elementary.sharpnessOfRhoStar`, `P15Hazard.globalAssemblyHazard`, `P15Interpolation.rhoStarEnclosure`, `P15Interpolation.coordinatewiseConcavity`, `P15Interpolation.hazardProductBound`, `P15Interpolation.hazardPhiBound`, `P15Interpolation.fullGroundPriceBound`, `P15Interpolation.fullGroundBudgetIff`, `P15Interpolation.localPriceBoundFull`, `P15Interpolation.fullTransformedPriceBudget` |
| `downstream-hard-gate` | landing_claim | ENGINEERING_HOLD | `none` | `downstream-graph` | — |
| `math.uniform-matrix-cap-lifetime` | graph_node | — | `specified` | `lifetime-parent` | 32 specified: `IsLPeriodic`, `IsSmooth`, `GaussianFieldInterface`, `IsCriticalPoint`, `HessianHasInertia`, `IsNondegMax`, `IsIndexSaddle`, `IsCandidatePair`, `IsMorseDistinct`, `SuperlevelBarInterface`, `IsFiniteBar`, `IsLifetimeDensityVersion`, `IsLPeriodic`, `IsSmooth`, `GaussianFieldInterface`, `IsCriticalPoint`, `HessianHasInertia`, `IsNondegMax`, `IsIndexSaddle`, `IsCandidatePair`, `IsMorseDistinct`, `SuperlevelBarInterface`, `IsFiniteBar`, `IsLifetimeDensityVersion`, `IsWindowPair`, `IsWindowElderPair`, `IsUnrestrictedPair`, `PairLawInterface`, `ParentUniformSelectionA`, `ParentCompactWindowDensityB`, `IsUnrestrictedLeadingCoefficient`, `ParentUnrestrictedLeadingDensityC` |
| `math.d1-component.congruence-erratum` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d1-component.section9-replacement-v1_1` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d1-component.marked-cylinder-cap` | graph_node | — | `none` | `marked-cylinder-cap` | — |
| `math.d1-component.reconciliation-record` | graph_node | — | `none` | `downstream-graph` | — |
| `math.lifetime-remainder` | graph_node | — | `specified` | `lifetime-remainder` | 1 specified: `RemainderBoundedR1` |
| `math.side24-coefficient` | graph_node | — | `specified` | `side24-proof` | 3 specified: `PeriodicCoefficientInterface`, `CoefficientEnclosure`, `ReferenceComparison` |
| `math.rn-count-interface` | graph_node | — | `none` | `rn-count-interface` | 5 specified: `HolderCountBoundN1`, `HolderConsequenceN2`, `ConditionalMeanIdentityN3`, `UniformCounterexampleN4`, `ExponentialTailLayerCakeN5`; 5 proved (proof packet): `RNProbability.conditionalMeanIdentityN3`, `RNProbability.holderCountBoundN1`, `RNProbability.holderConsequenceN2`, `RNProbability.uniformCounterexampleN4`, `RNProbability.exponentialTailLayerCakeN5` |
| `math.rn-fixed-remote-window` | graph_node | — | `specified` | `rn-fixed-remote-window` | 3 specified: `PinnedLawInterface`, `FixedRemoteMeanMeasureA2`, `FixedRemoteCountBoundA3` |
| `math.rn-region.fixed-remote` | graph_node | — | `specified` | `downstream-graph` | 2 specified: `FixedRemoteMeanMeasureA2`, `FixedRemoteCountBoundA3` |
| `math.rn-region.mesoscopic-scaled-annulus` | graph_node | — | `specified` | `d5-reconciliation` | 2 specified: `D5dSummedPinNeighbourhoodC2`, `AnnulusBridgeAllHeight2` |
| `math.rn-region.pin-collision` | graph_node | — | `specified` | `d5-reconciliation` | 3 specified: `D5aPuncturedPinDisks`, `D5bMicrodisk`, `D5dSummedPinNeighbourhoodC2` |
| `math.rn-region.intermediate-r-to-rho` | graph_node | — | `specified` | `d5-reconciliation` | 1 specified: `D5eIntermediateShellsI3I4` |
| `math.rn-region.witness-collision` | open_obligation | — | `specified` | `d5-reconciliation` | 2 specified: `WitnessCollisionNearPinPairsTarget`, `RemoteCollisionSecondFactorialD` |
| `math.rn-mesoscopic-reduction` | graph_node | — | `none` | `downstream-graph` | — |
| `math.p15-full-price` | graph_node | — | `specified` | `p15-full-price` | 3 specified: `FullTransformedPriceBudget`, `RhoStarEnclosure`, `SharpnessOfRhoStar`; 7 proved (main 91cffbb5, PR #358, not on main): `rho_star_endpoints_adjacent`, `rho_star_upper_endpoint_below_six_sevenths`, `sixteen_twentysevenths_below_six_sevenths`, `minimal_block_size_instances`, `e_rational_bounds_ordering`, `exp_eleven_sixths_taylor_certificate`, `h_star_rational_certificate_constants`; 3 proved (proof packet): `P15Elementary.sharpnessOfRhoStar`, `P15Interpolation.rhoStarEnclosure`, `P15Interpolation.fullTransformedPriceBudget` |
| `math.p15-price-boundary` | graph_node | — | `specified` | `p15-price-boundary` | 2 specified: `TwoCoordGeneratorPrices`, `DemandOneRefutation`; 6 proved (main 91cffbb5, PR #358, not on main): `smallest_member_block_size`, `demand_one_good_sets`, `demand_one_obstruction`, `good_measure_three_quarters`, `generator_prices_ninths`, `minimum_generator_price_exceeds_third`; 2 proved (proof packet): `P15Elementary.twoCoordGeneratorPrices`, `P15Elementary.demandOneRefutation` |
| `math.rn-selector-region-crosswalk` | graph_node | — | `none` | `downstream-graph` | — |
| `math.rn-fixed-annulus-window` | graph_node | — | `specified` | `rn-fixed-annulus-window` | 1 specified: `FixedAnnulusWindowCandidateA2` |
| `math.rn-region.fixed-annulus-window` | graph_node | — | `specified` | `downstream-graph` | 1 specified: `FixedAnnulusWindowCandidateA2` |
| `math.d5-pin-neighborhood-first-moment` | graph_node | — | `specified` | `d5-reconciliation` | 6 specified: `D5aPuncturedPinDisks`, `D5bMicrodisk`, `D5cCollarC1`, `D5dSummedPinNeighbourhoodC2`, `D5eIntermediateShellsI3I4`, `D5fGlobalWindowFirstMomentI5` |
| `math.d5-component.punctured-pin-proof` | graph_node | — | `specified` | `d5-reconciliation` | 2 specified: `D5aPuncturedPinDisks`, `D5bMicrodisk` |
| `math.d5-component.punctured-pin-review-111` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.punctured-pin-continuum-review` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.punctured-pin-continuum-check` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.collar-review` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.annulus-review` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.i5-review-xai` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.i5-review-claude` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.offpin-second-moment-review` | graph_node | — | `none` | `downstream-graph` | — |
| `math.d5-component.collar-proof` | graph_node | — | `specified` | `d5-reconciliation` | 1 specified: `D5cCollarC1` |
| `math.d5-component.annulus-proof` | graph_node | — | `specified` | `rn-annulus-bridge` | 1 specified: `AnnulusBridgeAllHeight2` |
| `math.d5-component.intermediate-window-proof` | graph_node | — | `specified` | `d5-reconciliation` | 2 specified: `D5eIntermediateShellsI3I4`, `D5fGlobalWindowFirstMomentI5` |
| `math.d5-component.remote-window-proof` | graph_node | — | `specified` | `rn-fixed-remote-window` | 2 specified: `FixedRemoteMeanMeasureA2`, `FixedRemoteCountBoundA3` |
| `D1-THM-A` | proof_index_result | — | `specified` | `lifetime-parent` | 2 specified: `PairLawInterface`, `ParentUniformSelectionA` |
| `D1-THM-B` | proof_index_result | — | `specified` | `lifetime-parent` | 4 specified: `IsWindowPair`, `IsWindowElderPair`, `IsLifetimeDensityVersion`, `ParentCompactWindowDensityB` |
| `D1-THM-C` | proof_index_result | — | `specified` | `lifetime-parent` | 3 specified: `IsUnrestrictedPair`, `IsUnrestrictedLeadingCoefficient`, `ParentUnrestrictedLeadingDensityC` |
| `OA-ELDER-LOWER-DENSITY-GAP-20260928-v1` | proof_index_result | — | `specified` | `elder-lower-density-gap` | 3 specified: `PlanarElderLowerE2`, `PlanarTwoSidedRateE4`, `PlanarDensityGapE11` |
| `two-scale-s6-s21` | proof_index_result | — | `none` | `—` | — |
| `axial-density` | proof_index_result | — | `none` | `—` | — |
| `fixed-transverse-chart` | proof_index_result | — | `none` | `—` | — |
| `CL-D5-DIMENSION-LIFT-20260929-v1` | proof_index_result | — | `specified` | `d5-dimension-lift` | 4 specified: `DimensionLiftPuncturedPinBallP`, `DimensionLiftCompactCollarC`, `DimensionLiftIntermediateShellsI`, `DimensionLiftGlobalWindowG` |
| `OA-C6-FOURIER-20260929-v1` | proof_index_result | — | `none` | `c6-fourier-cutoff` | 1 specified: `FourierCutoffPlanarSecondFactorialF4` |
| `CL-C6-PALM-20260929-v1` | proof_index_result | — | `specified` | `c6-palm-route` | 2 specified: `PalmRouteFactorialMomentsQ`, `PalmRouteSecondFactorialTheta` |
| `OA-C6-RARE-CLUSTER-20260929-v1` | proof_index_result | — | `none` | `c6-rare-cluster` | 1 specified: `RareClusterPoissonObstructionO`; 1 proved (proof packet): `RNCluster.rareClusterPoissonObstructionO` |
| `CL-D5-REMOTE-COLLISION-20260928-v1` | proof_index_result | — | `none` | `remote-collision` | 1 specified: `RemoteCollisionSecondFactorialD` |
| `cumulative-transfer-correction` | proof_index_result | — | `none` | `cumulative-transfer-correction` | — |
| `hist.D0-RN-UNIF-historical-predicates` | open_obligation | — | `none` | `downstream-graph` | — |
| `lean:main-formal-UniversalLaw` | existing_lean_package | — | `none` | `side24-enclosure` | 4 proved (main, origin/main f6deeba7): `d2_endpoints_adjacent`, `d3_endpoints_adjacent`, `d3_below_d2`, `endpoints_in_unit_tenth` |
| `lean:math-formal-ResearchFormalCoreR1` | existing_lean_package | — | `none` | `—` | — |
| `lean:math-companion-d2_moment_bridge_20261006` | existing_lean_package | — | `none` | `—` | — |
| `lean:math-companion-d2_square_schur_20261006` | existing_lean_package | — | `none` | `—` | — |
| `lean:math-companion-d2_atom_integral_20261006` | existing_lean_package | — | `none` | `—` | — |
| `lean:math-frontier-cap_first_exit_lean_20261002` | existing_lean_package | — | `none` | `marked-cylinder-cap` | — |
| `lean:math-w3scratch-side24-proposal` | existing_lean_package | — | `none` | `—` | — |
| `main:experiments/universality/finite_gaussian_contact` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_candidate_lifetime` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_fold_chart` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_weighted_fold_window` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_contact_genericity` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_elder_transfer` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/finite_h0_lifetime` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/fold_structural_universality` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/periodized_gaussian_h0` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/universality/iid_short_bar_process` | main_experiment_proof | — | `none` | `—` | — |
| `main:experiments/periodic_h0/spectral_shrinking_bins` | main_experiment_proof | — | `none` | `—` | — |
| `main:unmerged-codex-prs-330-350` | main_experiment_proof | — | `none` | `—` | — |
| `cert:main-periodic_h0-exact_h0_1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-certificate8` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-nodal1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-hessian1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-gaussian_tail1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-gaussian_coupling1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-word_polynomial1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-finite_count_loss1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-periodic_h0-count_confidence1` | exact_certificate | — | `none` | `—` | — |
| `cert:main-universality-fold_controls` | exact_certificate | — | `none` | `—` | — |
| `cert:math-side24_v1-coefficient.py` | exact_certificate | — | `none` | `side24-enclosure` | — |
| `cert:math-full_price-full_price.py` | exact_certificate | — | `none` | `p15-full-price` | — |

Packets listed by PROOF_INDEX by id only (82 rows, kind `proof_index_result`, formal progress `none`, no statements, bytes not pinned here; each row's `not_established` says the statement was not extracted and no verdict is implied):

`CL-SIDE24-PFV-20260929-v1`, `OA-SIDE24-REMAINDER-20260929-v1`, `CL-SIDE24-PERIODIZATION-REMAINDER-20260929-v1`, `OA-CONTACT-ANGULAR-TAIL-20260925-v1`, `OA-PIN-MICRO-COV-20260926-v1`, `OA-D5-MICRODISK-20260926-v1`, `D5-PIN-MICRODISK-20260927-v1`, `OA-BF-SIX-PIN-HESSIAN-20260926-v1`, `OA-CONTACT-KERNEL-SUBSTITUTE-20260926-v1`, `OA-D5-SHORT-EDGE-20260928-v1`, `CL-D5-PIN-DISK-20260928-v1`, `OA-D5-REMOTE-SINGLETON-LAW-20260928-v1`, `OA-REMOTE-HEIGHT-DECOUPLING-20260928-v1`, `OA-WINDOW-MULTIPLICITY-LOCAL`, `OA-WINDOW-MULTIPLICITY-REMOTE`, `OA-WINDOW-MULTIPLICITY-DISTANCE`, `OA-WINDOW-MULTIPLICITY-HEIGHT`, `OA-REMOTE-INVERSE-SEPARATION-20260928-v1`, `OA-ELDER-DIMENSION-LIFT-20260928-v1`, `CL-ELDER-LOWER-ALL-D-20260929-v1.1`, `CL-C6-FACTORIAL-20260929-v1`, `CL-C6-SHARPENED-20260929-v1`, `OA-C6-COUNT-CAP-BOUNDARY-20260929-v1`, `OA-LOCAL-CLUSTER-TIGHTNESS-20260929-v1`, `OA-REMOTE-MIXED-INVERSE-20260929-v1`, `CL-D1-UNRESTRICTED-DIFFERENCE-20260929-v3`, `CL-C7-TOTAL-BOUNDED-20260929-v1.2`, `OA-C7-ZERO-GAP-LIMIT-20260929-v1`, `OA-SARD-ROBUST-CHARTS-20260929-v1`, `OA-SARD-A2-FIXED-FRAME-20260929-v2`, `packet:sard_g_a2_regularity_20260929`, `packet:sard_g_pinned_transfer_20260929`, `OA-P15-BIPARTITE-OVERLAP-20260928-v1`, `OA-P15-TAIL-LOAD-20260928-v1`, `packet:planar_cubic_cluster_20260929`, `packet:c6_marked_fourier_20260929`, `packet:c6_cluster_law_20260929`, `packet:cluster_exponential_weight_20260930`, `packet:spectral_cluster_closure_20260929`, `packet:lifetime_moment_boundary_20260930`, `packet:radial_rate_20260930`, `packet:extreme_cluster_sampling_20260930`, `packet:fixed_r_inverse_lifetime_20260930`, `packet:local_elder_geometry_20260930`, `packet:concave_fibre_elder_20260930`, `packet:elder_partner_extremes_20260930`, `packet:remainder_vanishing_20260930`, `packet:elder_selected_extremes_20260930`, `packet:remainder_rate_20260930`, `packet:far_elder_rate_20260930`, `packet:short_bar_endpoint_law_20261001`, `packet:c2_finite_jet_transfer_20261001`, `packet:cusp_second_order_20261001`, `packet:candidate_third_order_20261001`, `packet:elder_third_order_20261001`, `packet:third_order_rate_20261001`, `packet:elder_cusp_parity_20261002`, `packet:separated_fold_multiplicity_20261002`, `packet:subpin_collision_20261002`, `packet:soft_rejected_pairs_20261002`, `packet:soft_fold_limit_20261002`, `packet:soft_closed_form_20261002`, `packet:cusp_torus_transfer_20261001`, `packet:moving_remote_collision_20261003`, `packet:ordinary_short_bar_occurrence_20261001`, `packet:replacement_bar_occurrence_20261001`, `packet:strict_unique_replacement_coefficient_20261003`, `packet:unique_replacement_bar_intensity_20261001`, `packet:planar_rejected_endpoint_margin_20261003/A4`, `packet:planar_soft_layer_chain_20261003/C91`, `packet:planar_soft_layer_chain_20261003/C92`, `packet:planar_soft_layer_chain_20261003/C93`, `packet:planar_soft_layer_chain_20261003/C94`, `packet:planar_soft_layer_chain_20261003/C95`, `packet:planar_soft_layer_chain_20261003/C96`, `packet:planar_soft_layer_chain_20261003/C97`, `packet:planar_soft_layer_chain_20261003/C98`, `packet:planar_soft_layer_chain_20261003/C99`, `packet:planar_soft_layer_chain_20261003/C101`, `packet:planar_soft_layer_chain_20261003/C102`, `packet:planar_soft_layer_chain_20261003/C103`, `packet:planar_soft_layer_chain_20261003/C124`.

## Frontier: the 64 registered Props without a theorem

At the Math- cut `8acc4856` the sibling proof packet proves 77 of the 141 registered Props. This section
classifies the 64 others, so that the remaining formal work is named where the register lives. The
classification is this packet's author's reading of the statement modules (the round-4 triage of 2026-10-11,
re-applied at this cut; the statement modules have not changed since `873c6c0`, so the cited lines hold),
by a single reader, not a reviewed fact; it moves no label: every one of the 64 stays `specified`, nothing is
promoted, reclassified or withdrawn. Scientific effect NONE.

Three classes, by what a proof would need:

- **vocabulary** (36): a reusable `def … : Prop` with explicit non-instance arguments, used by the modules inside
  definitions, as a hypothesis (`IsCriticalPoint d g x →`, `IsUnitTuple d q u →`, …) and in conclusions
  (`IsUnrestrictedLeadingCoefficient` is the conclusion of C at Lifetime.lean:544, `IsLifetimeDensityVersion` an
  existential conclusion of B at Lifetime.lean:502-503, `IsRealizedFamily` is concluded for `benchmark816` at
  P15.lean:521, `NegDefCone` supplies the indicator set at Side24.lean:683). Its unrestricted universal closure is
  false or empty, so a theorem of exactly its type is not a target; instantiated or conditional theorems about it are
  not excluded by this grouping. It is counted among the 141 because the register inventories every `Prop`-valued
  definition. Under this grouping, a planning convention of this section and not a proof that other meaningful
  targets are exhausted, 105 Props are targets, 77 are proved and 28 remain.
- **interface-blocked** (24): the Prop quantifies over every inhabitant of a pinned interface whose fields do not
  determine the conclusion. For each, a (mathematical, not kernel-checked) inhabitant that satisfies every field and
  violates the conclusion is named below; these are non-kernel mathematical judgements of a single reader. They refute
  the unrestricted universal closure for the given outer objects and nothing more: the Side24 scaling inhabitant shows
  that universal validity of either Prop would force the interface empty, without constructing an inhabitant; the
  Lifetime `p ≡ 0` / `p ≡ 1` refutations hold for given admissible `G`, `I` and do not construct those; the RN
  refutations are read on a common path space with a parameter-indexed family, finite counted sets and the stated
  region and window hypotheses, and the preserved RN probe supplies conditional Dirac, count and null-set reductions,
  not a formal countermodel receipt. So a theorem of exactly the Prop's type is not available without new fields; the
  route is a numbered successor interface carrying the missing field (a Kac-Rice intensity, the law of `p`, the
  conditional Hessian law), followed by successor Props. Nothing here says the informal source is wrong: the source's
  hypotheses are exactly what the interface does not transcribe.
- **Mathlib-blocked** (4): the interface fixes the Prop's truth value and no interface counterexample was identified;
  the statement follows the pinned source, and a proof is expected to need substantial Gaussian-field and density
  formalisation beyond the primitives Mathlib `d13f23b7` has (an effort estimate of weeks each, not an impossibility
  theorem). Provability is conditional on the corresponding informal claims, as the round-4 triage qualified it; the
  machinery named in a row is the proposed proof route, not a conjunct of the target's conclusion.

| Prop (`UniversalLaw.Spec.`) | defined at | class | what a proof would need |
|---|---|---|---|
| `Field.IsLPeriodic` | Field.lean:99 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsSmooth` | Field.lean:108 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsCriticalPoint` | Field.lean:169 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.HessianHasInertia` | Field.lean:182 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsNondegMax` | Field.lean:195 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsIndexSaddle` | Field.lean:205 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsCandidatePair` | Field.lean:272 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsMorseDistinct` | Field.lean:283 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsFiniteBar` | Field.lean:333 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Field.IsLifetimeDensityVersion` | Field.lean:370 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsLPeriodic` | Lifetime.lean:127 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsSmooth` | Lifetime.lean:136 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsCriticalPoint` | Lifetime.lean:197 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.HessianHasInertia` | Lifetime.lean:210 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsNondegMax` | Lifetime.lean:223 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsIndexSaddle` | Lifetime.lean:233 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsCandidatePair` | Lifetime.lean:300 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsMorseDistinct` | Lifetime.lean:311 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsFiniteBar` | Lifetime.lean:361 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsLifetimeDensityVersion` | Lifetime.lean:398 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsUnrestrictedPair` | Lifetime.lean:440 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsUnrestrictedLeadingCoefficient` | Lifetime.lean:522 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsWindowPair` | Lifetime.lean:417 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Lifetime.IsWindowElderPair` | Lifetime.lean:428 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `RN.IsCriticalPoint` | RN.lean:146 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `RN.HessianHasInertia` | RN.lean:159 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Side24.IsUnitTuple` | Side24.lean:209 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Side24.NegDefCone` | Side24.lean:673 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsRealizedFamily` | P15.lean:175 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsGood` | P15.lean:195 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsKDecomposable` | P15.lean:214 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsGeneratorCover` | P15.lean:236 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsPaletteFeasible` | P15.lean:296 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsMinimalForbidden` | P15.lean:341 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsTransversal` | P15.lean:352 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `P15.IsProperDownset` | P15.lean:865 | vocabulary | Reusable predicate with explicit non-instance arguments, used in definitions, hypotheses and conclusions of the modules; its unrestricted universal closure is false or empty, so a theorem of exactly its type is not a target (instantiated or conditional theorems about it are not excluded). |
| `Side24.CoefficientEnclosure` | Side24.lean:825 | interface-blocked | `PeriodicCoefficientInterface` (Side24.lean:770-822) leaves `coneMoment` (788) and `c24` (772) free up to `eq_15_2` (812-815): doubling both gives another inhabitant, so a theorem of this type would force the interface to be empty. Missing field: the conditional law of the transverse Hessian block conditioned on the pin `V_u = H_f u = 0` under `K24`, which pins `coneMoment`. |
| `Side24.ReferenceComparison` | Side24.lean:835 | interface-blocked | Same mechanism as `CoefficientEnclosure`: `|c24 2 / cRef 2 - 1| < 10^-106` and the doubled inhabitant's `|2 q - 1| < 10^-106` refute each other. Missing field: the same conditional law; the arithmetic is trivial. |
| `RN.AnnulusBridgeAllHeight2` | RN.lean:617 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). A Dirac law at a smooth periodic pinned path with one extra nondegenerate minimum at `8 r u` in the annulus has mean count >= 1 against a bound `C r^3 vol(E0)` that vanishes on a point. |
| `RN.D5aPuncturedPinDisks` | RN.lean:638 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Extra minimum at `M + r^2 u`, `E0` a single point of volume 0: left side >= 1, right side 0. |
| `RN.D5bMicrodisk` | RN.lean:659 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). With kappa = 1 and an extra minimum at `M + r^2 u`, the count is >= 1 while `C r^5 -> 0`. |
| `RN.D5cCollarC1` | RN.lean:678 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). The midpoint 0 lies in every admissible collar; an extra minimum there with `E0 = {0}` gives >= 1 against `C r^3 * 0`. |
| `RN.D5dSummedPinNeighbourhoodC2` | RN.lean:696 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). No volume factor: an extra minimum at the midpoint gives >= 1 against `C r^3 -> 0`. |
| `RN.D5eIntermediateShellsI3I4` | RN.lean:715 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). An extra window minimum at `8 r u` (height `b - k r^3/2`) lies in the shell for `A0 = 4`, `rho = s0`; right side `C r^3 (1/16 + s0^2) -> 0`. |
| `RN.D5fGlobalWindowFirstMomentI5` | RN.lean:733 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Conjunct (i) fails for the extra-minimum Dirac law (count >= 1 > C r^3); conjunct (ii)'s lower bound fails for the Dirac law with no window critical point (probability 0 < c r^3) and its upper bound for the extra-minimum law (probability 1). |
| `RN.DimensionLiftCompactCollarC` | RN.lean:780 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). General-d `D5cCollarC1` (`2 <= d`): same midpoint inhabitant with an index-(d-1) saddle at `S`; the lemmas used are dimension-free. |
| `RN.DimensionLiftGlobalWindowG` | RN.lean:823 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). General-d conjunct (i) of `D5fGlobalWindowFirstMomentI5`: extra window minimum at the midpoint, count >= 1 > C r^3. |
| `RN.DimensionLiftIntermediateShellsI` | RN.lean:800 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Either conjunct fails for the extra minimum at `8 r u`: take `s = 8 r` for (1.5) or `A0 = 4`, `rho = s0` for (1.6); both right sides -> 0. |
| `RN.DimensionLiftPuncturedPinBallP` | RN.lean:757 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). The second inequality (`C r^3 \int omegaD <= C' r^3`) is provable on its own; the first fails as `D5aPuncturedPinDisks` with `E0 = {r e_0}` (set integral over a point is 0). |
| `RN.FixedAnnulusWindowCandidateA2` | RN.lean:596 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). No absolute-continuity field either: a Dirac law at a pinned path with an extra nondegenerate index-j window critical point in the annulus, `E0` that point, gives >= 1 against `C k r^3 * 0`. |
| `RN.FixedRemoteCountBoundA3` | RN.lean:571 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Extra index-j window critical point at `x0 = (L/2) e_0` in the remote region, `A = {x0}` of volume 0: both left sides 1, both right sides 0. |
| `RN.FixedRemoteMeanMeasureA2` | RN.lean:548 | interface-blocked | The field `Lambda` (RN.lean:382) has no defining property in the structure, so any inhabitant can be copied with `Lambda = 0`, and `0 < c_Lambda <= Lambda` fails on the nonempty remote region. `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333).  |
| `RN.FourierCutoffPlanarSecondFactorialF4` | RN.lean:844 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). A two-point law `(1 - p_r) delta_g0 + p_r delta_g1` with `n_r >= 1/r` window critical points on `g1` and `p_r = r^3 / n_r` satisfies the first-moment hypothesis `G_2` (`E N = r^3`) while `E N(N-1) >= r^2 - r^3` exceeds `C r^3 log(1/r)` for small `r`; here `n_r` is the exact finite total `torusWindowCount` (with a zero-count branch), uniformly over the antecedent's marks and frames. |
| `RN.PalmRouteFactorialMomentsQ` | RN.lean:868 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). A Dirac law at a pinned path with exactly two extra window critical points has `(N)_2 = 2` for every `r`, against `C_q r^3 -> 0`. |
| `RN.PalmRouteSecondFactorialTheta` | RN.lean:885 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Two-sided: the pinned path with no window critical point gives `E (N)_2 = 0 < c r^3`; the two-extra-point path gives `2 > C r^3`. |
| `RN.RemoteCollisionSecondFactorialD` | RN.lean:960 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Two extra window critical points at remote `x0`, `x1` and `A = {x0, x1}` of volume 0: `(N)_2 >= 2` and the event `2 <= N` has mass 1, against right sides 0. |
| `RN.WitnessCollisionNearPinPairsTarget` | RN.lean:1001 | interface-blocked | `PinnedLawInterface` (RN.lean:339-382) constrains the sample paths and the law only at the two pins (`Q_pinned` 354, `W_typed` 366, `Z_pos` 370, `QW_eq` 375); the Kac-Rice representation is deliberately not transcribed (RN.lean:325-333). Two extra window critical points between the pins within torus distance `eta` of `pinM` give `nearPinPairCount = 2` for every small `r`, against `C r^3 -> 0`. (The source marks this target OPEN; the interface cannot carry it either way.) |
| `Lifetime.ParentUniformSelectionA` | Lifetime.lean:479 | interface-blocked | The only fields of `PairLawInterface` are `p_nonneg` (Lifetime.lean:463) and `p_le_one` (465); nothing ties `p` to `G` or `I`. The inhabitant `p = 0` breaks `1 - p <= C r^3` (probe Lifetime-B_1 elaborates the refutation). Missing field: the law of `p` itself. |
| `Lifetime.PlanarElderLowerE2` | Lifetime.lean:581 | interface-blocked | Same interface: `p = 1` makes `1 - p = 0`, so `c r^3 <= 1 - p` fails for every `c > 0` (probe Lifetime-B_1). |
| `Lifetime.PlanarTwoSidedRateE4` | Lifetime.lean:598 | interface-blocked | Same interface: `p = 1` satisfies the hypothesis `ParentUniformSelectionA` (with `C = 0`) and breaks the lower half of the conclusion (probe Lifetime-B_1). |
| `Lifetime.ParentCompactWindowDensityB` | Lifetime.lean:498 | Mathlib-blocked | Theorem B (1.2) as the Prop states it: versions characterised by the integral identity `IsLifetimeDensityVersion` (Lifetime.lean:398; no Kac-Rice representation and no coefficient formula is transcribed), the `c l^(-1/3)` asymptotics with `c > 0`, and the `O(l^(2/3))` gap. The `gaussian` field (Lifetime.lean:175-177) fixes the law, so the truth value does not depend on the inhabitant. Expected proof route (dependencies, not conclusion conjuncts): a Kac-Rice representation, a.s. finiteness and integrability of the pair counts, substantial Gaussian-field formalisation beyond the primitives of Mathlib `d13f23b7`; conditional on the informal claim. |
| `Lifetime.ParentUnrestrictedLeadingDensityC` | Lifetime.lean:542 | Mathlib-blocked | Theorem C (1.3) as the Prop states it: some `c` with `IsUnrestrictedLeadingCoefficient d L G I c` (Lifetime.lean:522-544), i.e. versions with the small-`l` asymptotics and a positive constant. Expected proof route (dependencies, not conclusion conjuncts): a.s. Morse genericity (so the elder relation is pinned a.s.) and the off-diagonal domination, Gaussian-field formalisation beyond the pinned Mathlib; conditional on the informal claim. |
| `Lifetime.PlanarDensityGapE11` | Lifetime.lean:623 | Mathlib-blocked | The `p`-hypothesis is inert: the free `p = 1` satisfies it (probe Lifetime-B_1), so a theorem would be the unconditional two-sided planar density gap for every `(G, I)` at `d = 2`. Expected proof route: the (E2) planar lower argument plus the compact-window `O(l^(2/3))` upper bound, Gaussian-field analysis beyond the pinned Mathlib; conditional on the informal claims. |
| `Lifetime.RemainderBoundedR1` | Lifetime.lean:559 | Mathlib-blocked | Theorem R (R1), in its conditional form: for every `c` with the leading asymptotics (the hypothesis), versions exist with an `O(1)` remainder (the definition chooses the versions existentially, Lifetime.lean:562-568); vacuous if no such `c` exists. Leading asymptotics alone does not imply an essentially `O(1)` remainder; re-choosing versions can remove null-set exceptions but cannot repair failure of the bound on sets of positive measure. The content is the source's remainder analysis, which has no Mathlib counterpart. |

Counts: 36 vocabulary + 24 interface-blocked + 4 Mathlib-blocked = 64; 64 + 77 = 141. The three Side24 Props the
round-4 triage filed as Mathlib-blocked (`ImageBound`, `EntrywiseCovarianceBound`, `CovarianceOrdering`) were proved
in round 6 by building the missing lemmas inside the proof packet, which is why they are absent here; the four
Lifetime entries above are not of that kind.

## What this packet is not

* Not a proof of anything. Every registered declaration is a definition or a structure. The
  source scan refuses `theorem`, `lemma`, `example`, `instance`, `axiom`, `sorry`, `native_decide`,
  `unsafe`, `opaque`, `partial`, `macro`, `notation`, `set_option` and the rest of the lexical list
  as whole tokens of comment-stripped code; the environment audit of `--execute` refuses, by
  Lean's `ConstantInfo`, any theorem, axiom, opaque constant, instance or definition whose type is
  a proposition that `structure`/`def` elaboration did not generate, and the build log is refused
  on `declaration uses` (sorry) and on Lean's "is a proposition; use `theorem`" linter.
* Not a receipt from a local run. `--execute --allow-dirty` was run once locally (2026-10-10, worktree
  UNVERIFIED: not evidence about any commit): `lake build` of the five modules passed against the
  pinned Mathlib `d13f23b7`; `lake env leanchecker Specifications` was killed by the sandbox's memory
  cgroup (exit 137 at 13.4 GB resident) before finishing, so `execute()` failed closed at that step
  and wrote no receipt. The later stages were then run with the same functions over the fresh oleans:
  `#check` of all 150 registered declarations, `#print axioms` (149 within `{propext,
  Classical.choice, Quot.sound}`, 1 within `{propext, Quot.sound}`), the environment audit (558
  constants, 0 refused: 141 statements, 9 structures, 123 auxiliary, 285 generated, equal to the
  register and the scan) and every negative control. The first receipt for the real packet is the
  hosted `lean-specifications.yml` run; its `leanchecker` step over `import Mathlib` has the
  `cap-first-exit-lean.yml` precedent (about 30 minutes on a 16 GB runner) within the 45-minute
  budget, and that budget and the runner's memory are untimed here.
* Not an alignment review. Whether a Prop says what its anchor says is the independent reviewer's
  question (DESIGN honesty checklist). The register only records `PENDING_INDEPENDENT_REVIEW`.
* Not a scientific status. Dispositions in the register are transcribed verbatim from
  `claims/LANDING_CLAIMS.json` at the pinned commit; the checker refuses any other value.
* Not changed by the proof packet. `specifications/lean_statement_proofs_20261011` depends on this
  packet by path and holds 78 theorems whose type is exactly one of 77 registered Props (P15, Side24
  and RN families; see the last column of the declaration tables). The register lists those theorems
  under the claim rows with label `proved`, the label of a committed proof package; `kernel-checked`
  appears only in that packet's hosted run receipts (`lean-specification-proofs.yml`; run 38121713840 receipted the 52
  theorems of PR head 910809d6 and run 38116277597 the first 38; each receipt's checked_commit is the synthetic merge of the PR head into the
  base that a pull_request run checks out, not the head itself: run 38121713840 PR head 910809d6, actual tested merge 5b198699…; run 38116277597
  PR head 00d2ac6e, actual tested merge c4c66bb8…; both with base b2d70d77… and each with the same tree as its head; the 8 theorems
  of `P15Interpolation`, added after PR head 79731b96 (run 38124262242, actual tested merge 318ff194…), are receipted by run 38149888699: PR head 0dcb0016, actual tested
  merge 4363b692… with base 79731b96…, the head of PR #422, and first by run 38149524759: PR head 88c21058, actual tested merge 3306233b…, same base; the 13 theorems
  of `Side24Analysis`, `Side24Interfaces` and `RNCluster`, added after PR head 12e072d6 (run 38150363063, actual tested merge 86a6e8c8…, receipt theorem_count 60), are receipted by run 38180705952: PR head 48bfcbae,
  actual tested merge 9833d6d9… with base 12e072d6…, the head of PR #423, receipt theorem_count 73, and were first receipted by that run; the 2 theorems
  of `Side24Covariance` and `Side24ConeMoment`, added after PR head be014794 (run 38187538744, actual tested merge 79d3964c…, receipt theorem_count 73), are receipted by run 38195810463: PR head c6ce0256,
  actual tested merge 1320c5b2… with base be014794…, the head of PR #434, receipt theorem_count 75, and were first receipted by that run; the 3 theorems
  of `Side24Images`, added after PR head c6ce0256 (run 38195810463, actual tested merge 1320c5b2…, receipt theorem_count 75), are receipted by run 38277458954: PR head 9e305ea5,
  actual tested merge 1376175c… with base a0323b94…, Math- main at the merge of PR #440 (the tested merge differing from the head only outside both packets), receipt theorem_count 78, and were first receipted by that run).
  The 150 declarations here remain `specified`; a kernel-checked theorem is evidence about its Prop
  only, never about the informal claim, its alignment (PENDING_INDEPENDENT_REVIEW), review or
  acceptance; no claim-level label is `proved`, including the one row (`p15-price-boundary`) whose
  claim-stating Prop `DemandOneRefutation` has a theorem.
* Not a kernel receipt for the research claims. `--execute` shows that the statements elaborate
  and depend on no axiom beyond `propext`, `Classical.choice`, `Quot.sound`; the receipt says
  `formalization_status: "specified"` and nothing stronger.
* Not evidence from a dirty worktree. A receipt produced with `--allow-dirty` records the
  worktree as unverified and is not evidence about any commit.

## How to run

```bash
# source mode (the optional positional `source` names the same default mode): identity against
# HEAD, lexical refusals, root imports, declaration inventory vs REGISTER.json and back,
# vocabulary, verbatim anchors (at least 16 characters), pinned bytes, lock, toolchain
python3 -B -S specifications/lean_statements_20261010/check.py
python3 -B -S specifications/lean_statements_20261010/check.py source

# unit tests and negative controls, both interpreter modes (no Lean needed)
python3 -B -S specifications/lean_statements_20261010/test_check.py
python3 -B -O -S specifications/lean_statements_20261010/test_check.py

# execution: fresh lake build, leanchecker, #check and #print axioms of every registered
# declaration, the environment audit of every constant the modules add, the negative controls
# (source-side, Lean-side and of the audit itself), receipt under .lake/evidence/ (needs the
# pinned toolchain and the Mathlib cache or a source build of Mathlib d13f23b7)
export PATH="$HOME/.elan/bin:$PATH"
python3 -B -S specifications/lean_statements_20261010/check.py --execute

# everything in order
bash specifications/lean_statements_20261010/replay.sh
```

Environment: Lean `leanprover/lean4:v4.34.1` (release commit `5045d0056413266e57c625dcd7c365b10e377c52`),
Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612`, full dependency lock in `lake-manifest.json`
(identical to `formal/lake-manifest.json` except the package name). CI uses
`leanprover/lean-action@f061402b660e0c34644504b324e830f2991d4865` with `use-mathlib-cache: 'true'`.
Without the Mathlib cache (hosts `cache.mathlib.org`, `release.lean-lang.org`,
`reservoir.lean-lang.org`), Mathlib must be built from source, which takes hours.
