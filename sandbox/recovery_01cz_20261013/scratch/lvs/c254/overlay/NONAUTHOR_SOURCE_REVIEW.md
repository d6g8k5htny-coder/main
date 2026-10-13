# C254 nonauthor source review

Disposition: **SCOPED SOURCE REVIEW COMPLETE; NO KNOWN REQUIRED SOURCE CORRECTION IDENTIFIED; COMPILATION HOLD**. This is not exact-source PASS, successful elaboration, kernel acceptance, independent organizational acceptance, or theorem completion. Compilation **NOT_RUN**; kernel audit **NOT_RUN**; local and hosted tests **NOT_RUN**; executed negative controls **NOT_RUN**; scientific effect **NONE**.

Reviewer: OpenAI/Codex `/root/c254_nonauthor_review`, fresh nonauthor subagent delegated by the parent for Dylan Roy. The reviewer did not author either candidate module or the contributor API reports. The reviewer is source-exposed and same-provider; organizational-independence credit **0**. This is AI source review, not human review. Author reports and API contributors were treated as inputs to verify, not as independent evidence of correctness.

Parent-reported claim: `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`, R17 Work Events 1559, expiry 2026-10-11T17:57:44Z. This reviewer did not independently read the register. The source-review checkpoint supplied by the parent was 16:25 UTC. The parent owns manifest integration and hosted admission separately.

## Exact source and exposure

Paths are relative to `work/continuation254/candidate/formal/` unless stated otherwise.

| File | Bytes | Lines substantively read | Independently observed SHA-256 |
|---|---:|---|---|
| `ResearchFormalCoreR1/TorusTopology.lean` | 11542 | **Entire file, 1–244**, including every statement and proof body | `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4` |
| `ResearchFormalCoreR1/ClosedSuperlevel.lean` | 7827 | **Entire file, 1–195**, including every statement and proof body | `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e` |

Also read in full: workspace `AGENTS.md`, C254 `PLAN.md`, `task-1-brief.md`, `TORUS_FILTRATION.md`, and `task-1-report.md`. The combined display of `QUOTIENT_API.md` and `FILTRATION_API.md` was partially truncated; only displayed portions count as exposure. Their key APIs were checked directly in primary sources below. A narrow memory keyword search returned no task-specific C254 evidence and was not used to decide mathematical findings. No broad memory or repository crawl was performed.

The cached C253 `Field.lean` was read at lines 20–75, 90–103 and 307–326, plus declaration-name search. Its entire bytes were hashed, but its entire text was **not** read. Observed SHA-256: `f22c91e082cd7e738ba47c9c3534e47ac687d066f0b78586d15103cf1132535e`. Lines 48, 56–57 and 99–100 directly confirm the candidate's ambient space, lattice vector and periodicity formulas. Lines 313–316 confirm that the relevant maximin interface existentially quantifies the higher endpoint.

## Concrete findings and dispositions

### F1 — Quotient construction and Field binding are appropriately concrete

`TorusTopology.lean:19–65` uses the actual Euclidean space and integer span of the scaled standard basis, then proves an integer-coordinate membership characterization. The formula at lines 22–26 literally matches Field lines 56–57; lines 123–130 literally match Field lines 99–100. The useful nontrivial adapter is the basis-sum identity at lines 43–58, followed by lattice membership at lines 60–65. The formula-only `rfl` declarations should not be represented as an elaborated theorem importing the externally owned Field module; the documentation correctly states this limitation.

At lines 88–99 the quotient equality orientation is `-x + y` in the lattice, hence `y = x + latticeVec n`. This agrees with pinned `QuotientAddGroup.eq`, generated from Coset/Defs lines 197–201. No sign reversal or unproved quotient-interface assumption was found. Lattice discreteness and full-span instances at lines 67–80 have corresponding pinned instances. Compactness at lines 112–117 follows the actual periodic compact-range theorem; path connectedness at lines 119–121 follows the quotient instance. The result is an `IsCompact univ` theorem, not a new registered `CompactSpace` instance.

**Disposition:** Mathematical construction and literal source binding acceptable within the proposed-source scope. No known correction. Scaled-coordinate simplification and instance elaboration remain uncompiled risks R1–R2 below.

### F2 — Specified starting representative and existential final lift are correct

`TorusTopology.lean:154–165` fixes the Euclidean starting point `M` and constructs the final point as the endpoint of the lifted path. Pinned Homotopy/Lifting lines 214 and 258–262 use the initial equality orientation `γ 0 = p e`, matching `γ.source`. The pinned library itself builds a `Path e (Γ 1)` by the same constructor shape at lines 279–282.

The constant-path uniqueness control at candidate lines 169–175 is mathematically valid: two lifts of the same constant projected path agree at time zero and therefore everywhere on the connected interval. Pinned Covering/Basic lines 368–370 provides exactly this uniqueness principle. Lines 179–188 then combine a duplicate quotient image with nonexistence of a lift ending at a different prescribed representative. The nonzero lattice-vector premise is explicit; this theorem is not a witness that distinct representatives exist in dimension zero.

The higher-endpoint equivalence at lines 192–212 projects a Euclidean path in one direction and uses the existential lift endpoint in the other. Descent identifies endpoint heights and every path height. Strict and closed specializations at lines 214–230 preserve this quantifier structure. There is no arbitrary-final-representative error.

**Disposition:** No known mathematical or inspected-API defect. Exact Path/ContinuousMap coercions and definitional reduction remain uncompiled risk R3. These statements transfer path predicates; they do not assert a path exists at the critical death height.

### F3 — Closed endpoints, negative parameter and inclusion direction are consistent

`ClosedSuperlevel.lean:20–21` defines `F(t) = {x | -t ≤ g x}`. For `s ≤ t`, the proof at lines 55–58 correctly uses `-t ≤ -s ≤ g x`, so inclusions run from `F(s)` into `F(t)`. The constant-height endpoint and parameter-sign controls at lines 25–46 reject strict endpoint substitution, `+t` substitution, and reversed inclusions. Own-height membership at lines 85–92 is correctly limited to point membership.

Closedness at lines 61–63 requires continuity of `g`. Continuous inclusions and their laws at lines 66–82 do not require it because the subspace inclusion itself is continuous. No missing continuity hypothesis was found in the candidate's path-predicate transfer, which only composes paths with the continuous quotient projection and lifts paths; scalar predicate transfer is pointwise.

The strict-threshold control at lines 95–102 proves density below zero and absence of a simultaneous `0 ≤ k < 0` witness. It is a valid elementary countercontrol, but it does not formalize a supremum statement, a compact Morse counterexample, or endpoint nonattainment for the target research problem. The author report acknowledges this limit. Essentiality at lines 49–52 and torus lines 232–242 correctly excludes a strictly higher endpoint at a global maximum; neither supplies a finite death.

**Disposition:** No known correction within stated scope. Preserve the distinction between the control actually proved and stronger endpoint claims.

### F4 — Actual path components and singular H0 are used; their comparison remains open

`ClosedSuperlevel.lean:105–159` uses `ZerothHomotopy`, the actual quotient by `Joined`. The supplied lambda to `ZerothHomotopy.lift` takes a `Path`, matching the pinned lift signature, and applies `Path.map`. Surjectivity reduces identity/composition to point representatives. Equality versus `JoinedIn` at lines 149–159 uses the two directions of `joinedIn_iff_joined` correctly. No substitution of connected components for path components was found.

At lines 162–171, preorder morphisms provide `s ≤ t`, and `TopCat.ofHom` wraps the actual continuous inclusions. Pinned Preorder and TopCat sources confirm the direction and composition conventions. At lines 173–180 the assumptions `[Category.{v} C]`, `[HasCoproducts.{w} C]`, `[Preadditive C]` and `[CategoryWithHomology C]` match the pinned singular-homology functor. Its type is `C ⥤ TopCat.{w} ⥤ C`, so `(singularHomologyFunctor C 0).obj R` is the correct topological-space functor for coefficient object `R : C`. There is no apparent swapped argument, missing coefficient object or use of H0 as a placeholder name.

The identity/composition targets at lines 183–193 reference that actual H0 definition. They are inherited functor laws, not a comparison theorem identifying H0 generators with the displayed component classes. Such comparison/naturality is explicitly deferred.

**Disposition:** No known type/API defect from inspected signatures. Universe inference and exact bundled-category elaboration remain uncompiled risk R4. No H0/component comparison, finite-dimensionality or barcode conclusion is accepted here.

## Known defects versus unresolved risks

**Known mathematical counterexample to a candidate theorem:** none identified. **Definite Lean/API error established by direct primary-source mismatch:** none identified. **Required source correction supported by this review:** none identified. These are bounded review findings, not guarantees that compilation succeeds.

| Risk | Exact location | What remains uncertain | Required resolving evidence |
|---|---|---|---|
| R1 | Torus lines 43–58 | The combined `simp` call through units, orthonormal basis coercion, integer scalar action and PiLp coordinates has not elaborated. Each key named API was found at the pin. | Source-bound hosted elaboration of this exact term or a reviewed repair. |
| R2 | Torus lines 60–151 | Quotient/setoid reducibility, integer-span instance inference and compact-range specialization have not elaborated together. | Hosted elaboration and applicable axiom checks. |
| R3 | Torus lines 154–212 | Path/ContinuousMap coercions, constructor fields and the forward predicate-transfer term rely on definitional reduction. Source signatures align but no elaborator checked the combined terms. | Hosted elaboration and applicable checks of exact source. |
| R4 | ClosedSuperlevel lines 149–193 | Subtype proof irrelevance, TopCat extensionality, preorder morphisms, and coefficient-category universe inference have not been checked by Lean. | Hosted elaboration and applicable axiom checks. |

No risk above is reported as an observed Lean failure. No model agreement, text scan, theorem count, control statement or successful hash read establishes theorem correctness or completion. The first actual hosted error must be retained as observed evidence rather than backfilled into this review as though already known.

## Primary-source retrieval actually performed

Every remote source retrieval used the exact Mathlib commit `d13f23b723b8a846827a245b89c10fc7d3f11612` through raw GitHub text. Sources were streamed to bounded text reads; no repository code or dependency was executed. The pinned [`lean-toolchain`](https://github.com/leanprover-community/mathlib4/blob/d13f23b723b8a846827a245b89c10fc7d3f11612/lean-toolchain) was read and contains `leanprover/lean4:v4.34.1`. Mathlib bytes were not independently Git-blob hashed, and no full dependency audit is claimed.

| Path below Mathlib/ | Direct inspected lines / purpose |
|---|---|
| `AlgebraicTopology/SingularHomology/Basic.lean` | 15–75; coefficient category, degree and functor direction |
| `Topology/Homotopy/Lifting.lean` | 192–217, 230–300; initial equality, path lifting, actual constructor pattern |
| `Topology/Connected/PathConnected.lean` | 110–220, 600–650; quotient APIs, JoinedIn correspondence and quotient path connectedness |
| `GroupTheory/Coset/Defs.lean` | 148–235; quotient equality, surjectivity and periodic projection |
| `Algebra/Module/ZLattice/Basic.lean` | 285–345, 395–445, 775–803; discrete/full-span instances and compact range |
| `Topology/Covering/Quotient.lean` | 275–322; additive quotient covering theorem generated by `to_additive` |
| `Topology/Covering/Basic.lean` | 345–390; uniqueness of lifts on preconnected domains |
| `LinearAlgebra/Basis/SMul.lean` | 90–135; scaled basis and its evaluation |
| `LinearAlgebra/Finsupp/LinearCombination.lean` | 365–390; finite coordinate characterization of span |
| `Analysis/InnerProductSpace/PiL2.lean` | 795–835; bounded search context at 480–499 and 637–646 for `coe_toBasis` |
| `Analysis/Normed/Lp/PiLp.lean` | 105–140, 160–185; scalar and single-coordinate evaluation |
| `Topology/Constructions.lean` | 730–750, 790–815; quotient continuity, extra bounded context |
| `Topology/Category/TopCat/Basic.lean` | 20–175; bundled morphisms, identity, composition and extensionality |
| `CategoryTheory/Category/Preorder.lean` | 20–115; morphism direction and `leOfHom` |
| `Topology/ContinuousMap/Basic.lean` | 408–435; continuous subspace inclusions |

Each primary source above is available at `https://github.com/leanprover-community/mathlib4/blob/d13f23b723b8a846827a245b89c10fc7d3f11612/Mathlib/` followed by its listed path. No default-branch API was substituted for the pin.

Inert checks: byte and line counts; SHA-256 of both candidate files and cached Field; namespace/theorem/prohibited-token text search. The inspected source has one `ResearchFormalCoreR1` namespace per module, the expected `torus_`/`superlevel_` theorem names, and no matched `sorry`, `admit`, `axiom`, `unsafe`, or `native_decide` token. This search is not a transitive axiom audit. No Python, Node, Lean, Lake, tests, project imports, external mutation tools or additional agents were run. This review file is the only reviewer write.

## Open scientific scope and handoff

Smooth quotient charts and smooth descent; derivative/Hessian invariance and intrinsic Morse-index compatibility; critical-free band invariance; the critical-level Morse local/handle theorem; merging versus attachment within one component; endpoint attainment; finite critical events and labels; H0/component-generator naturality; tameness; interval decomposition; elder correspondence; once-counting; and the finite-bar theorem all remain **OPEN**. The next substantive research dependency is the critical-level Morse/endpoint argument, with its smooth quotient compatibility prerequisites and explicit separation of merging from nonmerging attachments.

Compactness and closedness for a general continuous function do not imply finitely many path components. A point lying in its own closed level does not establish a new H0 birth. Transferring a closed path predicate does not establish its truth at the death height. Global maxima have no strictly higher endpoint. None of these boundaries is removed by this review.

The parent may retain this as completed scoped nonauthor source review of the two exact hashes, while preserving **compilation NOT_RUN / scientific NONE / full theorem OPEN**. Any source change invalidates the exact-hash binding and requires review of the changed proof and affected dependencies. Hosted execution, registration, integration, publication and custody are outside this reviewer task and are not attested here.
