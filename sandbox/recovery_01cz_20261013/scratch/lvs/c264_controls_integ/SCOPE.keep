# Scope note: C254 torus quotient and closed-superlevel filtration companion, with the C263 strict-cut and C264 ordinary-H0 modules

Dylan Roy — delegated AI work. Source authors: OpenAI/Codex lanes (C254: `TorusTopology`,
`ClosedSuperlevel`; C263: `SuperlevelBar`; C264: `OrdinaryH0`). Compile fixes, the C264 owner
repair, the owner-authored controls module `OrdinaryH0Controls` (C264 owner task item 4) and the
companion: Anthropic/Claude lane. Scientific effect NONE;
organizational-independence credit 0; no human review; alignment `PENDING_INDEPENDENT_REVIEW`.

This note fixes what the 118 theorem statements and 35 audited definitions of the five modules
(`TorusTopology.lean`, `ClosedSuperlevel.lean`, `SuperlevelBar.lean`, `OrdinaryH0.lean`,
`OrdinaryH0Controls.lean`; 153 declarations) cover and what they do not. An independent reviewer must compare each Lean
statement with this note and with the handoffs' own scope (TORUS_FILTRATION.md in comment
6111059166 for C254; the module docstrings and Math- #428 comments 6113033545 and 6113372506 for
C263 and C264) before any alignment record is written. The compile fixes and the C264 repair
changed proof terms, resolved names and three declaration spellings at the pin; whether every
statement still says what the handoffs intended is part of that review, not something this note
decides. The C263 and C264 modules are registered under the handoffs' own FORMAL HOLD, which
rests on the handoffs' own listed conditions (Math- #428 comments 6113033545 item 4, 6113372506
and 6113277485 item 5), among them a nonauthor exact-source review of the compiled successor
bytes, a distinct review of the source-bound execution output and source-bound hosted evidence and
custody; the handoffs say a source review alone does not meet them. Registration is not review. At
the registration commit `940dffc9` (2026-10-11) no nonauthor exact-source review of the successor
bytes existed. On 2026-10-12 the nonauthor engineering registration readback of
`4c33d8cd..940dffc9`, Math- #428 review 5485829137 (01:16Z, PASS, identity and registration only),
was posted and records organizational-independence credit 0; since 01:32Z Math- #428 review
5485883117 (Muse Spark / Meta, first read, PASS source-scoped for each module; count corrected in
comment 6116328276) is an exact-source review, source-scoped only: its reviewer did not run Lean and
keeps Lean execution as an open condition under FORMAL HOLD, and it states no
organizational-independence credit and was posted from the same project account. FORMAL HOLD is
unchanged; its disposition is for the C263/C264 handoff owners, alignment still needs a reviewer
with nonzero organizational-independence credit, and this note moves no label. `OrdinaryH0Controls`
applies seven of the 20 C264 theorems (and three of its definitions) to concrete fixtures; it is not
a handoff module, not a review of `OrdinaryH0`, and lifts nothing.

## Bound sources

- Originals (NOT_COMPILED, namespace `ResearchFormalCoreR1`): `TorusTopology.lean` SHA-256
  `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4`; `ClosedSuperlevel.lean`
  SHA-256 `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e`.
- Claim `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`; Math- PR #419 comments 6111056103, 6111059166,
  6111059498 (nonauthor source review: COMPILATION HOLD), 6111059848, 6111255007 (route finding);
  this lane: 6111093545 (local elaboration report), 6111139060 (pickup).
- Namespace mapping `ResearchFormalCoreR1.<n>` (handoff) -> `TorusFiltration.<n>` (companion);
  `SuperlevelBar` and `OrdinaryH0` were authored directly in namespace `TorusFiltration`, and so
  was the owner-authored `OrdinaryH0Controls`.
- C263 handoff `SuperlevelBar.lean` SHA-256
  `d74dc8b2c25eff5b90ce7133e51a0888c7885cae24ca6d5e648bc514089637a8` (14904 bytes; Math- #428
  comments 6113033147, 6113033351, 6113033545; allocation #419 comment 6112439759), failed at the
  pin with warnings as errors (deprecated `if_pos`, line 123; this lane's report 428/6113229784).
  Registered successor SHA-256 `346eea948e18d5904422904006827930e27d3141ccbc19ee8ac2c24bcf65a4a9`
  (15309 bytes): `if_pos hne` -> `ite_eq_left hne` in `superlevelBar_death_eq_sSup_of_nonempty`
  plus docstring lines; imports `Mathlib` only. Census at the pin 25 = 24 + 0 + 1.
- C264 handoff `OrdinaryH0.lean` SHA-256
  `261dd0526d42890d8a4ac1225fcf47228eb3c9799af4106cb643a528bcbf968e` (18469 bytes; Math- #428
  comments 6113276830, 6113277173, 6113277485), failed at the pin (this lane's record
  428/6113300903); owner repair handed to this lane in 428/6113372506 and posted with its full
  diff in 428/6113438262. Registered successor SHA-256
  `3d6b0d6b53413c484b1d010bccf0d53ae7f75becb2d747868f22dbd673fe0a88` (19803 bytes): `variable`
  scope split, explicit `rw`/`erw` naturality proof, `Functor.isoWhiskerLeft`, three ASCII `iota`
  names, docstring lines, no statement changed; imports `Mathlib` and `ClosedSuperlevel`. Census
  at the pin 38 = 27 + 0 + 11.
- Owner-authored `OrdinaryH0Controls.lean` (C264 owner task item 4, Math- #428 comment
  6113277485, against the corrected CONTROL_OBLIGATIONS of comment 6113277173; no handoff bytes,
  not a successor): SHA-256 `df9d3645282ca526b67eca3d5db691b622092a14b5fddee9be21013e39666184`
  (29590 bytes; first compiled form `e825837f…`, 28503 bytes, differing only in the
  module-docstring paragraph on review status); imports `Mathlib` and `OrdinaryH0`; 37 theorems,
  3 abbrevs, 4 defs, prefix `ordinaryH0Control_`. Census at the pin 48 = 44 + 0 + 4.
- Pinned environment read from `formal/`: `leanprover/lean4:v4.34.1`, Mathlib
  `d13f23b723b8a846827a245b89c10fc7d3f11612`; nothing from the core is imported or pinned.

## Targets, exact coverage, not established

| Targets | Exact coverage | Not established |
|---|---|---|
| `torus_field_latticeVec_formula`, `torus_basis_sum_eq_latticeVec`, `torus_lattice_mem_iff` | `torus_latticeVec d L n = ∑ i, EuclideanSpace.single i (L * n i)` by `rfl`; the integer combination of the `L`-scaled standard basis is that vector; lattice membership iff an integer vector produces the point | any elaborated theorem naming the externally owned `Specifications.Field` module; the companion carries the literal formula and does not import that file |
| `torus_proj_surjective`, `torus_proj_continuous`, `torus_proj_eq_iff`, `torus_control_lattice_translate` | the quotient projection is surjective and continuous; `proj x = proj y ↔ ∃ n, y = x + torus_latticeVec d L n`; a lattice translate has the same image | a fundamental-domain parametrisation; uniqueness of representatives |
| `torus_proj_isCoveringMap`, `torus_compact`, `torus_pathConnectedSpace` (with the three discreteness/`IsZLattice` instances) | the projection is a covering map; `IsCompact Set.univ` on the quotient; the quotient is path connected | a registered `CompactSpace` instance; smooth charts; any manifold structure |
| `torus_field_periodic_formula`, `torus_periodic_respects`, `torus_descend_proj`, `torus_descend_continuous` | `torus_isPeriodic d L g` is literally the Field periodicity formula; periodic `g` respects the quotient relation; `descend g (proj x) = g x`; the descent is continuous when `g` is | smoothness of descended functions; derivative or Hessian invariance; Morse-index compatibility |
| `torus_path_lift_from`, `torus_control_constant_path_endpoint`, `torus_control_no_arbitrary_endpoint` | a quotient path from `proj M` lifts to a path from the prescribed representative `M` whose endpoint is **existential** and projects to the path's endpoint; a lift of the constant path from `M` ends at `M`; no lift of the constant path ends at a different lattice translate | a lift ending at an arbitrarily chosen representative (the controls refute exactly this) |
| `torus_higher_path_iff`, `torus_strict_higher_path_iff`, `torus_closed_higher_path_iff`, `torus_global_max_iff` | existence of a strictly higher endpoint reached by a path whose heights satisfy `P` (resp. `h < ·`, `h ≤ ·`) transfers between `E` and the quotient for the descended function; global maximality transfers | critical-level attainment at `h = δ`; conversion of strict path predicates into closed critical-level merger; finite critical events |
| `superlevel_control_closed_endpoint`, `superlevel_control_strict_endpoint`, `superlevel_control_parameter_sign`, `superlevel_control_inclusion_direction`, `superlevel_control_essential`, `superlevel_control_strict_thresholds` | the handoff's own controls: closed endpoint membership vs strict failure; parameter sign `t = -h`; inclusion direction; no strictly higher endpoint at a global maximum; density of strict witnesses below `0` with no witness at `0` | a compact Morse counterexample; a supremum theorem; endpoint nonattainment for the research problem; a finite death from an essential maximum |
| `superlevel_mono`, `superlevel_isClosed`, `superlevel_inclusion_id`, `superlevel_inclusion_comp`, `superlevel_mem_at_own_height`, `superlevel_neg_height` | `s ≤ t → F(s) ⊆ F(t)`; `F(t)` closed for continuous `g`; the continuous inclusions are a functor; `x ∈ F(-g x)`; `F(-h) = {x | h ≤ g x}` | component birth at the own height; finiteness of components |
| `superlevel_componentMap_mk`, `superlevel_componentMap_id`, `superlevel_componentMap_comp`, `superlevel_componentInclusion_id`, `superlevel_componentInclusion_comp`, `superlevel_component_eq_iff_joinedIn` | the induced maps on actual `ZerothHomotopy` quotients are functorial; equal classes iff `JoinedIn` the closed superlevel | label tracking, elder events, once-counting, interval decomposition, bar bijection |
| `superlevel_h0Filtration_map_id`, `superlevel_h0Filtration_map_comp` | functor laws of `superlevel_filtration g ⋙ (singularHomologyFunctor C 0).obj R` with the coefficient category's coproduct, preadditive and homology structure stated explicitly | naturality comparing component maps with singular-H0 generators (supplied, for Mathlib's actual H0, by `OrdinaryH0` below); any identification of `PairLawInterface.p` |
| `superlevelBar_essential_iff`, `superlevelBar_mem_lt_start`, `superlevelBar_clearance_bddAbove`, `superlevelBar_clearance_empty_of_essential` (C263) | on `EuclideanSpace ℝ (Fin d)`: essentiality is `∀ y, g y ≤ g M`; every member of the total strict-clearance set `{h | ∃ y, g M < g y ∧ ∃ γ : Path M y, ∀ t, h < g (γ t)}` is `< g M`; the set is bounded above by `g M` even when empty; an essential `M` has empty clearance | a positive-dimensional or periodic quotient-space hypothesis (the C263 ambient type is Euclidean space, not the torus of `TorusTopology`); a converse for discontinuous heights |
| `superlevelBar_death_eq_zero_of_empty`, `superlevelBar_death_eq_zero_of_essential`, `superlevelBar_death_eq_sSup_of_nonempty` | `superlevelBar_deathHeight` is `sSup` of the clearance set when nonempty and the declared default `0` when empty, hence `0` for essential inputs | a geometric death event or finite persistence interval at that `0`; a finite death for an essential class; the handoff's own words: the scalar zero on an empty set is a default, not an essential-bar death |
| `superlevelBar_path_minimum`, `superlevelBar_clearance_nonempty`, `superlevelBar_clearance_nonempty_iff`, `superlevelBar_clearance_downward`, `superlevelBar_clearance_improve` | a continuous height attains a minimum on each path; for continuous `g` the clearance set is nonempty exactly off the essential locus (`Path.segment`, minimum minus one); clearance is downward closed; each strict witness has a strictly larger witness on its own path | a minimum over all paths or an optimal path; compactness of the whole Euclidean domain; a witness strictly above the supremum |
| `superlevelBar_strict_cut`, `superlevelBar_death_le_start_of_nonempty`, `superlevelBar_death_le_start`, `superlevelBar_clearance_eq_Iio`, `superlevelBar_death_not_mem_clearance`, `superlevelBar_cut_unique` | for continuous nonessential inputs: `h < death ↔ h ∈ clearance`; death `≤ g M`; the clearance set is `Set.Iio death`; death is not a clearance; a real with the same strict cut is the death height | closed-threshold attainment at the death height; the death/start bound on the empty default branch; equality of arbitrary total interface records off the valid locus |
| `superlevelBar_death_const`, `superlevelBar_control_negative_essential` | a constant height has death `0` everywhere; the handoff's own control: the constant `-1` has death `0` and `¬ 0 ≤ -1`, so no unconditional death/start bound holds on the essential default | any finite ordinary bar for a constant height; an inhabitant of the separately owned Field or Lifetime record; an elder relation; Morse endpoint attainment; ordinary H0 bar pairing; finite-bar counts; a Gaussian field or weighted law |
| `ordinaryH0_componentCoproductMap_iota`, `ordinaryH0_componentFiltration_map_iota` (C264) | the component functor `TopCat ⥤ Type` (objects `ZerothHomotopy X`, maps `superlevel_componentMap`) followed by `sigmaConst.obj R` sends the generator at `q` to the generator at `superlevel_componentMap f.hom q`; the same along a closed-superlevel inclusion `homOfLE hst` through `superlevel_componentInclusion` | that distinct source components stay distinct; a reverse inclusion or endpoint deletion; regularity, finite dimensionality or Morse hypotheses on the height |
| `ordinaryH0_simplicialPointClass_iso`, `ordinaryH0_simplicialPointClass_map`, `ordinaryH0_toSSet_pointMap`, `ordinaryH0_pointClass_map`, `ordinaryH0_pointClass_iso`, `ordinaryH0_componentIota_isoInv`, `ordinaryH0_hom_ext` | in Mathlib's actual singular homology: the point class of a zero-simplex composed with `S.homology₀Iso R` is the generator at `SSet.π₀.mk x`; the actual homology map transports point classes; `TopCat.singularHomology₀Iso` sends `ordinaryH0_pointClass X R x` to the generator at `ZerothHomotopy.mk x` (and inversely); morphisms out of singular H0 agree if they agree on all point classes | a replacement or free-module definition standing in for singular H0; a transported or postulated substitute for the actual homology map; injectivity of the induced map; a distinguished geometric representative; a basis of geometric maxima or finiteness of components |
| `ordinaryH0_iso_naturality` (with `ordinaryH0_naturalIso`) | Mathlib's objectwise comparison `TopCat.singularHomology₀Iso` commutes with every continuous map, proved from actual zero-cycle generators (the repaired proof: explicit reassociation of both sides), so it assembles into a natural isomorphism `(singularHomologyFunctor C 0).obj R ≅ ordinaryH0_componentCoproductFunctor R` | naturality by an added assumption or compatibility record (none is used); an interval decomposition or persistence-bar classification |
| `ordinaryH0_pointClass_augmentation`, `ordinaryH0_pointClass_ne_zero`, `ordinaryH0_pointClass_eq_of_component_eq`, `ordinaryH0_sameComponent_difference_zero`, `ordinaryH0_pointClass_ne_of_component_ne`, `ordinaryH0_pointClass_eq_iff`, `ordinaryH0_difference_map`, `ordinaryH0_difference_map_eq_zero_iff` | the ordinary point class composed with `singularHomology₀ε` is `𝟙 R`; with `𝟙 R ≠ 0` it is nonzero and two point classes are equal exactly when the path components are; same-component differences are `0` for any coefficients; the induced map of a difference is the difference of image classes and, with `𝟙 R ≠ 0`, vanishes exactly when the image points share a component | any nonzero or distinction claim for a zero coefficient object; reduced-H0 vanishing (the ordinary point class is retained); equality of connected and path-connected components; death of the individual point classes; a Morse merger event, elder selection, death height or barcode |
| `ordinaryH0_filtration_naturality`, `ordinaryH0_filtration_difference_zero_iff_joinedIn` (with `ordinaryH0_componentFiltration`, `ordinaryH0_filtrationIso`) | `superlevel_h0Filtration g R ≅ superlevel_filtration g ⋙ ordinaryH0_componentCoproductFunctor R` (`Functor.isoWhiskerLeft`); the H0/component square at every `s ≤ t`; with `𝟙 R ≠ 0`, the difference of two point classes of `superlevel_set g s` dies under the inclusion into `superlevel_set g t` exactly when the points are `JoinedIn` the target closed superlevel | strict-level conventions; attainment of a critical death threshold; an elder or barcode theorem; the later interval, elder, maximum/saddle, count, Field or stochastic bridges; any Morse chart, critical-level attainment theorem, elder correspondence, Field adapter, count identity or stochastic law |
| `ordinaryH0Control_intCoeff_id_ne_zero`, `ordinaryH0Control_mod2Coeff_id_ne_zero`, `ordinaryH0Control_mod2Coeff_id_add_self`, `ordinaryH0Control_zeroCoeff_isZero`, `ordinaryH0Control_zeroCoeff_id_eq_zero` (with the abbrevs `ordinaryH0Control_intCoeff`, `ordinaryH0Control_zeroCoeff`, `ordinaryH0Control_mod2Coeff`) (C264 item 4) | coefficient fixtures in `ModuleCat.{0} ℤ`: `𝟙 ℤ ≠ 0`, `𝟙 (ZMod 2) ≠ 0` and `𝟙 + 𝟙 = 0` on `ZMod 2`; the one-point module is a zero object with `𝟙 = 0` | any homology computation by itself |
| `ordinaryH0Control_empty_components`, `ordinaryH0Control_empty_componentCoproduct_isZero`, `ordinaryH0Control_empty_h0_isZero`, `ordinaryH0Control_zeroCoeff_componentCoproduct_isZero`, `ordinaryH0Control_zeroCoeff_h0_isZero` | the empty space has no path component, a zero component coproduct for every coefficient module and zero actual singular H0 with `ℤ` coefficients; with zero coefficients the component coproduct and actual singular H0 of every space in `TopCat.{0}` are zero objects (transported along `ordinaryH0_naturalIso`) | H0 of the empty space for other coefficients; vanishing with nonzero coefficients for a nonempty space |
| `ordinaryH0Control_point_augmentation`, `ordinaryH0Control_point_ne_zero`, `ordinaryH0Control_point_h0_not_isZero` | one-point space, `ℤ` coefficients: the point class augments to `𝟙`, is nonzero, and H0 is not a zero object | reduced-H0 vanishing (refuted here for the ordinary class); a computation of that H0 beyond its nonvanishing |
| `ordinaryH0Control_line_component_eq`, `ordinaryH0Control_line_difference_zero_int`, `ordinaryH0Control_line_difference_zero_zeroCoeff`, `ordinaryH0Control_line_classes_ne_zero` | `0` and `1` share a path component of `ℝ`; their point-class difference is `0` with `ℤ` and with zero coefficients; with `ℤ` each class is nonzero | the death of either class; path-connectedness of any closed superlevel |
| `ordinaryH0Control_discrete_component_ne`, `ordinaryH0Control_bool_component_ne`, `ordinaryH0Control_bool_pointClass_ne`, `ordinaryH0Control_bool_pointClass_ne_mod2` | distinct points of a discrete space lie in distinct path components; the two point classes of `Bool` differ with `ℤ` and with `ZMod 2` coefficients | path components of a non-discrete space; separation for every characteristic-two object or with zero coefficients |
| `ordinaryH0Control_collapse_difference_zero`, `ordinaryH0Control_collapse_image_ne_zero`, `ordinaryH0Control_collapse_difference_dies_classes_survive` (with `ordinaryH0Control_collapse`) | merger fixture: the collapse `Bool → Unit` kills the difference of the two point classes while each image class stays nonzero | a Morse merger event, elder selection or death height; the death of an individual class |
| `ordinaryH0Control_square_differences_ne_zero`, `ordinaryH0Control_fst_difference_zero`, `ordinaryH0Control_fst_difference_ne_zero` (with `ordinaryH0Control_fst`) | non-merger fixture: the non-injective first projection `Bool × Bool → Bool` kills the difference of `(true, false)` and `(true, true)` and keeps that of `(true, false)` and `(false, true)` nonzero; both differences are nonzero before the map | injectivity of the induced H0 map (which fails on the first pair) |
| `ordinaryH0Control_bool_zeroCoeff_eq`, `ordinaryH0Control_point_zeroCoeff_eq_zero`, `ordinaryH0Control_separation_needs_nonzero`, `ordinaryH0Control_nonvanishing_needs_nonzero` | with zero coefficients the two point classes of `Bool` are equal although the components differ and the point class of the point is `0`; so the separation and nonvanishing statements are false without `𝟙 R ≠ 0` | any failure of the C264 statements, which keep that hypothesis |
| `ordinaryH0Control_parabola_le`, `ordinaryH0Control_parabola_strict_growth`, `ordinaryH0Control_parabola_not_joined_quarter`, `ordinaryH0Control_parabola_joined`, `ordinaryH0Control_parabola_difference_zero`, `ordinaryH0Control_parabola_strict_difference_ne_zero` (with `ordinaryH0Control_parabolaLeft`, `ordinaryH0Control_parabolaRight`) | filtration fixture for `x ↦ x ^ 2`: `-1` and `1` in the closed superlevel at parameter `-1`; the strict inclusion into the level at `-1/4` keeps the difference of their classes nonzero (no path through `0` in `{x | 1/4 ≤ x ^ 2}`); the inclusion into the level at `0`, the whole line, kills it | a first merger level, threshold attainment, the behaviour strictly between `-1/4` and `0`; any Morse, interval, elder, count, Field or stochastic statement |

The audited definitions (`torus_E`, `torus_quotient`, `torus_latticeVec`, `torus_periodBasis`,
`torus_periodLattice`, `torus_proj`, `torus_isPeriodic`, `torus_descend`, the three lattice
instances, `superlevel_set`, `superlevel_inclusion`, `superlevel_componentMap`,
`superlevel_componentInclusion`, `superlevel_filtration`, `superlevel_h0Filtration`,
`superlevelBar_E`, `superlevelBar_isEssential`, `superlevelBar_clearanceSet`,
`superlevelBar_deathHeight`, `ordinaryH0_componentFunctor`, `ordinaryH0_componentCoproductFunctor`,
`ordinaryH0_componentFiltration`, `ordinaryH0_simplicialPointClass`, `ordinaryH0_pointClass`,
`ordinaryH0_naturalIso`, `ordinaryH0_filtrationIso`, `ordinaryH0Control_intCoeff`,
`ordinaryH0Control_zeroCoeff`, `ordinaryH0Control_mod2Coeff`, `ordinaryH0Control_collapse`,
`ordinaryH0Control_fst`, `ordinaryH0Control_parabolaLeft`, `ordinaryH0Control_parabolaRight`) are
constructions and fixtures, not claims; they are audited so that no hidden axiom enters through a
definition.

## Controls in the replay

| Control | Detects |
|---|---|
| `RejectSuperlevelSignFlip` | the parameter convention: `() ∈ F(-1)` for the constant `0` would hold if the definition read `t ≤ g x` instead of `-t ≤ g x`; Lean leaves `⊢ False` |
| `RejectSuperlevelBelowThreshold` | a point below the threshold admitted to the closed level (`3 ∈ {x | 4 ≤ x}`); `⊢ False` |
| `RejectConstantNotPeriodic` | the negation of a periodicity instance that `Positive.lean` uses (`fun _ _ => rfl`); `⊢ False` |
| `RejectEssentialDeathBound` (C263) | the death/start bound asserted on the essential default branch: `superlevelBar_deathHeight (fun _ : superlevelBar_E 1 => (-1 : ℝ)) 0 ≤ (-1 : ℝ)`; `superlevelBar_death_const` reduces the death to `0` and `0 ≤ -1` is false (the proposition of the C263 handoff's `FALSE_CONTROL_PROPOSAL` (Math- #428 comment 6113033351), whose tactic `rw [superlevelBar_death_const 1 (-1) (0 : superlevelBar_E 1)]; norm_num` is replaced here by `norm_num [superlevelBar_death_const]`); `⊢ False` |
| `RejectReducedPointClass` (C264) | a reduced-H0 substitution: the ordinary point class of the one-point space with `ℤ`-module coefficients asserted NOT to be augmented to the identity; `ordinaryH0_pointClass_augmentation` rewrites it to `𝟙 R ≠ 𝟙 R`; `⊢ False` |
| `RejectMergerKillsPointClass` (C264 item 4) | a merger confused with the death of a class: the collapse `Bool → Unit` asserted to kill the individual point class of `true`; `ordinaryH0Control_collapse_image_ne_zero` rewrites it to `False`; `⊢ False` |
| `RejectSeparationWithoutNonzeroCoeff` (C264 item 4) | a separation claim without its coefficient hypothesis: the two point classes of `Bool` asserted to differ with zero coefficients; `ordinaryH0Control_bool_zeroCoeff_eq` rewrites it to `False`; `⊢ False` |
| `RejectNonmergerKillsDifference` (C264 item 4) | a non-injective map confused with a merger: the projection `Bool × Bool → Bool` asserted to kill the difference of `(true, false)` and `(false, true)`, whose images stay in distinct components; `ordinaryH0Control_fst_difference_ne_zero` rewrites it to `False`; `⊢ False` |
| `RejectSorry`, `RejectCustomAxiom`, `RejectNativeDecide` | an admitted proof, a hidden premise behind an intermediate theorem, and a native-code decision each pass Lean (exit 0) and must be refused by the transitive axiom audit (`sorryAx`, `hiddenPremise`, `injected._native.native_decide.ax_1_1`) |

Each control is matched by `re.fullmatch` against its pinned diagnostic; the false-statement
diagnostics and the three injection reports were pinned from actual Lean 4.34.1 output at the
Mathlib `d13f23b7` pin. A crash, an extra message or a different exit code fails the control. For
the three item-4 controls the same tactic on each control's true sibling statement was run at the
pin and leaves a different diagnostic (`` `simp` made no progress ``, or a goal other than `False`
with an unused simp argument), which `negative` refuses; those sibling logs are pinned in the
protocol tests as negative controls, so each control's `⊢ False` is attributable to the falsehood
of its statement.
The Python suite additionally refuses every weakening of `check.py` it knows about (wrong
diagnostic, wrong exit code, unknown control, `sorryAx`, unknown or missing audit target,
unrecognized audit text, missing or extra type, a diagnostic after the type listing, membership
or byte drift, a committed `.olean`, a symlinked source, stale manifest, duplicate JSON key,
renamed theorem, hidden or indented declaration, injected admission, every honesty label and
schema field of `SOURCE_FILES.json` outside the successor record one at a time (the successor record's
coverage is stated at the end of this list), wrong or unpinned toolchain and dependency
lock, host identity mismatch, an emitted Lean file altered between `emit` and `finish`, an
unrecognized `leanchecker` invocation, leanchecker output in modules mode, wrong installed
dependency, a second `finish` on a directory already holding a receipt, an existing receipt
beside a newly failing input, any missing, extra or symlinked evidence member, an uninventoried
`alias` indented after the namespace opening, on a declaration line or at column 0, a keyword
spliced to a closed block comment or to a numeral, every refused command keyword indented after
the namespace opening, a declaration census (`Census.lean`) with an extra, missing or unlisted
constant, a user-written declaration with an internal-style name, a `census_auxiliary` entry
that Lean did not generate, a keyword spliced to a hex/binary/octal/exponent numeral, a wrong
census count or terminator, a `check.py` child running in a different
optimization mode from the outer test interpreter, which the fixture checks on every child, and,
since the C263/C264 registration, a companion import of the module itself, of a later module or
of the root, an `open scoped` line with a lowercase or missing namespace or anything else on the
line, the pre-registration two-import root module and three-module `leanchecker` command, a
module name outside `MODULES`, since the item-4 controls the pre-controls four-import root module and five-module
`leanchecker` command, an import of `OrdinaryH0Controls` by an earlier module, the three item-4 controls' true-sibling
diagnostics, single-field changes to the owner-module record of `OrdinaryH0Controls` (hash, byte count, author, kind,
review status, census, credit, a dropped comment pointer, an added field, a dropped or added entry, the record copied
into the successor record), and single-field changes to the successor record (a changed
handoff or successor hash, a successor byte count or a C26x entry of `provenance.original_bytes`, a changed census line, a nonzero credit, a lifted or
self-awarded `formal_status`, a dropped comment pointer, a dropped or added module entry); its remaining fields (handoff
byte count and status, lane, authors, namespace, imports, change summary) are refused through the whole-record
comparison in `metadata`, not by a control of their own).

## Evidence boundary

A green hosted run establishes, for these 153 declarations at this toolchain and dependency
lock, that Lean elaborated them with warnings as errors, that Lean's own census of the
constants the six modules (the five modules and the root) add to the environment
(`Census.lean`, with declaration-range and internal-name tags) has exactly these 153 as its
user-written declarations and exactly the one named compiler-generated auxiliary (internal
generated constants, 27 at the pin, are counted and otherwise ignored), that
`leanchecker TorusTopology ClosedSuperlevel SuperlevelBar OrdinaryH0 OrdinaryH0Controls TorusFiltration` replayed the
companion modules' own
declarations through the kernel against the cached, prebuilt dependency `.olean` files (the
import closure, all of Mathlib, is trusted from that cache and not re-checked; a `--fresh`
re-check of the closure does not fit a hosted job and is available only as an operator's local
run, which the receipt records as mode `fresh`), and that every transitive axiom lies in
`{propext, Classical.choice, Quot.sound}`. It establishes nothing about alignment with the
research programme, nothing about C253 or any q0 claim, and nothing the handoffs' own lists of exclusions name: critical-level
attainment, finite critical events, label tracking, naturality to singular-H0 generators beyond
the functor laws (C254; the C264 module proves that naturality for Mathlib's actual H0 and
nothing beyond it), interval decomposition, bar bijection, elder events, once-counting, any
identification of `PairLawInterface.p`, an inhabitant of the Field or Lifetime record, a Field
adapter, a Morse chart, a count identity, a Gaussian or stochastic law, closed-threshold
attainment at the C263 death height, reduced-H0 vanishing, or any claim for a zero coefficient
object; and, for `OrdinaryH0Controls`, nothing beyond its fixtures: no review of `OrdinaryH0`, no
first merger level or threshold attainment, no statement that its fixtures meet every row of the
CONTROL_OBLIGATIONS matrix or that item 4 of the C264 owner task is closed. The receipt's `formalization_status: kernel-checked` is a
record of that computation; `alignment_status` remains `PENDING_INDEPENDENT_REVIEW` and
`scientific_acceptance` remains `false` until an independent reviewer with nonzero
organizational-independence credit records otherwise. For `SuperlevelBar` and `OrdinaryH0` the
handoffs' own FORMAL HOLD stands in addition, on the handoffs' own listed conditions (6113033545
item 4; 6113372506; 6113277485 item 5), among them a nonauthor exact-source review of the compiled
successor bytes (346eea94…, 3d6b0d6b…), a distinct review of the source-bound execution output and
source-bound hosted evidence and custody, which they say a source review alone does not meet. At the
registration commit `940dffc9` no nonauthor review of those bytes existed. The engineering
registration readback 5485829137 (2026-10-12 01:16Z, identity and registration only, credit 0) and,
since 2026-10-12 01:32Z, the source-scoped review 5485883117 (its reviewer did not run Lean and keeps
Lean execution as an open condition under FORMAL HOLD; no organizational-independence credit stated,
posted from the same project account) exist; neither is a nonzero-credit review, and FORMAL HOLD is
unchanged, its disposition being for the C263/C264 handoff owners. The registration of 2026-10-11 is
not a review, and the compiler, repairer and registrar of both successors, and the author and
registrar of `OrdinaryH0Controls` (df9d3645…), is the companion's own lane at
organizational-independence credit 0. Aging approves nothing.

## C264 owner task item 4 controls (2026-10-12)

The owner-authored `OrdinaryH0Controls.lean` (task: Math- #428 comment 6113277485, item 4;
specification: the corrected CONTROL_OBLIGATIONS of comment 6113277173) applies seven of the 20
C264 theorems and three of its definitions to named spaces and coefficient modules with every premise
proved in the fixture (its thirteen premise lemmas name no C264 theorem): the
empty space, the point, one path component (`0`, `1` in `ℝ`), two components (`Bool`), a merger
(the collapse `Bool → Unit`: the difference dies, each class survives), a non-injective non-merger
(the projection of `Bool × Bool`: one difference dies, another survives), zero coefficients (all
of H0 is zero, so separation and nonvanishing need `𝟙 R ≠ 0`), characteristic two (`ZMod 2`
separates), and a closed-superlevel filtration of `x ↦ x ^ 2` (a strict inclusion keeps a
difference, the inclusion into the whole line kills it). It is registered with three false
controls (`RejectMergerKillsPointClass`, `RejectSeparationWithoutNonzeroCoeff`,
`RejectNonmergerKillsDifference`, table above), each refused with exactly `⊢ False` and each
discriminated from its true sibling statement. The evidence directory grows from 54 to 65 members
(one build pair, three control triples); the census from 133 to 181 constants (153 user range, 1
user generated, 27 internal generated). It does not establish: a review of `OrdinaryH0` or
`SuperlevelBar` (FORMAL HOLD unchanged for both); alignment (`PENDING_INDEPENDENT_REVIEW`); that
item 4 is closed (no label moves); any Morse chart, critical-level attainment, interval
decomposition, elder correspondence, count identity, Field adapter or stochastic law; any statement
about connected (as opposed to path) components; a fixture for the 13 C264 theorems it does not
name; any organizational-independence credit (0).

## Reproduction

    bash companions/torus_filtration_20261011/replay.sh
    python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py
    python3 -B -O -S companions/torus_filtration_20261011/test_torus_filtration.py

The replay must start without `.lake/torus-filtration-evidence` present and must not run
`formal/gate.py`, the formal tests, `lake build`, `lake update` or `lake exe cache`.
