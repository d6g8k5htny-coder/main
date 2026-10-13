# Scope note: C254 torus quotient and closed-superlevel filtration companion

Dylan Roy — delegated AI work. Source author: OpenAI/Codex lane (C254). Compile fixes and
companion: Anthropic/Claude lane. Scientific effect NONE; organizational-independence credit 0;
no human review; alignment `PENDING_INDEPENDENT_REVIEW`.

This note fixes what the 41 theorem statements and 17 audited definitions of
`TorusTopology.lean` and `ClosedSuperlevel.lean` cover and what they do not. An independent
reviewer must compare each Lean statement with this note and with the handoff's own scope
(TORUS_FILTRATION.md in comment 6111059166) before any alignment record is written. The
compile fixes changed proof terms and resolved names at the pin; whether every statement still
says what the handoff intended is part of that review, not something this note decides.

## Bound sources

- Originals (NOT_COMPILED, namespace `ResearchFormalCoreR1`): `TorusTopology.lean` SHA-256
  `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4`; `ClosedSuperlevel.lean`
  SHA-256 `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e`.
- Claim `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`; Math- PR #419 comments 6111056103, 6111059166,
  6111059498 (nonauthor source review: COMPILATION HOLD), 6111059848, 6111255007 (route finding);
  this lane: 6111093545 (local elaboration report), 6111139060 (pickup).
- Namespace mapping `ResearchFormalCoreR1.<n>` (handoff) -> `TorusFiltration.<n>` (companion).
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
| `superlevel_h0Filtration_map_id`, `superlevel_h0Filtration_map_comp` | functor laws of `superlevel_filtration g ⋙ (singularHomologyFunctor C 0).obj R` with the coefficient category's coproduct, preadditive and homology structure stated explicitly | naturality comparing component maps with singular-H0 generators; any identification of `PairLawInterface.p` |

The audited definitions (`torus_E`, `torus_quotient`, `torus_latticeVec`, `torus_periodBasis`,
`torus_periodLattice`, `torus_proj`, `torus_isPeriodic`, `torus_descend`, the three lattice
instances, `superlevel_set`, `superlevel_inclusion`, `superlevel_componentMap`,
`superlevel_componentInclusion`, `superlevel_filtration`, `superlevel_h0Filtration`) are
constructions, not claims; they are audited so that no hidden axiom enters through a definition.

## Controls in the replay

| Control | Detects |
|---|---|
| `RejectSuperlevelSignFlip` | the parameter convention: `() ∈ F(-1)` for the constant `0` would hold if the definition read `t ≤ g x` instead of `-t ≤ g x`; Lean leaves `⊢ False` |
| `RejectSuperlevelBelowThreshold` | a point below the threshold admitted to the closed level (`3 ∈ {x | 4 ≤ x}`); `⊢ False` |
| `RejectConstantNotPeriodic` | the negation of a periodicity instance that `Positive.lean` uses (`fun _ _ => rfl`); `⊢ False` |
| `RejectSorry`, `RejectCustomAxiom`, `RejectNativeDecide` | an admitted proof, a hidden premise behind an intermediate theorem, and a native-code decision each pass Lean (exit 0) and must be refused by the transitive axiom audit (`sorryAx`, `hiddenPremise`, `injected._native.native_decide.ax_1_1`) |

Each control is matched by `re.fullmatch` against its pinned diagnostic; the false-statement
diagnostics and the three injection reports were pinned from actual Lean 4.34.1 output at the
Mathlib `d13f23b7` pin. A crash, an extra message or a different exit code fails the control.
The Python suite additionally refuses every weakening of `check.py` it knows about (wrong
diagnostic, wrong exit code, unknown control, `sorryAx`, unknown or missing audit target,
unrecognized audit text, missing or extra type, a diagnostic after the type listing, membership
or byte drift, a committed `.olean`, a symlinked source, stale manifest, duplicate JSON key,
renamed theorem, hidden or indented declaration, injected admission, every honesty label and
schema field of `SOURCE_FILES.json` one at a time, wrong or unpinned toolchain and dependency
lock, host identity mismatch, an emitted Lean file altered between `emit` and `finish`, an
unrecognized `leanchecker` invocation, leanchecker output in modules mode, wrong installed
dependency).

## Evidence boundary

A green hosted run establishes, for these 58 declarations at this toolchain and dependency
lock, that Lean elaborated them with warnings as errors, that
`leanchecker TorusTopology ClosedSuperlevel TorusFiltration` replayed the companion modules' own
declarations through the kernel against the cached, prebuilt dependency `.olean` files (the
import closure, all of Mathlib, is trusted from that cache and not re-checked; a `--fresh`
re-check of the closure does not fit a hosted job and is available only as an operator's local
run, which the receipt records as mode `fresh`), and that every transitive axiom lies in
`{propext, Classical.choice, Quot.sound}`. It establishes nothing about alignment with the
research programme, nothing about C253 or any q0 claim, and nothing the handoff's own list of exclusions names: critical-level
attainment, finite critical events, label tracking, naturality to singular-H0 generators beyond
the functor laws, interval decomposition, bar bijection, elder events, once-counting, or any
identification of `PairLawInterface.p`. The receipt's `formalization_status: kernel-checked` is a
record of that computation; `alignment_status` remains `PENDING_INDEPENDENT_REVIEW` and
`scientific_acceptance` remains `false` until an independent reviewer with nonzero
organizational-independence credit records otherwise. Aging approves nothing.

## Reproduction

    bash companions/torus_filtration_20261011/replay.sh
    python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py
    python3 -B -O -S companions/torus_filtration_20261011/test_torus_filtration.py

The replay must start without `.lake/torus-filtration-evidence` present and must not run
`formal/gate.py`, the formal tests, `lake build`, `lake update` or `lake exe cache`.
