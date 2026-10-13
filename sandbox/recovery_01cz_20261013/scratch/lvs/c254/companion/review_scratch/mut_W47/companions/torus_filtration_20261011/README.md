# C254 torus quotient and closed-superlevel filtration (isolated Lean companion)

Dylan Roy — delegated AI work. Source author: OpenAI/Codex lane (C254). Compile fixes and
companion: Anthropic/Claude lane. Scientific effect **NONE**. Organizational-independence
credit **0** for both lanes; no human review. Alignment status `PENDING_INDEPENDENT_REVIEW`.
Nothing here promotes, discharges or reclassifies any claim.

## Purpose and route rationale

The C254 handoff (claim `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64`, Math- PR #419) proposed two
Lean modules, `TorusTopology.lean` and `ClosedSuperlevel.lean`, as **NOT_COMPILED** source under
namespace `ResearchFormalCoreR1`, intended for `formal/`. The handoff's own nonauthor source
review (comment 6111059498) ended in **COMPILATION HOLD**: no known required correction, but no
elaboration either. The local elaboration report (6111093545) then observed the first real
elaboration failures at the pin (`Basis` is `Module.Basis` at Mathlib `d13f23b7`, among others).

Placing the modules in `formal/` would change the whole `formal/` tree. The three D2 companions
(`companions/d2_*_20261006`) pin that tree at `38b6d1a7746075b560349d184a461933745c11ce`, the
primary manifest at SHA-256 `40cb6ffc…` and the gate at `278989bb…`; any change there is a new
review scope for all three. The C255 route finding (6111255007) therefore recommended an
isolated companion. This directory is that companion:

- it holds the compile-fixed modules under namespace `TorusFiltration` and a two-line root module;
- it compiles them against the **same** pinned toolchain and dependency lock as `formal/`
  (`formal/lean-toolchain`, `formal/lakefile.toml`, `formal/lake-manifest.json`), which it reads
  but never edits;
- it consumes nothing from the core: no import of `ResearchFormalCoreR1`, no pin on the core
  tree, manifest or gate, and its replay runs neither `formal/gate.py` nor the formal tests.

## Bound sources

| Item | Identity |
|---|---|
| Claim | `8a5c7a24-784b-4aaa-b951-cd9f37b0dc64` (R17 Work Events 1559) |
| Handoff | Math- PR #419, comments 6111056103 (modules), 6111059166 (overlay), 6111059498 (nonauthor source review, COMPILATION HOLD), 6111059848 (implementer report), 6111255007 (C255 route finding: D2 pins, companion route) |
| This lane | local elaboration report 6111093545; pickup 6111139060 |
| Original `TorusTopology.lean` (NOT_COMPILED) | SHA-256 `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4`, 11542 bytes |
| Original `ClosedSuperlevel.lean` (NOT_COMPILED) | SHA-256 `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e`, 7827 bytes |
| Namespace mapping | `ResearchFormalCoreR1.<n>` (handoff) -> `TorusFiltration.<n>` (companion); no other name changes |
| Toolchain | `leanprover/lean4:v4.34.1`, release commit `5045d0056413266e57c625dcd7c365b10e377c52`; Mathlib `d13f23b723b8a846827a245b89c10fc7d3f11612` via `formal/lake-manifest.json` |

`SOURCE_FILES.json` records bytes, SHA-256 and Git blob of every tracked companion file and the
workflow, the per-module declaration inventory and this provenance. `check.py source` refuses to
run unless the tracked tree, the working bytes and that manifest agree.

## What the 41 theorems and 17 definitions establish

Topology and scaffolding only: the period-lattice quotient, descent of periodic functions, path
lifting with an **existential** endpoint, closed superlevel inclusions and their path-component
and singular-H0 maps. The source labels these `proved`; the hosted receipt may call them
`kernel-checked`; neither label is alignment or acceptance.

### `TorusTopology` (21 theorems; 2 abbrevs, 6 defs, 3 instances)

| Theorem | What it states |
|---|---|
| `torus_field_latticeVec_formula` | `torus_latticeVec d L n` is literally `∑ i, EuclideanSpace.single i (L * n i)` (by `rfl`) |
| `torus_basis_sum_eq_latticeVec` | the integer combination of the `L`-scaled standard basis equals that lattice vector |
| `torus_lattice_mem_iff` | lattice membership iff some integer vector produces the point |
| `torus_proj_surjective`, `torus_proj_continuous` | the quotient projection is surjective and continuous |
| `torus_proj_eq_iff` | equal images iff `y = x + torus_latticeVec d L n` (orientation fixed) |
| `torus_control_lattice_translate` | control: a lattice translate has the same image |
| `torus_proj_isCoveringMap` | the projection is a covering map |
| `torus_compact` | `IsCompact (Set.univ : Set (torus_quotient d L hL))`; not a registered `CompactSpace` instance |
| `torus_pathConnectedSpace` | the quotient is path connected |
| `torus_field_periodic_formula` | the periodicity predicate is literally the Field formula (`Iff.rfl`) |
| `torus_periodic_respects` | a periodic function respects the quotient relation |
| `torus_descend_proj` | descent composed with projection is the function (`rfl`) |
| `torus_descend_continuous` | the descended function is continuous when `g` is |
| `torus_path_lift_from` | a quotient path from `proj M` lifts to a path from `M`; the endpoint `z` is existential, with `proj z` the path's endpoint |
| `torus_control_constant_path_endpoint` | control: a lift of a constant projected path from `M` ends at `M` |
| `torus_control_no_arbitrary_endpoint` | control: a nonzero lattice translate shares the image yet no lift of the constant path ends there |
| `torus_higher_path_iff` | a strictly higher endpoint reached by a path whose heights satisfy a predicate `P` exists in `E` iff it exists in the quotient for the descended function |
| `torus_strict_higher_path_iff`, `torus_closed_higher_path_iff` | the same with `h < g` and `h ≤ g` |
| `torus_global_max_iff` | `M` is a global maximum of `g` iff `proj M` is one of the descended function |

### `ClosedSuperlevel` (20 theorems; 6 defs)

`superlevel_set g t = {x | -t ≤ g x}`, parameter `t = -h`, endpoint membership with `≤`.

| Theorem | What it states |
|---|---|
| `superlevel_control_closed_endpoint`, `superlevel_control_strict_endpoint` | the threshold point belongs to the closed level; the strict inequality fails there |
| `superlevel_control_parameter_sign`, `superlevel_control_inclusion_direction` | height `0` lies in `F(1)`; `F(1) ⊄ F(-1)` for the constant `0` |
| `superlevel_control_essential` | a global maximum has no strictly higher endpoint |
| `superlevel_mono` | `s ≤ t → F(s) ⊆ F(t)` |
| `superlevel_isClosed` | `F(t)` is closed when `g` is continuous |
| `superlevel_inclusion_id`, `superlevel_inclusion_comp` | the continuous inclusions satisfy the identity and composition laws |
| `superlevel_mem_at_own_height` | `x ∈ F(-g x)`; point membership, not component birth |
| `superlevel_neg_height` | `F(-h) = {x | h ≤ g x}` |
| `superlevel_control_strict_thresholds` | scalar control: below `0` there are intermediate witnesses; no `k` with `0 ≤ k < 0` |
| `superlevel_componentMap_mk`, `superlevel_componentMap_id`, `superlevel_componentMap_comp` | the induced map on `ZerothHomotopy` acts on representatives and is functorial |
| `superlevel_componentInclusion_id`, `superlevel_componentInclusion_comp` | the induced component inclusions are functorial |
| `superlevel_component_eq_iff_joinedIn` | equal path components iff `JoinedIn` the closed superlevel |
| `superlevel_h0Filtration_map_id`, `superlevel_h0Filtration_map_comp` | functor laws of the singular-H0 filtration; they put the H0 definition into the axiom audit |

Audited definitions: `torus_E`, `torus_quotient` (abbrev); `torus_latticeVec`, `torus_periodBasis`,
`torus_periodLattice`, `torus_proj`, `torus_isPeriodic`, `torus_descend`; instances
`torus_lattice_discrete`, `torus_lattice_addSubgroup_discrete`, `torus_lattice_isZLattice`;
`superlevel_set`, `superlevel_inclusion`, `superlevel_componentMap`, `superlevel_componentInclusion`,
`superlevel_filtration`, `superlevel_h0Filtration`. Every one of the 58 declarations is a
`#print axioms` target.

## What is NOT established

Copied from the handoff's own exclusions (TORUS_FILTRATION.md, source review F3–F4):

- critical-level attainment: no theorem produces a witness at the critical height; strict
  thresholds `h < 0` have no witness at `0`, and the path correspondence transfers predicates
  only;
- finitely many critical events; compactness plus closedness does not give finitely many path
  components for an arbitrary continuous height;
- label tracking, elder events, once-counting;
- naturality comparing the displayed component maps with singular-H0 generators beyond the
  functor laws actually proved;
- any interval decomposition, tameness, finiteness, barcode or endpoint-preserving bar bijection;
- any identification of `PairLawInterface.p`, or any other interface predicate of C253;
- smooth quotient charts, Morse index transfer, handle analysis, critical-free band invariance;
- a compiled theorem naming the externally owned `Specifications.Field` module: the companion
  carries the literal formulas (`torus_field_latticeVec_formula`, `torus_field_periodic_formula`)
  and does not import that file.

Compile success, the axiom audit, `leanchecker` and the negative controls say that Lean
elaborated these statements with these proofs at this pin. They do not say that the statements
are the ones the research programme needs; that is the pending independent alignment review.

## Compile fixes at the pin

_To be completed by the integrator from the compile-fix report (location, original text, fixed
text, reason for each change; confirmation that no theorem statement, hypothesis, quantifier or
definition content changed)._ The module docstrings carry the fixer's own one-line record of
each change and the namespace move; the compiled bytes are the ones `SOURCE_FILES.json` hashes.

## Namespace mapping

Every declaration keeps its handoff name: `ResearchFormalCoreR1.<n>` (handoff) ->
`TorusFiltration.<n>` (companion). The inventory lists in `SOURCE_FILES.json` are derived from
the originals by column-0 scans (`^theorem`, `^(noncomputable )?def`, `^abbrev`, `^instance`) and
`check.py source` refuses any module whose scan differs, any `lemma`/`example`/anonymous
`instance`, any remaining `namespace ResearchFormalCoreR1`, and any admission token
(`sorry`, `admit`, `axiom`, `native_decide`, `unsafe`, `opaque`, `partial`, `implemented_by`,
`extern`, `skipKernelTC`, `autoImplicit`) in comment-stripped code.

## Vocabulary

- `proved` is the source's own label for a declaration whose proof text it supplies; it is
  transcribed, not awarded here.
- `kernel-checked` appears only in `receipt.json`, written by `check.py finish` after a hosted
  run in which every build exited `0`, `leanchecker --fresh` passed and every axiom report lies
  in `{propext, Classical.choice, Quot.sound}`.
- The words `verified`, `closed` (as a status) and `accepted` do not describe anything in this
  companion. `alignment_status` stays `PENDING_INDEPENDENT_REVIEW`; `scientific_acceptance` is
  `false`; a same-provider review would earn zero independence credit and would be recorded as
  such.

## How to run

From a checkout with the pinned toolchain installed (`elan`, Lean v4.34.1) and the Mathlib
cache for `formal/` retrieved:

    bash companions/torus_filtration_20261011/replay.sh

The replay: binds HEAD to the manifest (`check.py source`), runs the Python tests in normal and
`-O` modes, emits the Lean audit files, then inside `lake env bash` from `formal/` records
`lean --version` and every installed package HEAD, compiles `TorusTopology`, `ClosedSuperlevel`
and the root `TorusFiltration` with `-DwarningAsError=true`, compiles `Positive.lean` (six
concrete applications of the companion's theorems), `Audit.lean` (58 `#print axioms`),
`Types.lean` (41 `#check` with `pp.explicit`), runs `leanchecker --fresh TorusFiltration`, and
runs six negative controls:

| Control | Kind | Exact expectation |
|---|---|---|
| `RejectSuperlevelSignFlip` | false statement: `() ∈ superlevel_set (fun _ => 0) (-1)` (true only if the parameter were `+t`) | exit 1, `error: unsolved goals` / `⊢ False` |
| `RejectSuperlevelBelowThreshold` | false statement: `3 ∈ superlevel_set id (-4)` | exit 1, `⊢ False` |
| `RejectConstantNotPeriodic` | false statement: a constant is not `torus_isPeriodic` | exit 1, `⊢ False` |
| `RejectSorry` | `theorem injected : False := by sorry` | exit 0 with the `sorry` warning; audit refuses `sorryAx` |
| `RejectCustomAxiom` | `axiom hiddenPremise : False` behind one intermediate theorem | exit 0; audit refuses `hiddenPremise` |
| `RejectNativeDecide` | `native_decide` proof | exit 0; audit refuses `injected._native.native_decide.ax_1_1` |

The injection controls run **without** warnings-as-errors on purpose: Lean accepts those files,
and only the axiom audit refuses them, which is what the audit is for. `check.py negative` uses
`re.fullmatch` on each log, so a crash, an extra error or a different exit code is a failed
control, not a passed one. `check.py finish` re-runs the source binding, checks every status,
the toolchain line, the installed package HEADs against `formal/lake-manifest.json`, the axiom
and type inventories and the six controls, and writes `receipt.json` with `log_sha256` over the
evidence directory. The evidence directory `.lake/torus-filtration-evidence` must not pre-exist.

Local protocol tests: `python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py`
and the same with `-O`. They exercise `check.py` against a temporary Git fixture with
inventory-exact placeholder modules; they execute no Lean.

## What this companion does not do

It does not modify `formal/`, the D2 companions or their workflows. It does not enter the q0
dependency graph, raise any claim's grade, or substitute for the nonauthor source review, the
independent alignment review or the operator's decisions. The hosted receipt is a record of a
computation, not evidence of alignment with the research programme.
