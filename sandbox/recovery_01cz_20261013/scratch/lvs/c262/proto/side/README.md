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

The C254 handoff delivered both modules as NOT_COMPILED. At the pin they did not elaborate; the successor bytes in
this companion differ from the handoff bytes only as listed here. No theorem or lemma statement changed (all 41
statement lines are byte-identical before `:=`); the 21 + 20 theorem names and the 11 + 6 definition names keep their
order; no `sorry`, `axiom`, `native_decide` or `set_option` was added; every declaration's axioms are within
`propext`, `Classical.choice`, `Quot.sound`.

`TorusTopology.lean` (handoff sha256 `efa80d2879758b6d6473a0024a308ff758c308cdc8e4b5f400f39dde63a7f6f4`):
- `def torus_periodBasis`: declared type `Basis (Fin d) ℝ (torus_E d)` → `Module.Basis (Fin d) ℝ (torus_E d)`. At the
  pin the structure is `Module.Basis` (`Mathlib/LinearAlgebra/Basis/Defs.lean`, `namespace Module` line 76, `structure
  Basis` line 90); bare `Basis` was an unknown identifier that `autoImplicit` turned into a variable, the root of the
  first-run cascade. Same object; the body is unchanged.
- `theorem torus_basis_sum_eq_latticeVec`, proof: `Basis.unitsSMul_apply` → `Module.Basis.unitsSMul_apply` in both
  `simp` calls (`Mathlib/LinearAlgebra/Basis/SMul.lean` line 107); three unused simp arguments removed
  (`PiLp.single_apply` twice, `hij`) so the file elaborates with warnings as errors.
- `theorem torus_proj_isCoveringMap`, proof term: the discreteness witness `DiscreteTopology.isDiscrete` (which needs an
  instance on the set-coercion subtype that instance search did not bridge) → `SetLike.isDiscrete_iff_discreteTopology.mpr
  (torus_lattice_addSubgroup_discrete d L hL)` (`Mathlib/Topology/Constructions.lean` line 268), passed to the same
  lemma `AddSubgroup.isAddQuotientCoveringMap_of_comm`.
- `theorem torus_pathConnectedSpace`, proof: `infer_instance` → `exact inferInstanceAs <| PathConnectedSpace (Quotient _)`
  (the pattern Mathlib uses for `AddCircle`; `infer_instance` does not unfold `HasQuotient.Quotient` to reach
  `Quotient.instPathConnectedSpace`).
- Module docstring: two lines recording the above; `namespace ResearchFormalCoreR1` → `namespace TorusFiltration`
  (companion namespace; no qualified self-reference existed).

`ClosedSuperlevel.lean` (handoff sha256 `1fed3edada12b4dc7f5fcf874dae7ccde5d8f65af63cf103159debe58ba6789e`):
- `def superlevel_componentMap`: the second argument of `ZerothHomotopy.lift` is `∀ ⦃x y : X⦄, Path x y → f x = f y`
  (`Mathlib/Topology/Connected/PathConnected.lean` lines 153-157); strict-implicit binders are not introduced by `fun`,
  so `fun p => …` bound `p` to a point. Fixed to `fun _ _ p => ZerothHomotopy.sound (p.map f.continuous)`; the witness
  and the defined map are the same.
- Module docstring: three lines recording the above; `namespace ResearchFormalCoreR1` → `namespace TorusFiltration`.

Evidence of the fixes themselves is local (`lean -DwarningAsError=true --root=<companion>` inside the pinned `formal/`
Lake environment, exit 0 for both modules, `#print axioms` of all 58 declarations within the three allowed axioms): a
record of a computation on an unverified worktree, not a receipt. The receipt is the hosted companion run.

Both module docstrings still read `C254 proposed source. Compilation and kernel audit: NOT_RUN.`
(line 6 of each file). That line is the handoff's own status label, retained verbatim as a
transcribed source label; it is not updated by this companion. Whether the modules compiled and
were replayed through the kernel is recorded in the hosted `receipt.json`
(`formalization_status`, `leanchecker`), never in the module text, so the module line
under-states and the receipt is the record.

## Namespace mapping

Every declaration keeps its handoff name: `ResearchFormalCoreR1.<n>` (handoff) ->
`TorusFiltration.<n>` (companion). The inventory lists in `SOURCE_FILES.json` are derived from
the originals by column-0 scans (`^theorem`, `^(noncomputable )?def`, `^abbrev`, `^instance`).
`check.py source` re-derives them from the companion modules with a whitelist scanner
(`scan_module`, not a Lean parser) and refuses any module whose scan differs. The scanner strips
comments (`--`, nested `/- -/`, `/-- -/`, `/-! -/`) and then permits, at column 0, only:
`import Mathlib` or `import Mathlib.…` (before the namespace), `noncomputable section`,
`open <Namespaces>`, `universe …`, `variable …`, exactly one `namespace TorusFiltration` /
`end TorusFiltration` pair, and inside it a named `theorem`, `def`, `noncomputable def`, `abbrev`
or `instance` whose name matches `[A-Za-z][A-Za-z0-9_]*`. Everything else is refused: any other
command keyword anywhere outside comments (`lemma`, `example`, `axiom`, `opaque`, `structure`,
`inductive`, `class`, `private`, `protected`, `noncomputable instance`, an anonymous `instance`,
`attribute`, `set_option`, `section`, `end`, `mutual`, `deriving`, `macro`, `macro_rules`,
`syntax`, `elab`, `notation`, `scoped`, `local`, `export`, `include`, `omit`, `run_cmd`,
`initialize`, `… in`, a second command on a declaration line, an indented command), any `@[`
or `#`, any string literal (`"`), a primed or dotted declaration name, a declaration outside
the namespace or after `end TorusFiltration`, any remaining `namespace ResearchFormalCoreR1`,
and any admission token (`sorry`, `sorryAx`, `admit`, `axiom`, `native_decide`,
`ofReduceBool`, `trustCompiler`, `unsafe`, `opaque`, `partial`, `implemented_by`, `extern`,
`skipKernelTC`, `autoImplicit`) in comment-free code, and the declaration-producing commands of
Batteries, Aesop and Mathlib at the pin (`alias`, `irreducible_def`, `simproc`, `simproc_decl`,
`dsimproc`, `dsimproc_decl`, `builtin_simproc`, `builtin_dsimproc`, `notation3`, `unif_hint`,
`recall`, `binder_predicate`, `add_decl_doc`, `library_note`, `assert_not_exists`,
`register_simp_attr`, `declare_aesop_rule_sets`, `compile_inductive`, `compile_def`, `nonrec`).
The refused keywords are the explicit enumeration `COMMANDS` in `check.py`, not a Lean grammar:
a declaration-producing command absent from that enumeration would pass an indented or
same-line position uncounted, and the `#print axioms` audit covers the inventoried names only.
The C262 gate review (Math- #428 comment 6112740935, finding G2) found `alias` absent from the
enumeration at head `a898bb44`; the enumeration now lists it and the other commands above, and
the census claim is bounded to that list. A compile fix that needs a refused form must extend
the scanner and the inventory explicitly.

## Vocabulary

- `proved` is the source's own label for a declaration whose proof text it supplies; it is
  transcribed, not awarded here.
- `kernel-checked` appears only in `receipt.json`, written by `check.py finish` after a hosted
  run in which every build exited `0`, `leanchecker` replayed the companion modules' declarations
  through the kernel without output (the exact invocation is recorded in `leanchecker.cmd` and
  copied into the receipt), and every axiom report lies in `{propext, Classical.choice, Quot.sound}`.
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
`Types.lean` (41 `#check` with `pp.explicit`), runs
`leanchecker TorusTopology ClosedSuperlevel TorusFiltration`, and runs six negative controls.
The emitted statements below are quoted verbatim from `check.py` (`FALSE_CONTROLS`); each is
elaborated as `example : <statement> := by <tactic>` and must fail with exactly `⊢ False`:

| Control | Kind | Exact expectation |
|---|---|---|
| `RejectSuperlevelSignFlip` | false statement `() ∈ superlevel_set (fun _ : Unit => (0 : ℝ)) (-1)` (true only if the parameter were `+t`), tactic `norm_num [superlevel_set]` | exit 1, `error: unsolved goals` / `⊢ False` |
| `RejectSuperlevelBelowThreshold` | false statement `(3 : ℝ) ∈ superlevel_set (fun x : ℝ => x) (-4)` (`3 ∈ {x | 4 ≤ x}`), tactic `norm_num [superlevel_set]` | exit 1, `⊢ False` |
| `RejectConstantNotPeriodic` | false statement `¬ torus_isPeriodic 1 (1 : ℝ) (fun _ => (0 : ℝ))`, tactic `simp [torus_isPeriodic]` | exit 1, `⊢ False` |
| `RejectSorry` | `theorem injected : False := by sorry` | exit 0 with the `sorry` warning; audit refuses `sorryAx` |
| `RejectCustomAxiom` | `axiom hiddenPremise : False` behind one intermediate theorem | exit 0; audit refuses `hiddenPremise` |
| `RejectNativeDecide` | `native_decide` proof | exit 0; audit refuses `injected._native.native_decide.ax_1_1` |

The injection controls run **without** warnings-as-errors on purpose: Lean accepts those files,
and only the axiom audit refuses them, which is what the audit is for. `check.py negative` uses
`re.fullmatch` on each log, so a crash, an extra error or a different exit code is a failed
control, not a passed one. `check.py finish` refuses an evidence directory that already holds
`receipt.json` (a receipt is written once and never replaced; a new attempt needs a fresh
directory) and requires exactly the 41 evidence files, each a regular file and nothing else: the
nine emitted Lean files, a `.log` and `.status` pair for each of the seven Lean processes
(three module builds, `positive`, `axioms`, `types`, `leanchecker`), the six controls'
`.lean`/`.log`/`.status`, `leanchecker.cmd`, `packages.log`, `toolchain.log`, `source.json`,
`tests.log` and `tests-optimized.log` (`EVIDENCE_FILES` in `check.py`; C262 finding G3). It then
re-runs the source binding, re-derives the nine emitted Lean files and requires byte equality
with the ones Lean consumed, checks every status, the toolchain line, the installed package
HEADs against `formal/lake-manifest.json`, the recorded `leanchecker` invocation, the axiom and
type inventories and the six controls, and writes `receipt.json` with `log_sha256` over exactly
those 41 files. The evidence directory `.lake/torus-filtration-evidence` must not pre-exist.

### What `leanchecker` checks here, and why not `--fresh`

Both modules `import Mathlib`, so their import closure is all of Mathlib (6.1 GB of `.olean`
files at the pin). `leanchecker --fresh TorusFiltration`, the shape of the D2 companions' step
(`leanchecker --fresh D2MomentBridge`, affordable there only because that module imports
`Mathlib.Tactic` and one Bochner file), re-checks that entire closure from scratch: measured locally against the pinned cache at 4 threads it was still
running at 1500 s with 6.6 GB resident, so it cannot complete inside a hosted job. The hosted
default is therefore `leanchecker TorusTopology ClosedSuperlevel TorusFiltration`: leanchecker
builds the environment from the cached, prebuilt dependency `.olean` files (trusted, exactly as
`lean` itself trusts them when compiling the modules) and replays every declaration of the three
named modules through the kernel. Measured locally at `LEAN_NUM_THREADS=2`: 30 s wall,
6.6 GB resident, no output on success (the same footprint as one `lean` compile of
`ClosedSuperlevel.lean`, 9 s and 6.8 GB). `replay.sh` writes the exact invocation to
`leanchecker.cmd`; `check.py finish` accepts only the two pinned forms, requires the modules-mode
log to be empty, and copies command and mode into `receipt.json` under `leanchecker`.
An operator who wants the whole-closure re-check runs
`TORUS_LEANCHECKER_FRESH=1 bash companions/torus_filtration_20261011/replay.sh` locally with the
time and memory to spare; the receipt then records mode `fresh`. The workflow never sets that
switch.

Hosted duration: measured on the first green run,
[38165155316](https://github.com/d6g8k5htny-coder/Math-/actions/runs/38165155316) (attempt 1, job
114546073697, PR #428 head `0dd0cb17bfa12bab6983bc4b068a38fed5a9df9c`, tested merge
`aa288a1377acc76bfb544de1c5e3d3b4c282a5e3`): the job took 4 min 20 s end to end (2026-10-11T18:45:38Z to
18:49:58Z), of which the pinned-Lean install plus Mathlib cache retrieval took 2 min 47 s and the
replay step itself 1 min 27 s at `LEAN_NUM_THREADS=2`; artifact `torus-filtration-38165155316-1`,
42 files, 22,572 bytes, SHA-256 `c9bf02d33ddd9addb5d2883af982827b44ef3fd5e84d877ed78504efe8f19c0d`.
The workflow's `timeout-minutes: 45` is kept: a cache miss or a slow runner needs the margin and
lowering it buys nothing. One hosted completion is recorded here; every later run's receipt stands
on its own, and a receipt is a record of a computation, not evidence of alignment.

The replay writes `TorusTopology.olean`, `ClosedSuperlevel.olean` and `TorusFiltration.olean`
into the companion directory (Lean needs them on `LEAN_PATH` for the root module and the audit
files). They are build artefacts and must never be committed: the repository has no root
`.gitignore`, `git add -A` would stage them, and `check.py source` then refuses the companion
with `tracked companion membership differs` until they are removed in a follow-up commit.
Delete them before committing.

Local protocol tests: `python3 -B -S companions/torus_filtration_20261011/test_torus_filtration.py`
and the same with `-O`. They exercise `check.py` against a temporary Git fixture with
inventory-exact placeholder modules; they execute no Lean. The fixture spawns its `check.py`
children with the outer interpreter's optimization level (`-O` under `-O`) and strips
`PYTHONOPTIMIZE` from their environment, and two controls observe the mode the children actually
run in (`sys.flags.optimize` reported from inside the child, including from a child running the
real `check.py source`). Until head `a898bb44` the children ran unoptimized whatever the outer
mode was (C262 finding G1): the `tests-optimized.log` transcripts of hosted runs 38165155316 and
38165763430 therefore cover the imported helpers under `-O` and the CLI paths unoptimized; that
narrower attribution stands for those runs and is not revised.

## Amendments

- C262 (Math- #428, nonauthor engineering gate review [6112740935](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112740935),
  disposition [6112743879](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112743879), AMEND_ENGINEERING_GATE on
  head `a898bb44`): G1 the test fixture now propagates the outer optimization level to its `check.py`
  children, strips `PYTHONOPTIMIZE` and observes the child mode; G2 the scanner refuses `alias` and
  the other declaration-producing commands listed above, with indented, same-line and column-0
  negative controls, and the census claim is bounded to the enumeration; G3 `finish` never replaces
  a receipt and requires the exact 41-member evidence set of regular files, with controls for a
  repeat `finish`, an existing receipt beside a newly failing input, every removed member, an extra
  member and a symlinked member. No Lean byte, declaration, control statement or workflow line
  changed. The same review's Lean source verdict (PASS_SOURCE_SCOPE,
  [6112740740](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112740740)) and hosted-evidence verdict
  (APPROVE_OBSERVED_EXECUTION_SCOPE, [6112741132](https://github.com/d6g8k5htny-coder/Math-/pull/428#issuecomment-6112741132))
  were of head `a898bb44`; organizational-independence credit 0 throughout.

## What this companion does not do

It does not modify `formal/`, the D2 companions or their workflows. It does not enter the q0
dependency graph, raise any claim's grade, or substitute for the nonauthor source review, the
independent alignment review or the operator's decisions. The hosted receipt is a record of a
computation, not evidence of alignment with the research programme.
