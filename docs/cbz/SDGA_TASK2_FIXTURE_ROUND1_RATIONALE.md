# SDGA Task A fixture family, round 1: source-only proposal

Date: 2026-11-10 UTC. Root review required before publication. No new native execution is claimed.

## Candidate and preserved contracts

Candidate intended only for `tests/fixtures/sdga_component/Fixture/Safe.lean`: UTF-8 Git blob `35c407e867b19e60a22a3cca39edf49c4a07b088` in `d6g8k5htny-coder/main`. Exact UTF-8 readback matched the authored source.

The candidate preserves the three literal targets `Fixture.Safe.proof`, `Fixture.Safe.propValue`, `Fixture.Safe.natValue` and their True/Prop/Nat types and values. It preserves the public safe theorem, explicit private theorem and foreign-namespace theorem. `Switch` changes only its universe from Type to Prop; it still has exactly two source constructors and the equation-style `flip : Switch -> Switch`. Actual generated recursor CIs, raw owner rows and zero extras remain native obligations, not source-review facts.

For the single plain-name/unexported fixture declaration, it adds only a compile-time `meta import Lean` and a `run_elab` action using existing pinned checked declaration and visibility-branch APIs. The action creates safe `Fixture.Safe.unexported : Nat := 11` with its explicit ordinary raw Name. Kernel checking is explicitly enabled (`debug.skipKernelTC=false`); the action's checked addition is synchronous (`Elab.async=false`).

No Component/controller/control/launcher/test/workflow/profile/configuration/proof assertions are changed. In particular, Controls `b8ec51fbcee8516fcfeb89a637fa08097d918602`, launcher `d7f72edb1b7dac6458c568e9411edf21f2703476` and the existing pure dependency-test source must remain byte-identical. No ref, tree, commit, PR, comment or rerun was created by this work.

## Actual preceding native evidence is retained as adverse evidence

I read the primary logs of run `42743801064`, attempt 1, job `129590978474`. Both modes successfully compiled Component `ecb422db32b1825b71833e91120de3dbc59fd7a6` and emitted the complete actual inventory. The workflow-recorded original inventory SHA-256 is `a7fbc636c99e34078eb043c7879acd57c579c67c320c60363e16872f8143316b`, 20,086 bytes, identical in normal and optimized modes.

The original Safe source `814c0f3169b4e967f5b4940d205a479effebeed1` actually yielded 26 declaration rows, exported/server/private populations 23/23/26 and eleven genuine OLean extras. The emitted `unexported` CI is private-name mangled; there is no plain raw-name match. Task 1 correctly found UNSUPPORTED_EXTRAS and left classification BLOCKED. Each mode's six native methods failed their fixture prerequisite, leaving all 39 subcases not_run; no dormant control received semantic credit. The separate pure tests passed 12/12 in both modes.

Keep that original source, original native inventory and all eleven extras as source-bound genuine adverse evidence for the later C11 fixture work. They must not be filtered out, renamed, relabelled as green, or credited as a passing control. The production bridge requires no change to report these observations.

## Why ordinary source modifiers are insufficient

All following references are actual source at leanprover/lean4 commit `5045d0056413266e57c625dcd7c365b10e377c52`.

- DeclModifiers.lean blob `21c664badcd7d886d2be79349e3f3c21476d3a81`, lines 77-85 and 240-248: default nonpublic module declarations are interpreted private and `applyVisibility` calls `mkPrivateName`. That explains the observed mangling.
- AddDecl.lean blob `512ba0ff2cdf52843b9f5425ec636e94d15ab92b`, lines 119-124 and 138-157: ordinary `Lean.addDecl` under nonexporting scope still supplies an exported representation for a nonprivate spelling (a definition is weakened to an exported axiom). Merely using `withoutExporting` would not create the required population difference.

## Checked branch route and error propagation

The fixture uses this narrow, documented route:

1. `env.addConstAsync name .defn (exportedKind? := none)` creates the visibility split.
2. `setEnv pending.asyncEnv` enters that declaration's branch.
3. `Lean.addDecl (.defnDecl ... safety := .safe)` performs the actual checked addition of the Nat11 definition.
4. Only after successful return, `pending.commitCheckEnv (getEnv)` commits its real CI and checked kernel environment.
5. `setEnv pending.mainEnv` restores the outer branch whose exported map omits that raw identity.

Pinned source support:

| Source/blob | Lines | Relevant behavior |
|---|---|---|
| Environment `33928512dadcfa2e42a307499e9228596da0341a` | 1004-1071 | `exportedKind?=none` is documented for unexported constants; the outer mainEnv adds a private entry and leaves its public map unchanged. |
| AddDecl `512ba0ff2cdf52843b9f5425ec636e94d15ab92b` | 30-33, 177-194, 229-231 | `debug.skipKernelTC=false` passes doCheck=true into Environment.addDeclCore; Elab.async=false executes the checking branch synchronously. A checking exception attempts recovery and is rethrown. |
| Environment `33928512dadcfa2e42a307499e9228596da0341a` | 695-715 | `addDeclCore` uses addDeclCheck when doCheck=true and explicitly permits callers to customize public visibility through addConstAsync. |
| Environment `33928512dadcfa2e42a307499e9228596da0341a` | 1091-1136 | commitConst requires a found actual CI and matching name/kind/signature; commitCheckEnv commits it together with the checked kernel environment. |
| CoreM `61d372630f94a66febd188f998aa81b1361a4ce0` | 35-42 | Elab.async is the actual built-in option; the command line otherwise overrides its default. |
| BuiltinEvalCommand `2ec9947250df88e78330210acb37f9e960cb8d45` | 230-247, 252-253, 282-287 | run_elab compiles its evaluator inside withoutModifyingEnv, restores that temporary environment, executes the action, and reports action exceptions as errors. |
| Frontend `965a761b628ebc7b7c780c7a6d12fde6e2e9b339` | 378-385 | Snapshot errors return before normal writeModule serialization. |
| Environment `33928512dadcfa2e42a307499e9228596da0341a` | 1860-1884, 1901-1909 | The unchanged writer emits all checked local CIs in private data, while exported/server data filters through exporting-scope find?. |

The outer promise is visibility plumbing, not an unchecked declaration insertion. The candidate does not construct a standalone ConstantInfo, alter loaded ModuleData, call addDeclWithoutChecking, disable checking, inject unsafe data or commit on a checking-error path. Because the checked call is synchronous and errors propagate, it cannot reach the outer commit after a kernel rejection. A promise's error-recovery fallback does not make a successful compilation: reported errors prevent normal object serialization.

The source uses Lean.addDecl rather than addAndCompile for this compile-time-added declaration. The compiled OLean CI is the Task A fixture subject; this proposal provides no native-code/IR coverage claim for that declaration, and Task D IR qualification remains pending.

## Independent source review and remaining native obligations

A fresh scoped reviewer read the exact candidate blob and pinned API sources independently. It found no concrete source blocker and confirmed the checked/error-propagating route, outer visibility split, same-writer population semantics and restoration of evaluator temporary declarations. This review did not execute the source.

Root must still publish only after reviewing the candidate and this rationale, then require fresh native evidence for successful compilation, raw plain-name/private-only population, both real constructors/generated recursor, actual ownership, empty OLean extras, exact 3/1/2 target split, complete unchanged JSONL, all six methods and every one of the 39 subcase IDs in both modes. Any failed or unrun stage remains blocking. No criterion may be weakened to match the candidate.

Scientific/global/mathematical credit: NONE. scientific_effect=NONE; qualification=BLOCKED; classification_status=BLOCKED; execution_authenticated=false.
