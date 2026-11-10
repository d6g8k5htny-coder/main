# SDGA Task 2: exact Lean bridge design

Status: staged design for root review; no implementation or execution is represented by this document.
Date: 2026-11-10 UTC. Scientific effect is NONE. Classification and qualification remain BLOCKED.

## Source boundary

The production source remains historical Math commit `f532c3e2d91a0c7679a37c733741660917a9b21b`, complete tree `b713ebf4d444f0f0a304e50302b1fe616155201a`, package `frontiers/source_derivative_gaussian_20261025`. Its package subtree is `a56e5c36f2b0395a268aea3054cbc5df9d13fa6e`. A separate read observed Math default `9bdfc54b76173b57912503bfcf901a5f9d43b09c` with that identical package subtree. This is a bounded package-identity result, not an atomic federation snapshot or a rebinding of historical runs.

The fixed profile is Main blob `4c6b4ea59eb07a17ebfed536b07688c53c01eb64`: 67 production modules, 25 ordinary inputs, seven elder inputs and 1,137 literal written public targets. The 1,128 Prop/nine data split is an expectation to reconcile with actual Lean judgment, not permission to manufacture the result. All compiled owned declarations are a separate population.

Original inputs remain lakefile `b4955b68b300c16869d01b56145b6c1a059b26db`, lock `9fe1332352893f5f89e74b550a25ee187b871ed3`, toolchain `ba8ebf2dbaf6a668cd2a0e086186d6d569b69ff5` and raw SOURCE_MANIFEST `cc7425b0f95f2bdc53a01395cac2f3eede566f7c`. Do not reconstruct the raw manifest from its labelled projection or impose a 2 MiB limit that rejects the actual 2,097,345-byte source.

The original Task 1 protocol source is `d00a866e55c5d20fd4aeb8d44e51fa471f1d1031`; the separately observed PR787 source `14d4617b523c29007da4bdb95ee6ee4dc588a86c` retains its Name, inventory and classifier section byte-for-byte. The prior design `docs/cbz/SDGA_COMPONENT_DESIGN.md`, blob `1decd319dde119230892753ff8ae5dc41cf9b8fe`, remains historical Task 1 context. Sequential metadata reads observed PR787 head `2ca2824a22af76ce46468b2504cb1ed2e2dbcdbc` and Main default `eed4e646565bf670d87b5ec41b9ae18f8a20d35a`; implementation must reread current refs and preserve compatible work.

## Deliverable and files

- `tools/sdga/Component.lean`: one core-Lean bridge, custom Name wire encoding, exact object loading, inventory, strict axiom closure and exact replay.
- `tools/sdga_component.py`: a small fixed-scope controller importing Task 1; immutable command templates, source/artifact guards, sequential scheduling and original receipts. No generic command runner API.
- `tests/sdga_component_contract.py`: explicit hosted/native control launcher, outside ordinary offline `test_*.py` discovery. Missing Lean is failure here, never skip.
- `tests/fixtures/sdga_component/`: minimal core-only package, five genuine Lean fixture modules and `Controls.lean`.
- `.github/workflows/sdga-component-contract.yml`: read-only hosted fixture execution with the pinned toolchain.
- Additive Task 2 documentation and later append-only integration records. Change no original Task 1 assertions, profile/config bytes, proof source, existing formal gate or historical outcome.

This milestone implements and qualifies the bridge against tiny fixtures. It does not execute the 99-module SDGA build/replay. That production observation is a separate root-reviewed invocation after fixture/source review.

## One executable and one replay helper

`Component.lean` imports core `Lean`, `Lean.Meta`, `Lean.Replay` and the needed core utilities; it does not import LeanChecker. It defines its own root `main : List String -> IO UInt32`. Shared SDGA helpers must be publicly importable by Controls; if the bridge uses a `module` header, place them in a public section. Run observation commands from the Main checkout (the controller uses an absolute Component path when its subprocess working directory is the fixture package). Invoke it as:

```sh
lake env env LEAN_PATH="$SDGA_OUT:$SDGA_DEP_LEAN_PATH" \
  lean --threads=2 --run tools/sdga/Component.lean inventory request.json
lake env env LEAN_PATH="$SDGA_OUT:$SDGA_DEP_LEAN_PATH" \
  lean --threads=2 --run tools/sdga/Component.lean replay request.json
```

Only `inventory` and `replay` are accepted; argument/JSON errors are nonzero. Test controls call the same exported helpers directly, with no production CLI mutation or arbitrary-command mode.

This is a new adapter choice. Historical ExactReplay blob `400f3853d50df7ea2a9b4bb2fc1de302626f2efa` remains unchanged. Its documented `#eval` invocation and imported LeanChecker root `main` must not be confused with this new direct-Lean.Replay executable.

Typed interfaces in namespace `SDGA`:

- `readRequest (path : System.FilePath) : IO ModuleRequest`.
- `loadExact (request : ModuleRequest) : IO LoadedModule`.
- `encodeName (name : Lean.Name) : String`; `decodeName (wire : String) : Except String Lean.Name`.
- `strictAxioms (lookup : Lean.Name -> Option Lean.ConstantInfo) (roots : Array Lean.Name) : Except AuditError (Array Lean.Name)`.
- `inventory (loaded : LoadedModule) : IO Unit`, writing the fixed Task 1 JSONL.
- `replayOwned (base : Lean.Kernel.Environment) (owned : Std.HashMap Lean.Name Lean.ConstantInfo) : IO Lean.Kernel.Environment`.
- `runReplay (loaded : LoadedModule) : IO Unit`, constructing the fresh direct-import base and calling `replayOwned`.

`ModuleRequest` has exactly schema=1, scope_id, profile_sha256, module (Task 1 Name wire), olean_parts and ir_parts. Each part has exactly level, absolute normalized path and measured sha256. OLean levels are exported/server/private; IR levels are ir.sig/ir. The controller alone selects modules from the unchanged production profile or the separately fixed fixture manifest. It validates source bindings, module/path relationships and measured part hashes before writing a request. The Lean bridge validates request structure, wire canonicality, resolved base OLean equality and corresponding part paths. Missing/extra parts and changing bytes are fatal.

SHA-256 is measured independently by the controller before and after the native process. The Lean bridge records the request's measured digest only after resolving the exact requested object; its repetition is not independent hash authentication. Preserve the request bytes, native output and controller measurements together.

`LoadedModule` retains the requested module identity, ordered raw OLean/IR data with live compacted regions, the final private ConstantInfo map and the actual import environment. Its lifetime must outlast every expression reference; never free compacted regions while those references remain.

## Inventory and identity rules

Read OLean parts together using `Lean.readModuleDataParts`, in exported/server/private order. A legacy non-module file has only exported data. A module-system file must have all three corresponding files; missing parts cannot degrade silently to legacy.

For every part, assert equal lengths and exact ordered equality of `constNames` and `constants.map ConstantInfo.name`; reject duplicates before making a map. Require every earlier part population to be a subset of final private. Keep raw arrays and hashes. Inventory every final ConstantInfo, including private, generated, data, inductive, constructor, recursor and foreign-namespace declarations.

Import with explicit `Lean.importModules`, `level := .private`, `loadExts := false` and recorded trust level zero. Cross-check each final name's actual owner through `getModuleIdxFor?`, `getModuleIdx?` and `header.moduleNames`. Namespace prefixes, names in the written target list and guessed generated-name patterns never determine ownership. Reject owner disagreement or unexplained duplicate/coalesced identities.

The fixed JSONL remains header/module/declaration/end with exactly Task 1 fields. A complete inventory may exit zero while reporting actual owned axioms, unsafe/partial declarations or OLean extras: those are observed facts for Task 1's blocking classifier, not successful qualification. Structural/loading/lookup/inference failures return nonzero and no successful end record. Replay policy blockers return nonzero before kernel replay. One native module transcript has module_count=1 and its complete actual declaration count. The controller verifies each transcript, rejects duplicate/missing modules, and forms one final 99-module transcript only after every fixed module succeeds. A partial transcript remains partial evidence, never a full-profile result.

`is_private` means `Lean.isPrivateName name` OR absence from that module's exported-part ConstantInfo names. This captures unexported module-system declarations and explicit private names in legacy files. `kind` maps the eight ConstantInfo constructors to Task 1's fixed strings.

The Name wire is exactly Task 1's component encoding: string component = `s` plus lowercase hexadecimal UTF-8 bytes; numeric component = `n` plus canonical nonnegative decimal; components are separated by `/`; anonymous encodes as empty but is forbidden as a module/declaration identity. Preserve empty string components, numeric versus string components, arbitrary Nat size and Unicode code points without normalization. Do not use dotted display, `String.toName` or default Name JSON for identity.

## Judgment and strict axiom closure

Observe `Meta.isProp ci.type` in the actual imported environment, using `MetaM.toIO`. Reject unexpected free variables, metavariables or loose binders before accepting the judgment. Inference exceptions/unsupported cases block; declaration kind is not a substitute. A definition whose type is True is a proposition-bearing proof; a definition whose type is Prop is data for this split. Pretty-print `type_display` as a diagnostic only; it is not a canonical/lossless AST or a semantic-equivalence certificate.

Use `ConstantInfo.getUsedConstantsAsSet` for the actual type, theorem/definition/opaque value and inductive/recursor links. Recursively resolve every reached name using the supplied actual lookup; a missing name is an `AuditError.missingConstant` and never an empty closure. Cache visited work without dropping edges. Do not use built-in `collectAxioms` as the authority: its missing-name branch and imported summaries are not this strict contract.

Audit all owned declarations, not merely public targets. Record the exact axiom union and per-declaration closure. An owned axiom always blocks, even if named exactly propext. Dependency axiom admission is by the three exact Names only: propext, Classical.choice, Quot.sound. Everything else blocks. Roles and exact type semantics remain unresolved/BLOCKED.

## OLean extras and separately bound IR

Task 1 `object_parts` denotes OLean ConstantInfo populations only. Its `extra_const_names` is the lossless union of raw `ModuleData.extraConstNames` from those owned OLean parts; preserve the per-part raw arrays in the sidecar and block any nonempty union.

Record all raw IR part names, arrays, paths, file hashes and relationship to the selected module in a separate artifact manifest. IR `extraConstNames` uses `includeDecls := true`, so legitimate final ConstantInfo names may occur there. Account for those exact CI-backed names without reporting a false unsupported-extra finding. Every remaining IR name is explicit unexplained residue with qualification BLOCKED until understood. Do not erase, guess a role, add silent Task 1 JSONL fields or claim full IR/native-kernel verification.

## Replay rules

For each selected module, create a new environment from its actual direct imports, at private visibility and with the selected module excluded. Before replay, require every owned name absent from that base. Any collision is explicit uncovered/blocking evidence; upstream tolerates some duplicate theorems by type/name/levels/all and can skip checking the differing proof value.

Validate safety/partial flags on every final ConstantInfo; any unsafe or partial item blocks all-owned replay coverage. The unchanged upstream operation is `base.replay owned`, ending in actual kernel `addDeclCore 0 0`. No success marker, simulated checker or preflight imitation substitutes for the native kernel operation.

The malformed-CI negative starts from an actual compiled safe True theorem and submits only that theorem in a singleton map. Preserve its identity/type/levels/safety and replace only its value with the actual Nat.zero expression. Assert the theorem is absent from the base and all referenced constants are present; submit a map containing exactly that single altered theorem through `replayOwned`. Require an actual kernel type rejection. Missing imports, malformed requests, compilation failures, timeouts and unrelated exceptions are not that rejection.

## Tiny genuine core fixtures and controls

Five modules suffice: `Fixture.Safe`, `Fixture.BadDependency`, `Fixture.AxiomUse`, `Fixture.Skipped` and `Fixture.OwnedAxiom`. Fixtures import only pinned core Lean/Init. The last uses a separate prelude-only environment and exact `public axiom propext : Prop`, avoiding imported-name collision.

The companion plan fixes each required subcase ID and its complete census; fifteen Python method counts alone do not establish execution. The fixed control IDs are C01 Name wire; C02 all-part/foreign/private/generated ownership; C03 actual Prop/data judgment; C04 type/value/opaque axiom routes; C05 missing lookup; C06 owned exact-allowlist axiom; C07 unsafe/partial; C08 safe actual replay; C09 malformed-CI actual kernel rejection; C10 contaminated-base refusal; C11 raw OLean extras; C12 malformed part tables; C13 missing object part; C14 IR raw provenance/accounted names/residue; C15 request/path/transcript identity.

For C04, a genuine dependency can define BadProof : True as an axiom, TypeAlias : Prop := (fun (_ : True) => True) BadProof, typeOnly : TypeAlias := True.intro and valueOnly : True := BadProof, plus an opaque counterpart. Assert the compiled typeOnly ConstantInfo actually retains the alias in its type and does not reference BadProof in its value before crediting type-only coverage.

Missing references and malformed ConstantInfo cannot be compiled as valid ordinary source; label the tests as alterations of actual loaded ConstantInfo/ModuleData and call the same helpers. For the contaminated-base control, populate the base by replaying the valid Safe map, then submit only the same singleton altered theorem; this isolates the upstream theorem-duplicate shortcut from unrelated duplicate definitions or inductives. A genuine nonempty compiler-extra fixture receives no source-generation credit until such an extra is actually observed. The altered metadata negative tests refusal only.

## Bounded hosted scheduling

Use a core-only fixture package with lean-toolchain leanprover/lean4:v4.34.1, no Mathlib and no dependency cache. Existing Math workflow blob `b9165ec9ce50337087ea19cad148fe68eb15b037` is the setup precedent: lean-action `24581d74d674803a25a81b10b059c5c0cdd344c6`, auto-config/build/test/lint false. For fixtures also set use-mathlib-cache=false and use-github-cache=false, contents:read and persist-credentials:false. Guard the actual installed `lean --githash` against the pinned release commit.

Later real SDGA compilation must be a fixed topological sequence, one compiler process at a time. Lake has no supported child-job cap in this pinned source; LEAN_NUM_THREADS is not a Lake child-process bound. Never invent Lake -j/--jobs/LAKE_JOBS. Replace LEAN_PATH after lake env, excluding every old owned/formal build directory:

```sh
lake env env LEAN_PATH="$SDGA_OUT:$SDGA_DEP_LEAN_PATH" \
  lean --threads=2 --root="$SDGA_SRCROOT" \
  -o "$SDGA_OUT/$MODULE_PATH.olean" -i "$SDGA_OUT/$MODULE_PATH.ilean" \
  "$SDGA_PKG/$SOURCE_REL"
```

The source roots are the package root for production, inputs for ordinary inputs and inputs/elder for elder inputs. Destination paths derive from module names, without source-directory prefixes. Create destination parents first, await completion and stop dependents after failure. Module-system -o writes its server/private and IR siblings automatically; preserve them.

Use pinned `lean --deps-json` over the exact fixed source vector to obtain header imports. It returns zero even when individual rows contain errors: require exactly one result per file, every result present and every errors array empty. Then reject duplicates, missing local edges or cycles; do not assume lakefile order is topological. This is direct Lean elaboration, not literal Lake build, complete Lake facets, G3 or a full cache/native/kernel audit.

## Evidence and resource limits

Importing private Mathlib data, retained compacted regions, map copies, pretty printing and transitive closure can consume substantial memory. Process one owned module per fresh native process, stream JSONL, avoid retaining whole rendered transcripts in Lean and cache closure visits within a process. Every timeout, signal, OOM, malformed/truncated output, missing stage, changed source/object or unsupported residue remains blocking. No silent retry, skip or truncation.

Retain exact source/tool/request/part hashes, before/after source and dependency measurements, UTC and monotonic timing, exit/signal/timeout, original stdout/stderr, resource observations and native workflow run/attempt. The Task 1 receipt validator still validates declared fields only; its execution_authenticated remains false. Separately observed native artifacts describe only their actual executed component.

## Pinned API evidence

All Lean paths below are in leanprover/lean4 release commit `5045d0056413266e57c625dcd7c365b10e377c52`.

| Source | Git blob | Interfaces/boundary |
|---|---|---|
| src/Lean/Environment.lean | 33928512dadcfa2e42a307499e9228596da0341a | ModuleData109-128; owner APIs1195-1211; readModuleDataParts1782-1790; private/parts/import2438-2449; raw IR1890-1898 |
| src/Lean/Replay.lean | 71012211da8d22961873d9ff70feadac973715a8 | kernel addDeclCore59-60; duplicate theorem84-96; public replay176-190 skips unsafe/partial |
| src/Lean/Declaration.lean | 52a32d4fdeae1fe12e46cd46a1635420a3b62195 | ConstantInfo430-438; isUnsafe453-461; isPartial463-465; value?483-488 |
| src/Lean/Util/FoldConsts.lean | 5429906bfda3805967920ea17f77e68611007fdf | ConstantInfo dependency traversal67-73 |
| src/Lean/Util/CollectAxioms.lean | dc9897059a4463dfd143a2b3f8fbb4f4a3c3f154 | Missing lookup/imported-summary shortcuts are insufficient |
| src/Lean/Meta/InferType.lean | 0e3d57afdffbb1d4d62bb423b8847afa4216af13 | Meta.isProp323-332 |
| src/Lean/Meta/Basic.lean | e207f8f137bda69488be8424896fd82bad6cca8d | MetaM.toIO648-650 |
| src/Init/Prelude.lean | 66af67237683d3c7f8230bcf473c0871e67749a2 | Name anonymous/str/num4725-4742 |
| src/Lean/PrivateName.lean | 80f0d81c37f6b98fabf4196d1bfdbd13cbed70f6 | isPrivateName32-35 |
| src/Lean/Data/Json/FromToJson/Basic.lean | 2b2815f5d412cd0ece92838b0a2b0267ddd563d0 | Default Name JSON183-187 is display, not lossless wire |
| src/Lean/Shell.lean | 3f73f6fe35d11cbccd9b6effe1bafad030fdbdce | root/-o/threads149-165; deps-json386-387,489-496 |
| src/Lean/Elab/ParseImportsFast.lean | 7ef54dfcae1459216929a50bd4317934c9e37c47 | Per-file JSON errors caught268-274 |
| src/lake/Lake/Config/Workspace.lean | 92df575a8aa0cd7ea4a85b0831986359d40a735f | lake env prepends workspace Lean paths355-392 |
| src/lake/Lake/Build/Context.lean | c4690f1031faec12983ac07d47245a25cbdf31e6 | BuildConfig16-62 has no child-job cap |
| src/lake/Lake/Build/Job/Monad.lean | bf8e86ae2798b6c516bb4599dfc91c574461b7f3 | Job.async168-173 starts BaseIO task |
| src/lake/Lake/Util/Proc.lean | 9ec2cba3546f9067f6121c1b9f6c2bd7122293ba | rawProc29-36 directly starts process |

Pinned lean-action commit is `24581d74d674803a25a81b10b059c5c0cdd344c6`; action.yml blob `8ebeca15357a8c0b1de96190608f7ba877353bc3` defines the explicit setup inputs. Source-only review of these interfaces is not execution evidence.

## Review gate

Root reviews this additive design and the companion implementation plan before any code. Review the exact field semantics, clean-base guard, fixture causality and hosted command census. No parent/full G3, full federation G0-G11, CBZ, scientific-status or proof-closure credit is available from this milestone.
