# SDGA Task 2 Lean Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a small actual Lean inventory/replay bridge and qualify its fixed contracts with core-only hosted fixtures while every SDGA classification/qualification remains BLOCKED.

**Architecture:** One `Component.lean` exposes exact loading, Name encoding, inventory, missing-fatal axiom traversal and a shared actual-kernel replay helper. A small Python controller uses the existing Task 1 protocol, fixed requests and sequential command templates; it retains native evidence and separately bound IR provenance. The first milestone executes only tiny core fixtures; the fixed 99-module SDGA observation remains a later root-reviewed invocation.

**Tech Stack:** CPython 3.11 standard library, pinned Lean/Lake 4.34.1 release `5045d0056413266e57c625dcd7c365b10e377c52`, core Lean only for fixtures, GitHub Actions read-only setup.

**Spec:** `docs/cbz/SDGA_TASK2_LEAN_BRIDGE_DESIGN.md`, immutable Main blob `3487f54476b5e6a283839902797bc05aae2ce623` ([Git object](https://api.github.com/repos/d6g8k5htny-coder/main/git/blobs/3487f54476b5e6a283839902797bc05aae2ce623)). Read both documents. This is staged planning only; root reviews their immutable blobs before code. No native execution was performed to prepare this plan.

## Global Constraints

- Historical production source is Mathf532 `f532c3e2d91a0c7679a37c733741660917a9b21b`; package subtree `a56e5c36f2b0395a268aea3054cbc5df9d13fa6e` equals the separately observed Math9bdf `9bdfc54b76173b57912503bfcf901a5f9d43b09c` subtree. Do not rebind old runs.
- Frozen profile blob `4c6b4ea59eb07a17ebfed536b07688c53c01eb64`, its 99 module/path/source bindings and 1,137 literal targets remain unchanged.
- Keep Task 1 JSONL, lossless Name wire and classifier contract unchanged. Original source `d00a866e55c5d20fd4aeb8d44e51fa471f1d1031`; current observed source `14d4617b523c29007da4bdb95ee6ee4dc588a86c` preserves that contract. Reread its actual current source before integration.
- scientific_effect=NONE; qualification/classification_status=BLOCKED. Task 1 declared receipt execution_authenticated stays false.
- No source proof, bound, historical failure, existing formal gate, frozen input/config/profile or Task 1 assertion changes. No generic AST serializer, generic command/orchestration framework or arbitrary module filter.
- No full G3, whole IR/native/kernel coverage, final G0-G11, CBZ or scientific-acceptance claim.
- Hosted execution only while local environment is unavailable. No local command result may be inferred from source inspection. Use isolated branches, fresh expected-head guards and ordinary protected checks at later publication/integration.
- All process invocations are sequential. LEAN_NUM_THREADS is not a Lake child-job cap; do not invent Lake jobs flags.
- Partial, missing, skipped, timed-out, cancelled, drifted, resource-failed or unsupported stages are blocking. Preserve first failures; no implicit retry.
- Root must review the design and exact file scope before implementation. This document authorizes no ref, PR, comment or implementation mutation by itself.

## Review Focus

- A valid-looking duplicate theorem already in the replay base can suppress checking its altered proof; Task C tests guard rejection before upstream replay.
- Module-system private declarations may lack private-name spelling; Task A tests exported-population difference and actual owner mapping.
- Missing transitive type or opaque-value dependencies can disappear under a permissive collector; Task B tests each missing route through the same strict helper.
- Per-file deps-json errors coexist with exit zero; Task E requires every row/result/error guard and an adverse malformed-header case.
- IR extraConstNames includes ordinary CI-backed names; Task D preserves raw arrays, accounts exact matches and blocks unexplained residue without false extras.

## File map and exact interfaces

Create `tools/sdga/Component.lean`, namespace SDGA, with these interfaces. Shared helpers must be public to Controls; use a public section if the file has a module header:

- `readRequest (path : System.FilePath) : IO ModuleRequest`.
- `loadExact (request : ModuleRequest) : IO LoadedModule`.
- `encodeName (name : Lean.Name) : String`; `decodeName (wire : String) : Except String Lean.Name`.
- `strictAxioms (lookup : Lean.Name -> Option Lean.ConstantInfo) (roots : Array Lean.Name) : Except AuditError (Array Lean.Name)`.
- `inventory (loaded : LoadedModule) : IO Unit`.
- `replayOwned (base : Lean.Kernel.Environment) (owned : Std.HashMap Lean.Name Lean.ConstantInfo) : IO Lean.Kernel.Environment`.
- `runReplay (loaded : LoadedModule) : IO Unit`.
- Root `main : List String -> IO UInt32`, accepting only `inventory request.json` or `replay request.json`.

`ModuleRequest`: exactly schema=1, scope_id, profile_sha256, module (Task 1 wire), olean_parts, ir_parts. Part records: exactly level, path, sha256. OLean levels exported/server/private; IR levels ir.sig/ir. `LoadedModule` retains exact identity, raw ordered part data/live regions, final private ConstantInfo map and the actual import environment. `AuditError` must distinguish missingConstant, ownedAxiom, forbiddenAxiom, malformedConstant, unsafeReplaySkip, partialReplaySkip, duplicateBase, unsupportedExtras and unsupportedIRResidue; diagnostics carry lossless names.

Create `tools/sdga_component.py` with narrowly bounded interfaces:

- `parse_dependency_headers(payload: str, expected_files: tuple[str, ...]) -> dict[str, tuple[str, ...]]`: require exact ordered result census, every result present and errors empty.
- `topological_modules(edges: dict[str, tuple[str, ...]], expected: tuple[str, ...]) -> tuple[str, ...]`: fixed vertices, missing-local/duplicate/cyclic refusal.
- `compile_owned_module(source_root: Path, source_file: Path, module: str, output_root: Path, dependency_path: str, evidence_root: Path) -> dict`: fixed Lean command, fresh destination parents, no command argument supplied by caller.
- `observe_module(mode: str, request_path: Path, output_root: Path, dependency_path: str, evidence_root: Path) -> dict`: only inventory/replay, fixed bridge command; actual exit/stream/resource evidence.
- `combine_inventory(transcripts: tuple[str, ...], scope: dict) -> str`: exact complete module set, duplicate-free declarations, correct counts and union; parse through unchanged Task 1.
- Controller CLI: `python3 -B -S tools/sdga_component.py fixtures --output <new-external-dir>`. No production-build CLI is added in this fixture milestone.

Create `tests/sdga_component_contract.py`, an explicit unittest launcher using the controller and shared Lean controls; exactly 15 named methods C01-C15 below. It is outside ordinary offline test discovery; missing native prerequisites fail, never skip. `Controls.lean` imports the compiled bridge as module `Component`, defines namespaced tests and uses `#eval SDGA.runControls`; it defines no competing root main.

Fixture package under `tests/fixtures/sdga_component`: lean-toolchain exactly `leanprover/lean4:v4.34.1\n`, minimal lakefile.toml with no dependencies; five fixture files under Fixture, plus Controls.lean. Compile `tools/sdga/Component.lean` with source root `tools/sdga` to `<out>/Component.olean`; fixture source root is the fixture-package directory.

Each test method must preserve the relevant original native output and assert the native exit plus actual structured observations. Inventory completion (exit zero with complete JSONL) is distinct from blocking findings: owned axioms, unsafe/partial items and raw OLean extras remain recorded for Task 1 classification. Structural/loading/lookup/inference failures are nonzero without a successful end record; replay policy blockers are nonzero before kernel replay. No bare PASS-line or checker-failure heuristic establishes success.

---

## Task A: actual objects, lossless names and inventory judgment

**Files:** Create Component.lean request/load/wire/inventory portions; Fixture/Safe.lean; fixture package configs; Controls.lean C01/C02/C03/C12/C13/C15; explicit Python methods with the same IDs.

**Interfaces consumed:** Fixed Task 1 parse_inventory/classify_inventory, source/request rules above.
**Produces:** ModuleRequest/LoadedModule; exact Name codec; complete fixture JSONL accepted by Task 1 while classification remains BLOCKED.

- [ ] Write C01 assertions for anonymous round trip as a value but rejected identity; string versus numeric 12; empty string component; literal dot/slash; Unicode precomposed versus decomposed strings; non-BMP text; numeric component greater than 2^64; malformed/noncanonical lowercase-hex and decimal wires. Encode/decode must preserve exact Lean Name constructors.
- [ ] Write Fixture.Safe with module header/core imports, an explicitly public foreign-namespace declaration, explicit private theorem, unexported declaration, safe public True theorem, `def proof : True := True.intro`, `def propValue : Prop := True`, Nat-valued definition, simple inductive with two constructors/recursor and equation-style function. Include expected known declarations; generated-population evidence must come from actual object data, not guessed namespace counts.
- [ ] Write C02 asserting raw part-table equality/duplicates/subset rules, all final declaration rows, actual owner for foreign/private/generated names, and no dependency-owned declaration in the owned census. Assert the source-generated name is actually present before claiming generated coverage.
- [ ] Write C03 asserting actual Meta.isProp(proof.type)=true, Meta.isProp(propValue.type)=false and correct kind separation; malformed/open types block. Assert is_private from both explicit private spelling and absence from exported population.
- [ ] Write C12 altered-real-ModuleData controls for unequal parallel lengths, changed name identity, duplicate names, missing final declaration and earlier population absent from final private. Write C13 missing/reordered/truncated server/private part refusals and C15 wrong requested/resolved module path, unknown mode/fields and malformed JSONL identity controls.
- [ ] Publish the test-first candidate through root's ordinary isolated branch and execute the hosted targeted tests against the absent bridge. Retain the actual missing-file/import/compile failure as implementation absence only; do not call it a semantic negative or mutation kill.
- [ ] Implement the exact interfaces with pinned readModuleDataParts/private import/owner APIs. Use actual ConstantInfo variants, MetaM.toIO and Meta.isProp; emit fixed JSONL incrementally, with stderr diagnostics and nonzero errors.
- [ ] Re-run the Task A controls hosted. Require all named case bodies actually executed, complete stdout parsed by Task 1 and no skipped or missing cells. Retain before/after object hashes and original logs.
- [ ] Record the exact committed source, observed command/results and limitations, then commit the coherent Task A delivery for review.

## Task B: strict all-owned axiom traversal and skipped declarations

**Files:** Component.lean strictAxioms/safety additions; Fixture/BadDependency.lean; Fixture/AxiomUse.lean; Fixture/Skipped.lean; Fixture/OwnedAxiom.lean; C04-C07 controls.

**Consumes:** Task A's exact loaded ConstantInfo map and actual owner environment.
**Produces:** Actual per-declaration/union axiom observation with missing-fatal closure and blocking owned/safety findings.

- [ ] Write C04 with dependency `public axiom BadProof : True`, TypeAlias/ typeOnly/ valueOnly and opaque routes from the design. Assert actual compiled typeOnly.type retains TypeAlias and its value lacks BadProof before grading the type-only route. Require each actual closure contains the exact dependency axiom.
- [ ] Write C05 by altering an actual loaded ConstantInfo type and, separately, its value to reference one absent name. Call strictAxioms with the actual lookup and assert missingConstant with that exact wire name. Cover an absent inductive/recursor link too; do not substitute source that cannot compile.
- [ ] Write Fixture.OwnedAxiom as `module; prelude; public axiom propext : Prop`. C06 requires actual imports=[], actual CI name exactly propext, actual owned axiom kind and OWNED_AXIOM despite dependency-allowlist membership.
- [ ] Write genuine unsafe and partial definitions. C07 asserts the actual ConstantInfo flags, preserves their inventory rows and separately invokes the shared replay guard for each case, requiring unsafeReplaySkip or partialReplaySkip before upstream replay.
- [ ] Run the new controls against the prior implementation and retain their actual semantic failures or specific missing-helper failure, with no pass credit for absence.
- [ ] Implement strictAxioms using actual ConstantInfo.getUsedConstantsAsSet traversal, every lookup fatal on none, type plus theorem/opaque values and inductive/recursor links. Audit every owned constant and permit only the three exact dependency axiom Names. Do not reuse permissive built-in collectAxioms results as proof.
- [ ] Re-run all Task A/B controls hosted. Require per-route actual assertions, exact names, zero skipped case bodies and retained original failures before committing.

## Task C: same actual kernel replay route and duplicate-base guard

**Files:** Component.lean replayOwned/runReplay; C08/C09/C10 in Controls.lean and Python launcher.

**Consumes:** Complete final private map, module's actual direct imports and Task B safety/closure checks.
**Produces:** Source-bound native replay result or an explicit blocking failure through unchanged upstream Kernel.Environment.replay.

- [ ] Write C08 requiring actual replay of the Safe fixture's complete final private map into a fresh direct-import base, including actual inductive/constructor/recursor coverage. The target module and every owned name must be absent from that base.
- [ ] Write C09 retrieving the actual Safe True theorem ConstantInfo, altering only value to Nat.zero and preserving name/type/levels/group/safety. Assert all expression references resolve and target absence; submit a singleton map containing only that altered theorem to the same replayOwned helper used by production. Require actual upstream kernel type rejection, with the native kernel error retained. An earlier missing lookup/setup failure is inconclusive/failure, never the expected rejection.
- [ ] Write C10 constructing a contaminated base by successfully replaying the original good map first. Submit only the same singleton bad-theorem map through replayOwned and require duplicateBase before upstream replay. No unchecked environment insertion is used to set up contamination. Do not resubmit the full Safe map: unrelated duplicate definitions/inductives would confound the intended upstream theorem-duplicate observation.
- [ ] Observe RED against a missing/no-op replay implementation, preserving the actual failed good-control or wrong accepted result. Then implement base construction from actual direct imports at private visibility and the all-owned-name absence guard.
- [ ] Call unchanged `base.replay owned`; keep actual unsafe/partial blockers. Do not copy or simulate kernel replay logic and do not import/call LeanChecker's parallel prefix CLI.
- [ ] Hosted replay all A-C controls sequentially; assert exact helper path, actual success/rejection distinction and no skipped names. Commit only after source review of the fresh-base and malformed-CI evidence.

## Task D: exact OLean extras and separate IR evidence

**Files:** Component.lean extras inspection; controller artifact-sidecar handling; C11/C14.

**Consumes:** Raw ordered loaded OLean/IR data and Task A final private CI map.
**Produces:** Unchanged Task 1 OLean extras field plus separately bound complete raw IR provenance and explicit residue status.

- [ ] C11 alters actual loaded ModuleData.extraConstNames to contain one distinct name, then calls the same extras helper and requires blocking UNSUPPORTED_EXTRAS. Preserve raw per-OLean-part arrays and verify their union equals the JSONL field.
- [ ] C14 loads actual IR parts generated by the pinned Safe fixture compile, retains all raw IR names in the sidecar and requires final-CI-backed raw names to be accounted without false unsupported-extra findings.
- [ ] Add a labelled alteration of actual IR metadata with one name absent final CI; require that exact name retained as unexplained residue and qualification BLOCKED. This proves refusal, not source-generated-extra coverage.
- [ ] Run controls against absent/incorrect extras handling and retain RED. Implement raw OLean-extra blocking and IR sidecar recording/accounting without adding JSONL fields.
- [ ] Re-run all A-D controls, verify raw arrays/part hashes/sidecar identities and preserve unexplained cases. Commit the bounded result.

## Task E: hosted-only controller, sequential qualification and adverse mutations

**Files:** Complete tools/sdga_component.py, tests/sdga_component_contract.py and .github/workflows/sdga-component-contract.yml; additive design/TOOLING documentation and append-only DECISIONS at later integration.

**Consumes:** Tasks A-D helpers and the unchanged Task 1 contract.
**Produces:** A reproducible read-only native fixture job with exact case census, original artifacts and truthful limits. No production SDGA observation is executed by this task.

- [ ] Implement the 15 explicit Python unittest methods C01-C15. They invoke fixed source/fixture paths and shared helpers, measure real outcomes and fail on absent Lean or incomplete observations. No conditional enable flag, skip path or arbitrary caller command.
- [ ] Implement controller request/part/source guarding and fixed command vectors. Output roots must be new, external to every source root; reject nonempty/reused roots. Measure hashes before/after and never expose old owned/formal caches. Keep all dependency/source/request/part/log identities and every failed observation.
- [ ] Implement parse_dependency_headers using pinned deps-json: exact ordered file census, result present and errors=[] per row despite zero process exit. Add an actual malformed-header source adverse case plus duplicate/cycle/missing-local metadata tests under C15. Do not use regex imports or lakefile array order as topological evidence.
- [ ] Define fixed fixture compile order: BadDependency, Safe, AxiomUse, Skipped, OwnedAxiom. Compile bridge separately as module Component under source root tools/sdga; compile Controls only after its prerequisites. Await each process and stop its dependents after failure.
- [ ] Use only the real compile command below; make source roots/destination module paths exact and pre-create parents. For fixture runs dependency_path is empty except the fresh owned output and the toolchain's builtin core. LEAN_PATH replacement occurs after lake env.
- [ ] Add a read-only Ubuntu24.04 workflow. Use pinned checkout with persist-credentials=false, setup-python3.11, and lean-action24581d74d674803a25a81b10b059c5c0cdd344c6 with lake-package-directory=tests/fixtures/sdga_component, auto-config/build/test/lint=false, use-mathlib-cache=false, use-github-cache=false. Existing Math formal workflow b9165ec9ce50337087ea19cad148fe68eb15b037 supplies the setup precedent. Check actual lean --githash equals5045d0056413266e57c625dcd7c365b10e377c52 before tests.
- [ ] Run the fixture contract launcher normally and optimized, sequentially, into two fresh directories. Each mode must actually run exactly15 methods and all declared Lean subcases, with zero failures/errors/skips/cancelled/unknown outcomes for a passing contract result. A failed normal attempt does not suppress the optimized attempt unless cancelled; any failure keeps the job failed.
- [ ] Run unchanged Task 1 offline controls in both Python modes; verify the existing pinned-source workflow remains required and intact. Do not alter its count to accommodate this implementation.
- [ ] In disposable external source copies, exercise the eight fixed mutations M1-M8 below. Each selected control must fail its intended assertion after successful unmutated setup. Compilation/import errors, timeout/OOM, missing test bodies or unrelated failures do not qualify as kills. Match each source edit exactly once and retain mutant source digest, actual child exits, complete logs and assertion identity.
- [ ] Always upload original requests, complete fixture/object/IR trees, JSONL, sidecars, controller identities, stdout/stderr, case results, timing/resource records and mutation evidence. Bind artifact to exact candidate commit/run/attempt, including failures. Inspect artifact contents; a summary PASS line alone is insufficient.
- [ ] Fresh source-exposed review checks the complete candidate and source evidence. Root assesses integration only after the actual hosted results and required protected checks. Classification/qualification remain BLOCKED even if all fixture contracts pass.

Exact compile command, run from the fixture package (absolute source/tool paths supplied):

```sh
lake env env LEAN_PATH="$SDGA_OUT:$SDGA_DEP_LEAN_PATH" \
  lean --threads=2 --root="$SDGA_SRCROOT" \
  -o "$SDGA_OUT/$MODULE_PATH.olean" -i "$SDGA_OUT/$MODULE_PATH.ilean" \
  "$SDGA_PKG/$SOURCE_REL"
```

Observation-mode bridge commands run from Main (or use its absolute source path when cwd is the fixture package). Exact fixture contract commands, run from Main checkout:

```sh
python3 -B -S tests/sdga_component_contract.py --output "$RUNNER_TEMP/sdga-task2-normal" -v
python3 -B -O -S tests/sdga_component_contract.py --output "$RUNNER_TEMP/sdga-task2-optimized" -v
```

These are planned commands, not observed results.

| Mutant | Deliberate defect | Selected intended failure |
|---|---|---|
| M1 | Replace custom Name encoding with dotted/display identity | C01 string/numeric/escaped component identity mismatch |
| M2 | Inventory exported population only | C02 missing actual private/generated owned rows |
| M3 | Missing closure lookup returns empty closure | C05 missingConstant assertion |
| M4 | Traverse value only, omit declaration type | C04 actual type-only dependency absent |
| M5 | Replace actual replay with a successful no-op | C09 malformed theorem accepted despite successful good setup |
| M6 | Ignore unsafe/partial skip blockers | C07 all-owned coverage improperly accepted |
| M7 | Drop unexplained IR names from sidecar/status | C14 explicit residue retention/blocking assertion |
| M8 | Remove owned-name absence guard on replay base | C10 duplicateBase assertion (upstream duplicate tolerance is preserved) |

## Production observation handoff after fixture review

Do not execute this section in the fixture milestone. Root separately authorizes a source-bound run of all99 modules after the bridge/controller/native fixtures pass and exact source review is complete.

- Recheck the frozen historicalMathf532 source/config99hashes/profile, exact nine dependency revisions, resolved object paths and Lean release before/after. Identical9bdf package subtree is context, not an execution substitute.
- Obtain all99 headers via actual pinned `lake env lean --deps-json`; require every row/error guard, then derive a complete fixed topological order.
- Compile all99 owned sources fresh, with roots package/inputs/inputs/elder and output module paths, one process at a time. Reuse only verified external dependency objects; formal/.lake/build and other owned caches never enter LEAN_PATH.
- Inventory/replay one exact owned module per fresh process; bind raw all-part objects and separate IR sidecar. Reconcile every actual owned declaration and all1137literal targets; report actual Prop/data discrepancy, unsafe/partial/extras/IR-residue or closure/replay failure without weakening expectations.
- Preserve exact failures, resource limits and partial coverage. Report direct Lean elaboration rather than literal Lake build/full facets. No fullG3 or finalCBZ credit follows.

## Fixed component subcase census

The fixture results artifact must contain exactly these 67 unique `case/subcase` IDs per Python mode, each exactly once, with its actual assertion outcome, executor (Lean or Python controller) and source/native-evidence binding. C15 graph/transcript parsing subcases execute in the controller; real import/compile/replay/CI cases execute in Lean. Preserve the actual executor instead of calling a pure parser check native execution. Require the complete set and all successful expected-observation assertions for a passing control suite. Missing, duplicate, unexpected, skipped or unexecuted IDs fail. Count actual subordinate assertions from this artifact as well as exactly 15 Python method outcomes; do not infer subordinate execution from a method PASS or summary. Negative controls pass only when their specific expected cause was actually observed.

| Case | Required subcase IDs |
|---|---|
| C01 | `anon-roundtrip`, `anon-identity-reject`, `numeric-vs-string`, `empty-component`, `dot-component`, `slash-component`, `unicode-normalization-distinct`, `nonbmp`, `large-nat`, `uppercase-hex-reject`, `leading-zero-reject`, `invalid-utf8-reject` |
| C02 | `raw-tables`, `foreign-owner`, `explicit-private`, `module-private`, `generated-present`, `dependency-excluded`, `final-rows-owner-census` |
| C03 | `proof-def-prop`, `prop-valued-data`, `nat-data`, `malformed-type-block` |
| C04 | `type-route`, `value-route`, `opaque-route` |
| C05 | `missing-type`, `missing-value`, `missing-link` |
| C06 | `imports-empty`, `exact-owned-propext`, `owned-axiom-block` |
| C07 | `unsafe-info`, `partial-info`, `unsafe-replay-skip`, `partial-replay-skip` |
| C08 | `clean-base-absence`, `full-safe-replay` |
| C09 | `actual-source-ci`, `singleton-resolved-refs`, `clean-base-absence`, `kernel-type-rejection` |
| C10 | `good-base-setup`, `singleton-resolved-refs`, `duplicate-base-guard` |
| C11 | `raw-union-preserved`, `raw-extra-block` |
| C12 | `table-length`, `table-identity`, `duplicate-name`, `missing-final-row`, `earlier-subset` |
| C13 | `missing-server`, `reordered-parts`, `truncated-private` |
| C14 | `actual-ir-binding`, `raw-names-preserved`, `ci-backed-accounted`, `residue-retained-blocked` |
| C15 | `wrong-resolved-path`, `unknown-mode`, `unknown-field`, `malformed-transcript`, `deps-json-error-row`, `duplicate-vertex`, `missing-local-edge`, `cycle` |

## Self-review and root handoff

- Spec coverage: A covers source-bound objects/identity/judgment; B closure/axioms/safety; C real kernel route/clean base; D extras/IR; E hosted scheduling/provenance/adverse controls. Frozen production profile and original failures remain outside modification scope.
- Interface consistency: later tasks consume the named ModuleRequest/LoadedModule/strictAxioms/replayOwned helpers; Controls imports module Component and defines no second root main.
- Review-focus cases each have explicit owning controls above. No generic AST serializer or orchestration API is introduced.
- Root reviews design/plan blobs and exact field semantics before code. Until actual commands run, every execution/result described here is planned only.
