# SDGA fresh-owned component observer

Status: design only. Task 1 supplies source-scope and protocol contracts, not Lean execution or qualification.

The source is Math commit f532c3e2d91a0c7679a37c733741660917a9b21b, tree b713ebf4d444f0f0a304e50302b1fe616155201a, package frontiers/source_derivative_gaussian_20261025. Scope is exactly 67 production modules, 25 ordinary inputs and seven elder inputs. The 1,137 literal written public targets are a separate population from all compiled declarations. The documented target split is 1,128 Prop and nine data targets; actual compiled observations must reconcile it.

## Boundaries

- scientific_effect is always NONE. qualification and classification_status remain BLOCKED in this milestone.
- No full G2–G4, final CBZ, proof-closure, parent-application or scientific-status claim.
- Historical manifest status/catalog prose is not admission authority. The projection fixture contains only source_modules, targets, inherited_input_count and the selected 99 source hashes.
- Exact type semantics and full declaration roles remain unresolved. Emit type displays and bind original compiled object parts; do not call displays a lossless or canonical type representation.
- Task 1 validates declared receipt fields only. Matching caller assertions cannot authenticate execution. Later execution requires independently measured source/tool/object/log identities.
- No default PASS path, arbitrary configured command, implicit retry, source modification or whole-federation orchestration.

## Minimal later observer

A Python adapter consumes the frozen scope profile. One Lean collector imports the exact 99 modules at private visibility and enumerates actual module ownership, including foreign namespaces, private declarations, generated declarations, definitions, data, inductives, constructors and recursors. It loads and binds all exported/server/private object parts. Each part must reconcile constNames with ConstantInfo names in length, identity and duplicates; the most private part must contain earlier parts. Reconcile all final declaration rows and per-module counts. Enumerate extraConstNames or block explicitly.

Use a strict transitive axiom traversal following Lean's ConstantInfo cases, with missing lookups fatal. Reject every owned axiom declaration even if its name resembles an approved axiom. Audit all owned constants and the literal public targets against only propext, Classical.choice and Quot.sound. Report observed Prop/data using Lean's type judgment, not ConstantInfo kind. All unreviewed roles and exact type semantics remain BLOCKED.

Start from fresh owned outputs. Reuse only verified external dependencies from the pinned Math formal package cache. Verify nine revisions, their resolved dependency/object-part identities and toolchain before/after; never expose formal/.lake/build or another package's owned objects. In particular, ResearchFormalCoreR1 input names must resolve to the fresh SDGA build. Bind source, cache, resolved paths and artifacts before/after.

Reuse Main cbz/contributions/math514/kernel_replay/ExactReplay.lean, blob 400f3853d50df7ea2a9b4bb2fc1de302626f2efa, observed at f1ef3a2. Its interface is CBZ_KERNEL_REPLAY_MODULE plus CBZ_KERNEL_REPLAY_OLEAN; invoke as a Lean file, not lean --run, because imported LeanChecker defines main. Run one exact module per process, sequentially. It uses upstream replayFromImports; unsafe/partial constants are skipped upstream and must be reported as uncovered and block all-owned kernel qualification.

Pinned Lean 4.34.1 is release commit 5045d0056413266e57c625dcd7c365b10e377c52. Existing formal-lean.yml blob b9165ec9ce50337087ea19cad148fe68eb15b037 supplies accepted read-only setup using lean-action 24581d74d674803a25a81b10b059c5c0cdd344c6, build/test/lint/auto-config false, mathlib cache true and GitHub cache false. Use two Lean threads and one kernel process at a time. LEAN_NUM_THREADS=2 does not cap Lake's parallel processes, and this Lake has no assumed -j option; build scheduling/resource controls require separate review before execution.

Preserve actual commands, UTC/monotonic times, exits/signals/timeouts, peak resource observations, stream hashes, object-part hashes and native run/attempt identity on success and failure. Missing, stale, skipped, timed-out, cancelled, drifted or unsupported stages cannot receive pass credit. No execution is part of Task 1.

## Task 1 files

- tools/sdga_scope.py: future standard-library scope extraction, Name/protocol validation and blocked classifiers. Deliberately absent from the initial RED draft.
- tests/test_sdga_scope.py: 20 pure API contracts discovered by ordinary Main test discovery; initial ImportError is absence evidence, not an adverse-control kill.
- tests/sdga_source_observation.py: one mandatory pinned-source control, invoked explicitly; missing required source environment fails.
- tests/fixtures/sdga_scope/: frozen profile, three small original configuration inputs, labelled source-scope projection.
- docs/superpowers/plans/2026-11-09-sdga-task1-scope-classifiers.md: executable test-first handoff.

Original source bytes and MCP provenance are preserved outside the public staging tree in scratch. The full original manifest exceeds 2 MiB; a 2 MiB input limit would reject the actual source.

## Mandatory control organization

The pinned-source control is outside ordinary test_*.py discovery so ordinary Main runs do not acquire a dormant skip. Coverage is preserved: the dedicated read-only component workflow always checks out pinned Math, guards its exact commit/tree, supplies all three required environment values and runs all 21 controls in each Python mode. Missing environment values fail; there is no optional enable flag or skip.

Required commands are:

    python3 -m unittest discover -s tests -p 'test_sdga_scope.py' -v
    python3 tests/sdga_source_observation.py -v
    python3 -O -m unittest discover -s tests -p 'test_sdga_scope.py' -v
    python3 -O tests/sdga_source_observation.py -v

All four workflow matrix cells are mandatory: 20 offline contracts plus one pinned-source control in normal mode, and the same 21 in optimized mode. Ordinary discovery runs the 20 pure methods only. This organization change removes conditional coverage; it does not drop the original input/source hash assertions or confer execution qualification.
