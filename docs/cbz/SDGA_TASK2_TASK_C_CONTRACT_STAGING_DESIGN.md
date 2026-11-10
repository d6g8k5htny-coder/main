# Task C source-only integration onto qualified Task B

Status: STAGED_SOURCE_ONLY; ready for root integration source review. No publication, ref, PR, workflow, native execution or production implementation is performed.

Exact conceptual integration base: B805 commit 81fcda5f2f36309011ae7f021bb8d509f76fa282, tree fc977ff1f694f6bb585d460f9bc283572daa5bd5. Source pins were independently read as Git blobs and compared byte-for-byte with the same paths fetched at that exact commit:
- tests/sdga_component_contract.py: 54421c70918fdebf6b676cd8cfd49cde0c9d67b9.
- tests/fixtures/sdga_component/Controls.lean: 42a80b09a78ab7777469d8966c6e46ac9bc748f1.
- tools/sdga/Component.lean: 7f89460edca71754dbce9a07de229ad39ad3e851.

Parent-provided independently reviewed B GREEN provenance: run 42757239189, attempt 1, job 129634336145; both modes 10 methods/52 subcases, pure protocol 12, separate custody 3 methods/6 routes, 21 observations/10 request bindings/55 paths PASS with zero drift. This integration staging does not rerun or rebind that historical evidence.

The parent supplied moving-head context A8dad/Main2f after B4d. Fresh read-only API observation during this task found A797 head 8dad51ed3f34ee3db90867da34b2807d6b87fb1b, A797 base 3f05f8493afac71c6f4aac5c1db5aeb46194d98a, B805 head unchanged at 81fcda5f2f36309011ae7f021bb8d509f76fa282 and B805 base/Main b6be588c82d2256c917b7d26d901085bc90319fb. These are moving-head facts only. This artifact remains tied to exact qualified B81fc, not to moving Main or A heads; future publication requires fresh parent/head guards and review.

## Reviewed C contract retained exactly

Binding brief c0d871f3dfafb885eb43d71d860e0fc4177a1e86; prior reviewed design/report 9a4b8a3944815c9c866c8fb2833bfeded920ced0; prior C-only transfer 3eb9c89c9b1ac7fc937860da524b54dd14a82823.

The complete Controls result is byte-for-byte the root-reviewed C snapshot c25cd01838c152403eba1acd39fb029a488bb21f. All three reviewed Python test method bodies and their exact assertions are unchanged from the prior C transfer. All pinned Lean API choices, native diagnostic comparison clauses, singleton bad theorem construction, actual full good replay, duplicate-base causality, typed exact SDGA.replayOwned calls, safety/no-skip assertions and independent fresh direct-import oracle remain unchanged. No new production base helper, mock, semantic fake or native-failure pass is introduced.

C08 has two Lean subcases (clean-base-absence, full-safe-replay); C09 has four (actual-source-ci, singleton-resolved-refs, clean-base-absence, kernel-type-rejection); C10 has three (good-base-setup, singleton-resolved-refs, duplicate-base-guard). Total 13 methods/61 subcases =55 Lean +6 Python_controller. Original A6/39 and B4/13 retained; pure12 and separate custody3/6 remain outside the 61.

Current actual production 7f89460edca71754dbce9a07de229ad39ad3e851 is untouched: guardReplayConstant rejects actual unsafe/partial CI, replayOwned shares that guard and then throws NOT_IMPLEMENTED, and runReplay remains NOT_IMPLEMENTED. C08/C10 must preserve that actual failed good-replay result; C09 must treat NOT_IMPLEMENTED as wrong-cause failure. No compiled or executed C outcome is claimed.

## Custody repair preserved

The older combined launcher 14ee84dee1806b3b14ae6a787fde3ad3b91894ef was never used as an integration base or replacement. Only uniquely anchored C-owned edits from the reviewed transfer were applied to the complete actual repaired 544 source; the first heading anchor was adapted to its current B wording. All other original C insertions/assertions are byte-exact.

Source-text comparisons establish:
- Reversing the exact integrated transfer edits recovers the complete 54421c70918fdebf6b676cd8cfd49cde0c9d67b9 source byte-for-byte.
- All ten original A/B test method bodies are byte-exact. All prior helper method bodies are byte-exact except capture_sources/flush census metadata and finalize_objects' two fixed expected-observation literals.
- Top-level measure_objects and object_custody, bind_request, observe_lean, native_case, accept_rows, cleanup_group/command/cleanup preflight, request_for, complete bootstrap including B dependency binding, task_b_inventory/task_b_native/reconciliation, original final object registry/drift/errors/cleanup checks and all B fixture/source binding capture remain intact.
- finalize_objects changes ONLY expected_observation_count:21→24 and len(self.object_observations)==21→24. Expected_request_count=10 and expected_path_count=55 remain fixed and unchanged. No expectation is derived from the observed count.
- Main result/census labels and 10→13/52→61 acceptance counts receive only reviewed C census updates; objects_complete, object_custody_failures and all cleanup/drift blockers remain in place.
- No fixture, source-history assertion, object-custody regression file, Task1 protocol/classifier/assertion, frozen profile, historical proof, source bound, formal gate or production file is modified.

No syntax check, Python execution, Lean elaboration or kernel run was used for these source comparisons.

## Independent fixed observation enumeration: 24/10/55

The fixed complete-success schedule follows unittest's default lexical test method order. These are static expected call sites, not observed-derived expectations. A failure can leave later steps or rows not_run; missing/partial/unsupported observations block rather than lower the fixed census.

01. inventory: Safe
02. control-C01: Safe
03. control-C02: Safe
04. control-C03: Safe
05. task-b-AxiomUse-inventory: AxiomUse + BadDependency
06. control-C04: AxiomUse + BadDependency
07. control-C05: Safe
08. task-b-OwnedAxiom-inventory: OwnedAxiom
09. control-C06: OwnedAxiom
10. task-b-Skipped-inventory: Skipped
11. control-C07-unsafe-info: Skipped
12. control-C07-partial-info: Skipped
13. control-C07-unsafe-replay-skip: Skipped
14. control-C07-partial-replay-skip: Skipped
15. control-C08: Safe
16. control-C09: Safe
17. control-C10: Safe
18. control-C12: Safe
19. control-C13-missing-server: Safe + missing-server
20. control-C13-reordered-parts: Safe + reordered-parts
21. control-C13-truncated-private: Safe + truncated-private
22. control-C15-wrong-resolved-path: Safe + wrong-path
23. cli-unknown-mode: Safe
24. cli-unknown-field: Safe + unknown-field

The 21 existing B observations are retained and exactly three are added: control-C08, control-C09, control-C10. Each calls existing native_case with only its fixed case id; native_case already wraps STATE.observe_lean with requests=(STATE.request_path,). They reuse the original Safe request and its five object-part paths. None writes/binds a new request, creates a new object copy or changes B request/path source bindings.

Ten fixed request bindings remain:
1. Safe: request.json; 5 newly registered object paths and 1 request path.
2. BadDependency: task-b-BadDependency-request.json; 5 newly registered object paths and 1 request path.
3. AxiomUse: task-b-AxiomUse-request.json; 5 newly registered object paths and 1 request path.
4. OwnedAxiom: task-b-OwnedAxiom-request.json; 5 newly registered object paths and 1 request path.
5. Skipped: task-b-Skipped-request.json; 5 newly registered object paths and 1 request path.
6. missing-server: missing-server-request.json; 5 newly registered object paths and 1 request path.
7. reordered-parts: reordered-parts-request.json; 5 newly registered object paths and 1 request path.
8. truncated-private: truncated-private-request.json; 5 newly registered object paths and 1 request path.
9. wrong-path: wrong-path-request.json; 5 newly registered object paths and 1 request path.
10. unknown-field: unknown-field.json; 0 newly registered object paths and 1 request path.

Path derivation: five genuine fixture populations (Safe, BadDependency, AxiomUse, OwnedAxiom, Skipped) each have five object parts plus one request=30; three C13 adverse-copy roots each register five part path names plus one request=18; C15 wrong-path root has five object parts plus one request=6; unknown-field adds one request and reuses Safe parts=1. Total 30+18+6+1=55 paths, of which ten are request files. The intentionally missing server path stays in measured expected registry with exists=false. BadDependency has a binding only, zero new inventory/observation/subcase credit, and is included in both actual AxiomUse observation custody sets.

## Later publication and native RED

Root owns fresh source review, source binding guards, stacked C test-first publication after B final review, and hosted execution. The intended change paths remain Controls.lean, the existing launcher, and this C design/report; no production implementation is staged. The transfer artifact contains exact unique-anchor changes against qualified B544 and full fixed call/request enumeration, and must be reviewed before application to any newer base.

Use the unchanged launcher entry point after authorized publication, on the pinned toolchain and a fresh output directory, with all processes sequential:
python3 tests/sdga_component_contract.py --output "$RUNNER_TEMP/sdga-task-c-red" -v

All A/B assertions must remain active. Expected RED against present safe replay stub is actual failed good control/wrong-cause rejection with original stdout/stderr and before/after object-custody evidence. C controls must first compile; setup/absence/API/diagnostic drift cannot receive kernel rejection credit. Later production runReplay direct-import base construction and exact shared replayOwned guard/upstream path still require separate source-equivalence review.

scientific_effect=NONE; qualification/classification_status=BLOCKED; execution_authenticated=false. No full G3, whole IR/native/kernel coverage, final G0-G11, CBZ or scientific acceptance claim is made.
