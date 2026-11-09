# SDGA Task 1 Scope and Classifiers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Extract and validate the frozen SDGA source scope and strictly classify declared inventory/receipt fields while every qualification remains BLOCKED.

**Architecture:** A small Python 3.11 standard-library module has no process, Git, network, Lean, filesystem-write or retry capability. Pure functions consume caller-provided bytes/dictionaries. Original-source verification and offline projection tests remain distinguishable.

**Tech Stack:** Python 3.11+, unittest, hashlib, json and tomllib.

**Spec:** docs/cbz/SDGA_COMPONENT_DESIGN.md

## Global Constraints

- Math source: f532c3e2d91a0c7679a37c733741660917a9b21b; tree b713ebf4d444f0f0a304e50302b1fe616155201a.
- Exactly 99 module/path/hash/role bindings: 67 production, 25 ordinary inputs, seven elder inputs.
- Preserve the literal ordered 1,137-target vector, including Unicode. Its UTF-8 newline-terminated vector SHA256 is 2c22d5ba939ed6cf74669a99a23ad8337ec9a54abcafb3bf43ef5799e5536cae.
- scientific_effect=NONE, qualification=BLOCKED and classification_status=BLOCKED for all Task 1 outputs.
- No type AST serializer, Lean execution, proof changes, role admission, artifact authentication, generic orchestrator or external writes.
- Reject bool where an integer is required. Do not normalize Unicode or collapse numeric/string Name components.
- Initial missing-module import is valid absence evidence only. It provides no adverse-rejection or mutation credit.
- The dedicated read-only component workflow always checks out pinned Math and executes all 22 offline methods plus the one explicit pinned-source method in both normal and optimized Python. No optional coverage, skipTest, conditional disable or selective green matrix.
- Do not add a placeholder production scaffold. Root publishes the immutable test-first draft, captures hosted RED, then authorizes implementation.

## Review Focus

1. The real manifest exceeds 2 MiB; the pinned-source test must read and hash the original successfully.
2. Two distinct Unicode spellings and quoted-dot/numeric Name components must remain distinct.
3. Exported/server/private cumulative parts may repeat a name across parts; duplicates within a part or mismatched parallel tables must fail.
4. Empty stdout/stderr may be valid; bind their measured hashes rather than requiring nonempty logs.
5. Expected receipt fields must be supplied independently in later execution; Task 1 reports declared-fields validity only.

## Task 1: Pure scope and fail-closed classifiers

**Files**
- Create later: tools/sdga_scope.py
- Test first: tests/test_sdga_scope.py (22 pure API methods)
- Test first: tests/sdga_source_observation.py (one mandatory explicit pinned-source method; outside test_*.py discovery)
- Frozen data: tests/fixtures/sdga_scope/{scope-f532.json,lakefile.toml,lake-manifest.json,lean-toolchain,source-scope-projection.json}

**Interfaces**
- ScopeError(ValueError), NameProtocolError(ValueError), InventoryProtocolError(ValueError); each exposes stable string attribute code.
- extract_scope(raw_inputs: dict[str, bytes], profile: dict, observed_identity: dict) -> dict
- verify_source_files(scope: dict, source_files: dict[str, bytes]) -> None
- encode_name(name: tuple[str | int, ...]) -> str
- decode_name(encoded: str) -> tuple[str | int, ...]
- parse_inventory(text: str) -> dict
- classify_inventory(inventory: dict, scope: dict) -> dict
- classify_receipt(receipt: dict, expected: dict) -> dict

extract_scope validates exact four input keys and each raw length/SHA256/Git-blob binding from profile.inputs before parsing, then exact source identity against profile.source, pinned toolchain, dependency lock, package name, default production roots, the 25+7 input roots, source-module reconciliation, unique target strings and selected source hashes. It reconciles exact module/path/hash/role rows, dependency pins, target vector and counts with the profile; it returns modules [{module,path,role,source_sha256}], targets, source, scientific_effect, qualification and classification_status. Reject duplicate JSON keys, absolute/escaping source paths and duplicate/colliding module identities. Frozen profile comparisons live in tests and the later fixed adapter; accepting a caller profile is not provenance authentication. Scope error codes are INPUT_BINDING, SOURCE_IDENTITY, MODULE_SCOPE, TARGET_SCOPE, DEPENDENCY_SCOPE, TOOLCHAIN_SCOPE, SOURCE_BINDING, PATH_SCOPE and JSON_PROTOCOL.

verify_source_files requires exactly the scope's selected relative file set and compares every raw source SHA256. It performs no path reads itself.

Names use the existing lossless component convention: anonymous is empty, string components are s followed by lowercase UTF-8 hex, numeric components are n followed by canonical nonnegative decimal; components separated by /. Preserve empty string components and Unicode code points exactly. Declaration/module identities must be nonanonymous even though the codec supports anonymous.

Inventory is strict JSONL, one header first and one end last, no noise/blank records/trailing records/duplicate JSON keys or nonfinite numbers. Schema 1 header has scope_id. Module rows have name, const_count, object_parts and extra_const_names. Each part has level (exported/server/private), sha256, const_names and constant_names; order is increasing visibility, exported first, unique levels, equal parallel vectors with unique names, earlier names contained in the last part. Declaration rows have module,name,kind,is_prop,is_private,unsafe,partial,type_display,object_part_sha256,axioms. Kinds: theorem,definition,opaque,axiom,inductive,constructor,recursor,quotient. Type display is explanatory only. All booleans must be actual booleans. Declaration rows exactly cover each module's last-part names and count; object binding must match its last part. End has module_count,declaration_count,axioms. End axiom vector is the exact union of row vectors. Return normalized scope_id/modules/declarations/axioms without dropping unknown constants. Reject unknown protocol fields; permit only a final newline, not blank records.

classify_inventory checks scope ID/exact modules and targets, production ownership of targets, observed Prop/data counts, owned axioms, forbidden axioms, unsupported extras and unsafe/partial replay skips. Findings are dictionaries with code and name; names are lossless encoded where applicable. No namespace-prefix filtering. Return counts (owned_declarations,public_targets,proposition_targets,data_targets), findings and fixed BLOCKED/NONE fields. Unresolved blockers are EXACT_TYPE_SEMANTICS_UNRESOLVED, FULL_DECLARATION_ROLES_UNRESOLVED, EXECUTION_NOT_OBSERVED and KERNEL_COVERAGE_UNESTABLISHED, even for valid fixtures. Finding codes tested are MODULE_SCOPE, TARGET_SCOPE, TARGET_OWNER, TARGET_KIND_COUNTS, UNSUPPORTED_EXTRAS, UNSAFE_REPLAY_SKIP, PARTIAL_REPLAY_SKIP, FORBIDDEN_AXIOM and OWNED_AXIOM.

classify_receipt checks exact source/tool/profile/cache/object/log/run/attempt/module bindings and required step coverage. Positive result binding_status=DECLARED_FIELDS_VALID; adverse result binding_status=INVALID with stable findings. Require signal and timed_out fields explicitly; zero integer exit, null signal, false timeout and nonnegative numeric resource/time fields for every required step. Reject bool exit, source/cache drift, missing/extra/duplicate steps, missing object/module coverage and unsafe/partial skips. It returns execution_authenticated=false and never qualification PASS. For a single defective fixture return exactly its designated code. Deterministic validation priority is RECEIPT_SCHEMA, STEP_COVERAGE, STEP_FAILED, SOURCE_DRIFT, CACHE_DRIFT, MODULE_COVERAGE, REPLAY_SKIPPED, RECEIPT_BINDING. Names/notes may identify the defective field. Use the same four fixed unresolved blockers as the inventory classifier.

- [ ] Step 1: Publish the 22 pure API methods, explicit pinned-source control and immutable fixtures without tools/sdga_scope.py. Keep tests/sdga_source_observation.py outside ordinary test_*.py discovery.
- [ ] Step 2: On the hosted draft always check out pinned Math, guard native commit/tree and set SDGA_PINNED_MATH_ROOT, SDGA_OBSERVED_MATH_COMMIT and SDGA_OBSERVED_MATH_TREE. Execute all four mandatory commands below as independent matrix cells; preserve complete output and exact head/attempt. ModuleNotFoundError for sdga_scope is the expected initial absence observation, not adverse-control credit.

    python3 -m unittest discover -s tests -p 'test_sdga_scope.py' -v
    python3 tests/sdga_source_observation.py -v
    python3 -O -m unittest discover -s tests -p 'test_sdga_scope.py' -v
    python3 -O tests/sdga_source_observation.py -v
- [ ] Step 3: After root authorization, implement only the interfaces above using standard-library parsing and explicit checks, never Python assert for validation.
- [ ] Step 4: Run all four mandatory commands again. Exactly 22 offline methods and one explicit pinned-source method must pass in each mode, with no skips. No production fixture or expected hash may be rewritten to accommodate an observed result.
- [ ] Step 5: Review the explicit pinned-source results from both modes. Required environment checks must fail if unavailable; the original assertions still independently hash all four raw inputs and all 99 source files before comparing extracted scope. Preserve raw output and native source bindings. Every component workflow run must include these results.
- [ ] Step 6: Review the exact diff and evidence. Do not claim Lean execution, real axiom closure, replay coverage or mutation kills from Task 1. Root decides integration; this plan authorizes no commit/ref/PR action.

Later milestones need genuine tiny Lean positive/adverse controls, private-level all-part collection, strict missing-fatal axiom traversal and exact named replay before their own admission. Those are deliberately outside this Task 1.

## Test organization rationale

The original pinned-source method moved intact to tests/sdga_source_observation.py except that its optional skip became required environment assertions. It imports shared ContractCase/fixtures/API helpers from test_sdga_scope. Ordinary discovery therefore adds 22 pure contracts and no dormant skip. The dedicated component matrix retains all 23 controls in each mode and cannot claim complete Task 1 coverage from the offline subset alone.

## Targeted receipt regressions before repair

The original 20 pure methods and pinned-source assertions remain unchanged. Add two pure methods, making 22 pure methods plus one mandatory pinned-source method per mode (46 method records across normal and optimized Python).

- test_date_only_basic_or_incomplete_utc_rejected_when_bindings_match: date-only 2026-11-09Z, basic 20261109T000000Z, hour-only 2026-11-09T00Z and minute-only 2026-11-09T00:00Z in both receipt/expected must produce exactly RECEIPT_SCHEMA. Step timestamps require full extended UTC grammar YYYY-MM-DDTHH:MM:SS[.digits]Z and timezone-aware UTC parsing.
- test_empty_module_coverage_rejected_when_both_bindings_match: both arguments have empty module_coverage and only build/inventory steps; the result must be INVALID with exactly MODULE_COVERAGE. This fixed nonempty SDGA component scope does not admit empty coverage.

Stage and execute these tests first against unchanged production blob d00a866e55c5d20fd4aeb8d44e51fa471f1d1031. No production repair is included in this revision. Root must observe the actual semantic failures before authorizing minimal UTC and nonempty-coverage checks. All result assertions retain BLOCKED/NONE; these are declared-field rejection controls, not Lean or scientific qualification.
