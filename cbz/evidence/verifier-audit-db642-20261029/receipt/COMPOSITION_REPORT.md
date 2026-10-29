# Bounded formal source composition for main#684

Completed local source composition at instrumented UTC `2026-10-29T17:50:17.805414Z`. This is implementation and Python/source verification evidence, not independent engineering/scientific review, fresh Lean/kernel evidence, hosted qualification, or a CBZ-0 declaration.

## Exact source identities

| Role | Identity |
|---|---|
| Worktree | `/workspace/scratch/db6428847655/cbz/worktrees/main-composition` |
| Branch | `codex/cbz-formal-composition-db642-20261029` |
| First native parent, receipt PR692 | `5aa6fbac2c66fac0e6e2332b7e95cacc3c37350e` |
| First parent tree | `7f6013ee615bef4512521acf2c097bccb5e92e07` |
| Second native parent, negative-diagnostic PR689 | `4da34569f2e6e70267d2ee9d3170916b9b598279` |
| Second parent tree | `cb799942fcdf2440f8cce261ccdecd9b20e349aa` |
| Actual Git merge-base | `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff` |
| Governance ancestor of receipt input | `81acc060341e8a31cc4cd4f9523ef71fb6c06175` |
| Local merge commit | `da7396732233dbd66745c13b160a9a104f91341e` |
| Final / verified source tree | `507961762a1ec7c816458b712d09e27f3153b81c` |
| Combined formal manifest SHA-256 | `1c7aebd2e289cee0d9f82e15052d355efaaef642ff0cb5c47332435adfea75ea` |

The original brief incorrectly inferred that both native inputs shared governance base81acc060. Direct native `git merge-base` establishes4ef1867175dbe1072a54a02e95f80c4e2c70a4ff. PR689 is a direct child of that older source; governance81acc060 is retained through PR692. The parent supplied an explicit correction before mutation, and this ancestry correction is appended in the contribution ledger. Root AGENTS/STATUS/GOVERNANCE/DECISIONS bytes are exact to5aa6; absent root PROOF_INDEX/TOOLING/BASELINE files were not created. No shared canonical file was edited.

Both PR descriptions in `PR689_DESCRIPTION.md` and `PR692_DESCRIPTION.md`, the exact receipt design/plan, native source diffs and prior review reports were read before implementation. Input descriptions retain their authors' earlier execution claims; those claims are not adopted as execution of this composition. Native input diffs are retained as `composition-input-pr689.diff` and `composition-input-pr692.diff`.

Only exact frozen PR689 commit4da34569 is included. Subsequent PR689 changes, including any compiled-inventory extension, are **not included**. No remote read/write, foreign branch change, subagent, ref cleanup or host-control bypass was performed.

## Merge and conflict resolutions

`git merge --no-commit --no-ff 4da34569f2e6e70267d2ee9d3170916b9b598279` produced the four predicted conflicts. Original conflict output and unresolved combined diff remain in `composition-merge-conflict.log` and `composition-unresolved.diff`.

| Conflicted path | Resolution |
|---|---|
| `formal/manifest.json` | Retain PR689's full seven-control diagnostic contracts, mandatory diagnostic-test binding and README binding. Recompute only the merged gate and gate-test SHA-256 values. Every non-file manifest field and every other file hash is byte/value-identical to frozen PR689. |
| `tools/atlas_formal_evidence_build.py` | Retain PR692's exact immutable `bind_historical_receipt`; add PR689's unchanged `historical_source_identity`. Import current `safe_path` and `strict_json`; historical reconstruction uses the fixed reader and then the fixed receipt/archive validator. The live CLI remains strict and has no historical route. |
| `tests/test_atlas_formal_evidence_build.py` | Union all existing PR692 receipt/archive tests and PR689 historical source-identity tests, with their original bodies unchanged. |
| `tests/test_atlas_formal_source_bridge.py` | Retain PR692's copy-time frozen-control selection, which implements the same pinned-byte custody as PR689's subsequent overwrite. It preserves both sets of SHA assertions and all test bodies; the resulting file equals PR692 exactly. |

The producer and gate-test changes auto-merged. Against frozen PR689, the producer gains only the actual `GITHUB_RUN_ATTEMPT` field; the test file gains only the `os` import and the two original attempt/local-identity tests. PR692's strict binder and positive fixture attempt values remain unchanged. No missing attempt defaults to1 or receives hosted authority from the consumer.

PR689's arithmetic controls require exit1, complete expected stdout including path/location/proposition, and empty stderr. Axiom controls require exit0, exact complete reports, empty stderr, and actual `ForbiddenAxiomError` rejection by the auditor; a permissive auditor mutation must fail. The full prior attempt directory moves intact before executable preflight, including missing-tool/version failures; current success cannot survive a failed rerun. All these test controls were retained and run in both Python modes.

## Final changed files

Thirteen paths differ from native receipt parent5aa6. Files outside this table are unchanged from that parent. Exact sizes, SHA-256 and Git blobs are also in `composition-candidate.json`.

| Path | Bytes | SHA-256 | Git blob |
|---|---:|---|---|
| `cbz/evidence/verifier-audit-db642-20261029/DECISION_ADDENDUM.md` | 7452 | `5c3914a0f3e6ab2ae7212ad3b0640d035816d30f63fb5f6ec016e1b7b4c5cc7c` | `10e874d3f09926541288c4a901a71c536da0237f` |
| `cbz/evidence/verifier-audit-db642-20261029/README.md` | 10147 | `c1f28a3da2b41ad3e74cc89ac9c310e3907f28b490f96debe976483e674c86a6` | `6de6a67e5ecb58b71037ca242411445a990e6488` |
| `cbz/evidence/verifier-audit-db642-20261029/SHA256SUMS` | 3047 | `da9750c6d1fbaca64bd249257730162d7b9b93ca937facba5d14d32d3e8f8978` | `27a38c9067ea7c3dc162518442aa6a2f8767665c` |
| `cbz/evidence/verifier-audit-db642-20261029/TOOL_REGISTRATION_ADDENDUM.md` | 4895 | `06fefcb94555f4e8a7a16d5ce55e0bb5ad330b545d6dd4eb16a7e2b4761befc3` | `b5913c53eb4a496a7f4e5158777a5302a2e5f021` |
| `formal/README.md` | 10274 | `1fbbb9ec763f36aa68894bfae594294bd36aa1b4d3e755737f9a0971131452f0` | `7adeb84ab90b72033defd881e78fadafcc0a65b2` |
| `formal/manifest.json` | 19412 | `1c7aebd2e289cee0d9f82e15052d355efaaef642ff0cb5c47332435adfea75ea` | `5735b1f61aee18cd4bbb4f4e5a961fcf5cca5254` |
| `tests/fixtures/atlas_formal_bridge_profiles/README.md` | 1832 | `43bd5b8df4dfe155740ff87ec0bf822b585a0970ee5cf2f375f20d0c0c22f37e` | `73001776943743b142e81699ed2829302dec5638` |
| `tests/test_atlas_formal_evidence_build.py` | 10315 | `b2afc10f2f6ca69bdd7af1688aa3dc36c8b88d32748e5a87c7b6405436394805` | `82ac34d90b1920ce886abee92cd4ffcded0b92f0` |
| `tests/test_formal_gate.py` | 28217 | `1d44e75b6a108ce01132962b7bc31d6d506fb485f0d225e83c7e09dfa67ca5f4` | `63f33491a8965405fad65f699c3d5cb78d54cbf7` |
| `tests/test_formal_negative_diagnostics.py` | 9817 | `3847f5382966e39e951e54d8558387227540cff775d96eb33ce3d6396f52107d` | `a09342292729eb1cbc2204fee096e481a5d39662` |
| `tools/atlas_formal_evidence_build.py` | 11973 | `c7c9cb540daf850f02b2ee96510149b5fa3f0f150009cb30841636dba5897347` | `f850912dbb2959aa6c32febe625bfe6dcdadc947` |
| `tools/atlas_formal_source_bridge.py` | 11850 | `63d8795be8739d91c554565225a92460ef451c0834effa9a8b9da4b37c523b7c` | `a3e56945fb0f6d08871839cec4c404984f952ba6` |
| `tools/formal_gate_check.py` | 27688 | `335758c00b7888b0f90d2586f37b2ce3db128996678eb0dd239c41a0aee9b667` | `598e3f3032ebf74ecd91db08228d6a943ada3a64` |

## Proof, historical evidence and assertion preservation

`composition-verify.py` compares actual bytes against both native parents. Forty-eight existing proof/historical/fixture-data/canonical files remain exact; `composition-preservation.json` records each file's bytes, SHA-256 and Git blob. Fixture README documentation from PR689 is intentionally retained and is the only fixture-directory documentation delta; fixture data is unchanged. Every existing Lean file remains exact, and no Lean file is added or removed. The proof inventory/targets and mathematical statements are unchanged.

Selected exact identities:

| Path | SHA-256 |
|---|---|
| `docs/site/atlas-formal-evidence-data.mjs` | `441babc5f605492b361822c26d59ebee4126e45b90695915195bf951c3c943ef` |
| `docs/site/atlas-formal-source-bridge-data.mjs` | `b444116e19afc3f6af89950533894bc7368544f0b45f469577e56d1eb83bea77` |
| `docs/site/evidence/side24-main-3b39/receipt.json` | `1ae8341c86a043ba917ec1fe68bf7e66ec528cd94b15170528487009f090c7e0` |
| `docs/site/evidence/side24-main-3b39/review.json` | `ce0540dd8ac49970fd7ddb92fdb470725e2fb7ee0a69be27a6e955c8b8632852` |
| `docs/site/evidence/side24-main-3b39/review.md` | `78c21c96d851dfd6a4fbcf68edb091d3aac21b757885e615403d3fe6337e1692` |
| `docs/site/evidence/side24-main-3b39/verified-execution.zip` | `80ac88cf770afbadd89f4801a4692d6b6540a9269dfb916c249d491c2b41f737` |
| `formal/UniversalLaw.lean` | `cb94ad3490a00bc35b420a53a42ff65b6fc66a46b8a26ef1b87035d7bb15587e` |
| `formal/UniversalLaw/Side24/ConeMoments.lean` | `2bcd4a83683188856bce010c3dcdec6904e45a6d73b50b9f3891e632ff58985c` |
| `formal/UniversalLaw/Side24/Endpoints.lean` | `095102245297707316cfd7e244d5509f9b102fe5aa638b7acdbbcf9d3c9d52bd` |
| `formal/UniversalLaw/Side24/Ledger.lean` | `ae4a14f3db8cf26668e55e2d2f80e0dcecb0703826b40396d91fb6e0d266d3f7` |
| `formal/sources/side24_v1/ENCLOSURE.json` | `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811` |
| `formal/sources/side24_v1/PROOF.md` | `c06daccc4ba4b9168522b9888b76a7d599934fc3b91bd753ee5d492262917769` |
| `tests/fixtures/atlas_formal_build_3b39/manifest.json` | `8dfcfe4e9296bca47eede517f37b546ea420ec6b51944b7c16164a906b40e090` |

The original historical receipt has no `workflow_run_attempt`; it is not rewritten. The exact original archive binding is reproduced only by the fixed historical route. The live binder refuses that same legacy receipt. Unknown historical manifests, changed bound bytes, symlinks, unbound package files, altered archive/manifest/receipt/logs and current-root substitution remain adverse cases. Published historical evidence is reproduced exactly, including the REJECTED package alignment review and its one ALIGNED selected row, without promoting package status.

Both AST comparison and literal source-body comparison confirm no original test method disappeared. Every original test body equals its native input except PR692's positive runtime log-set test, which retains all old expected logs and gains four exact stderr-log names from PR689. No adverse/failing assertion was changed or removed. Details are in `composition-test-body-bytes.json` and `composition-preservation.json`. The 13 diagnostic test bodies are exact toPR689; the receipt-attempt tests are exact toPR692.

Thirty prior retained contribution evidence/log/report files remain exact to5aa6. Existing RED, first-full-failure, mutation and historical observations are preserved, not superseded as though they never failed. The contribution's decision ledger is append-only; its new measured-UTC entry corrects manually typed17:20/17:36 labels and records that native comment6503783865 was created2026-10-29T17:30:37Z. Source logs/native `created_at` remain authoritative. README and tool-registration changes describe the bounded composition; all33 regenerated contribution checksums pass.

## Verification argv and results

All commands ran from the worktree above. Every test argv, start UTC, duration, exit, count, skip and log filename is recorded in `composition-test-results.json`. The source/manifest behavior was frozen for all test runs; evidence/registration documentation was finalized while the full run was active. No tested production/test/manifest bytes changed during or after the test commands. Final commit tree equals the staged candidate recorded before completion.

| Module / command family | Normal run / skips | Optimized run / skips | Exit |
|---|---:|---:|---:|
| `test_formal_gate.py` | 51 / 1 | 51 / 1 | 0 / 0 |
| `test_formal_negative_diagnostics.py` | 13 / 0 | 13 / 0 | 0 / 0 |
| `test_required_formal_check.py` | 33 / 0 | 33 / 0 | 0 / 0 |
| `test_required_formal_binding_control.py` | 7 / 0 | 7 / 0 | 0 / 0 |
| `test_formal_workflow_shell.py` | 6 / 0 | 6 / 0 | 0 / 0 |
| `test_atlas_formal_evidence_build.py` | 16 / 0 | 16 / 0 | 0 / 0 |
| `test_atlas_formal_source_bridge.py` | 20 / 0 | 20 / 0 | 0 / 0 |

Focused totals per mode: 146 run, 145 passed, one explicit Lean skip, zero failures/errors. Full top-level normal discovery: **1567 run, 1543 passed, 24 explicit skips, zero failures/errors, exit0**, duration400.878s. No full optimized discovery is claimed.

Exact argv:

```sh
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_gate.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_negative_diagnostics.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_required_formal_binding_control.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_workflow_shell.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_atlas_formal_source_bridge.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_gate.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_negative_diagnostics.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_required_formal_check.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_required_formal_binding_control.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_workflow_shell.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_atlas_formal_source_bridge.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p 'test_*.py' -v
python3 -B -S tools/formal_gate_check.py
python3 -B -O -S tools/formal_gate_check.py
python3 -B /workspace/scratch/db6428847655/cbz/implementation/receipt/composition-verify.py
```

Both source modes exit0 and report SOURCE_IDENTITY_PASS at the combined manifest above, 31 targets and PENDING_INDEPENDENT_REVIEW. These are not Lean executions. The permissive-axiom-auditor mutation test and adverse infrastructure-output controls pass in both focused modes. Inherited receipt-check-removal mutation evidence is retained and inspected; it was not rerun as a new mutation on this composition.

Explicit full-run skips (the focused normal/-O Lean skip is the same real-package case):

- test_kernel_enforces_cpu_memory_and_process_limits (test_architecture_sandbox.ArchitectureSandbox.test_kernel_enforces_cpu_memory_and_process_limits) ... skipped 'requires hosted restricted container'
- test_network_namespace_has_no_external_interface_or_route (test_architecture_sandbox.ArchitectureSandbox.test_network_namespace_has_no_external_interface_or_route) ... skipped 'requires hosted restricted container'
- test_nonroot_and_no_new_privileges (test_architecture_sandbox.ArchitectureSandbox.test_nonroot_and_no_new_privileges) ... skipped 'requires hosted restricted container'
- test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space (test_architecture_sandbox.ArchitectureSandbox.test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space) ... skipped 'requires hosted restricted container'
- test_source_mount_is_read_only_and_environment_has_no_publishing_secret (test_architecture_sandbox.ArchitectureSandbox.test_source_mount_is_read_only_and_environment_has_no_publishing_secret) ... skipped 'requires hosted restricted container'
- test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary (test_atlas_focus_geometry.AtlasListFocusGeometry.test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary) ... skipped 'Optional Playwright and local Chromium are required'
- test_new_selection_after_a_long_read_keeps_its_heading_visible (test_atlas_reader_scroll.AtlasReaderScroll.test_new_selection_after_a_long_read_keeps_its_heading_visible) ... skipped 'Optional Playwright and a local Chromium browser are required'
- test_real_package_executes_with_receipt (test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt) ... skipped 'Lean toolchain not installed'
- test_525_composition (test_graph_binding_check.Composition525Test.test_525_composition) ... skipped '29b7d596 or Math- dbe6c0a1 unavailable'
- test_real_records_cited_artifacts_when_present (test_living_project_validation.RealRecordTests.test_real_records_cited_artifacts_when_present) ... skipped 'no real record cites repo artifact paths; missing-artifact checks are not exercised'
- test_real_records_dependencies_when_present (test_living_project_validation.RealRecordTests.test_real_records_dependencies_when_present) ... skipped 'no real record carries depends_on; dependency checks are not exercised. depends_on is not interface v0. Dependency validation is not proof verification.'
- test_real_records_evidence_id_pattern (test_living_project_validation.RealRecordTests.test_real_records_evidence_id_pattern) ... skipped 'no real record has evidence_url, released_url, or resolution_url; the evidence id pattern is not exercised on records/node_status'
- test_real_records_fresh_against_live_source_head (test_living_project_validation.RealRecordTests.test_real_records_fresh_against_live_source_head) ... skipped 'fresh records eng.d0-packet-allowlist-fix, eng.hard-gate, math.lifetime-remainder, math.side24-coefficient are not compared to a live source head; tools/validate_node_status.py performs no network fetch, so this check is skipped instead of passed'
- test_real_records_owner_released_url_consistency (test_living_project_validation.RealRecordTests.test_real_records_owner_released_url_consistency) ... skipped 'every real record has owner null; owner/released_url consistency is not exercised (released_url must stay on the owner object and must not clear it)'
- test_changed_byte_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_changed_byte_is_not_hidden) ... skipped 'installed pytest required'
- test_clean_root_run_prevents_package_bytecode (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_clean_root_run_prevents_package_bytecode) ... skipped 'installed pytest required'
- test_dirty_bytecode_is_preserved_and_rejected (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_dirty_bytecode_is_preserved_and_rejected) ... skipped 'installed pytest required'
- test_discovery_keeps_neighbours_and_same_basename (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_discovery_keeps_neighbours_and_same_basename) ... skipped 'installed pytest required'
- test_empty_cache_directory_is_accepted (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_empty_cache_directory_is_accepted) ... skipped 'installed pytest required'
- test_exclusion_is_exact (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_exclusion_is_exact) ... skipped 'installed pytest required'
- test_explicit_B_baseline_works (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_explicit_B_baseline_works) ... skipped 'installed pytest required'
- test_extra_file_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_extra_file_is_not_hidden) ... skipped 'installed pytest required'
- test_missing_file_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_missing_file_is_not_hidden) ... skipped 'installed pytest required'
- test_neighbour_failures_are_not_suppressed (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_neighbour_failures_are_not_suppressed) ... skipped 'installed pytest required'

## Retained unsuccessful checks and verification boundaries

- `composition-preservation-first.log` retains the first read-only preservation-check transcript: the selector initially included the fixture README, which PR689 intentionally documents. It was corrected to distinguish unchanged fixture data from that README. This was a verification-script scope error, not a production failure or permission to change fixture data.
- `composition-preservation.log` retains a second read-only checker failure: the one permitted stronger positive log-inventory test was initially identified by the wrong method name. Only the checker name was corrected; `composition-preservation-final.log` and literal body comparison then passed. No production/test behavior was modified to satisfy either checker.
- `git diff --check 5aa6fbac` exits0. Against4da34569 it exits2 only for existing receipt RED-log trailing whitespace and terminal blank lines in the inherited plan/spec. `composition-diff-check-4da34569.log` preserves those warnings; original failure-log bytes were not cosmetically rewritten.
- An optional process-status read failed with `fatal library error, lookup self`; it was not retried or bypassed and is not a test/kernel result.

Fresh local Lean launch is unavailable in this session. No `--run-lean`, kernel execution or hosted execution was attempted or invented here. Synthetic subprocess observations exercise real Python validation/logging/auditing code but are not compiler or kernel evidence. Hosted qualification for this exact combined tree belongs to the parent after review. There is no claim of complete compiled declaration inventory, semantic alignment, all Layer A, full environment coverage, federation regression closure, or CBZ-0. Explicit skips remain open environmental/data coverage.

## Final local state

The coherent local merge has exactly the two native parents above, the expected final tree, no unresolved conflicts and a clean worktree. No remote write was made. Future source changes require their own composition/qualification; the frozen4da PR689 work must not be described as including a newer head.


# Successor composition: frozen PR689 50575e98

Instrumented completion UTC `2026-10-29T18:04:35.706380Z`. This section extends the preceding accepted composition; earlier sections and original results describe their own frozen inputs. Every predecessor report byte is preserved as this file's prefix and as `composition-successor/predecessor-report.md` (SHA-256 `a38e9703a95ee0bae72ccd25f07144f7056d59725aaee9425e410a4cbc509bc9`, 22656 bytes). The original `composition-test-results.json` is unchanged (SHA-256 `5c25a63ef222affee1b9e68fea54471ea0cff53bf8bec3cb97b9607e7ebb6dd7`). All new raw logs, source diffs, argv/results, guards and identities are under `composition-successor/`.

## New exact identities and scope

| Role | Identity |
|---|---|
| Accepted predecessor / first merge parent | `da7396732233dbd66745c13b160a9a104f91341e` |
| First parent tree | `507961762a1ec7c816458b712d09e27f3153b81c` |
| Frozen PR689 successor / second merge parent | `50575e9823b883f7d766e0d9a45008c5e49c2fc6` |
| Second parent tree | `c583a14f87f8ce08fe805e24dda9c21e47ab2850` |
| Actual successor merge-base | `4da34569f2e6e70267d2ee9d3170916b9b598279` |
| Final successor merge commit | `627b4049884d0b64ce879838361eb0c02e03a31f` |
| Final tested candidate / commit tree | `a7e0ee448e512df80b69883d8af86879b325f241` |
| Combined manifest SHA-256 | `34767f4d89e0d1b5135a7425fc617a471238ea2a64da7e3ad78080490d31c914` |
| Branch | `codex/cbz-formal-composition-db642-20261029` |

The appended brief ruling and `PR689_505_DESCRIPTION.md` authorize this necessary live-successor composition. `COMPOSITION_REVIEW.md` accepts the unchanged predecessor da739673. Both original native receipt/diagnostic inputs remain in its preserved ancestry; da739673 and exact 50575e98 are this merge's two direct parents. PR689 successor 505 itself has parents 4da34569 and 81acc060. Root AGENTS/STATUS/GOVERNANCE/DECISIONS are byte-equal to native 81acc060 and both direct inputs. No canonical root surfaces were edited.

The source now includes the frozen successor's complete compiled-package checker: 31 existing target theorems plus 14 exact non-target kernel records (45 kernel constants in the declared clean inventory), with type/value transitive axiom closure, and 11 separately bound IR ownership names. IR names remain explicitly distinct from kernel constants. These are retained source contracts, not a freshly observed compiled result on this combined commit. No mathematical statement, bound, target, Lean source, historical receipt/archive or fixture data was changed. Later PR689 heads are excluded; this composition stops at 50575e98.

## Resolution and custody

The exact command `git merge --no-commit --no-ff 50575e9823b883f7d766e0d9a45008c5e49c2fc6` produced only one conflict, `formal/manifest.json`. Its original conflict log and unresolved diff are retained. The manifest retains every505 inventory/control/target field and every bound path. Only `tools/formal_gate_check.py` and `tests/test_formal_gate.py` hashes were recomputed for the compatible merged content. All non-file manifest values and all other bound hashes equal505 exactly.

`resolved-control-vs-505.diff` demonstrates that, against the successor, the producer gains only actual `GITHUB_RUN_ATTEMPT` serialization. Its test file gains only the original `os` import and two receipt/local-identity tests. The compiled helper, its 43-test module, workflow, shell regression, compiled-inventory document and formal README are byte-identical to 505. The workflow's official elan 4.2.4 archive/digest, failure propagation, all-control modules in both modes and their placement before the dedicated final kernel receipt/binding remain intact.

Against da739673, the sole changed original test body is the positive runtime log/event test: it adds `Inventory.lean` in the execution order and `compiled-inventory.log` in the bound log set. Removing just those two new expectations restores the original body exactly. Every original test body from 505 is byte-identical; all original methods from both inputs remain. Compiled/IR/protocol/axiom controls, seven Python auditor mutations, permissive axiom-auditor mutation, negative diagnostics and receipt/history tests remain present and run. No adverse assertion was removed or weakened.

The real gate lifecycle retains build, leanchecker and target axioms, then generates and audits `Inventory.lean` using exact kernel and IR bindings and cross-checks the target axiom set; only after types, exact negative controls and source/runtime rechecks can it write the compiled-inventory receipt and current attempt identity. Strict live binder and both historical readers are byte-identical to da739673. All seven arithmetic diagnostic controls and previous-attempt custody remain.

`verify.py` directly verified 48 proof/historical/fixture-data/canonical files against both native inputs, all exact successor files above, the unchanged da receipt/history modules, every final bound hash, all 33 contribution checksums, append-only ledger and the expected merge-head guards. The contribution append documents the newly authorized scope and does not rewrite its earlier snapshot or timestamp erratum.

## Exact successor file delta

Thirteen paths differ from da739673. Every other path equals that accepted predecessor. `candidate.json` records all file identities, input trees, literal test-body comparison, preservation records and inventory counts.

| Path | Bytes | SHA-256 | Git blob |
|---|---:|---|---|
| `.github/workflows/formal-verification.yml` | 5658 | `b1d54513c2fad747f56ccda9519b8e625e031165d4ef234624b5a427295fbaa9` | `37aeb9baf7fc402256828804103e6afa50c822d2` |
| `cbz/evidence/verifier-audit-db642-20261029/DECISION_ADDENDUM.md` | 8412 | `0d22359896bc9a34cca0e42b7dafd7f5d436a365da7203570dea951e8d57c269` | `574b9cd6674f8d48a079763e0ef78c28ec826865` |
| `cbz/evidence/verifier-audit-db642-20261029/README.md` | 11491 | `8378348d46cfff5aa963f0d339e8a4ebdcde534129f63df9de5235b0046ea041` | `2eaa9c50a536411c1afbf8abd9bfa53a28be5ab3` |
| `cbz/evidence/verifier-audit-db642-20261029/SHA256SUMS` | 3047 | `0761edb7fc26ba49658eb5b31ec2649c37760cb578a68c944ad27ba5d4b0db4e` | `3f1f4ed0aefd629b971b55a5d6056e5543a2a964` |
| `cbz/evidence/verifier-audit-db642-20261029/TOOL_REGISTRATION_ADDENDUM.md` | 6312 | `adc235f8e66de2f72e0ac826f7f90466cd74c81749d94cf8a5df07371d876486` | `5d3a89036d9b62232bfb29d7a39df115abdff67a` |
| `formal/COMPILED_INVENTORY.md` | 3494 | `1496ab2501276ed463b8cca27efcd25b9b380317d8f4de5883af6836c19b2ef1` | `5599cccdb279da5c7851c96f1b5e53ef86858e09` |
| `formal/README.md` | 11856 | `8cc24be73afee43522612bf8877f93d7c4a27098874c71f2c58ebb8acf949a04` | `a39337931007083b575f6b1e25354e24f3e07fac` |
| `formal/manifest.json` | 22812 | `34767f4d89e0d1b5135a7425fc617a471238ea2a64da7e3ad78080490d31c914` | `4974726b86bd508cf8ff8385df2dad8989dde0e8` |
| `tests/test_formal_compiled_inventory.py` | 23998 | `f209cf72239b3068430b5652034a125495478805c9a083fcfeb12a4f51fd0062` | `9c8dbcb926a2a92d4b7e28ec36d9a2c3b3bf87ed` |
| `tests/test_formal_gate.py` | 29110 | `304b84f6bf4af77c75f397307a32488ebe9554c9d9e436ddba49f3c4bcf54113` | `46e0b6e9316142488b00e6e60f67d08bcda10117` |
| `tests/test_formal_workflow_shell.py` | 7138 | `0c316bd2c971b18ebef1df128365fefeb6bc1006c842e700060cbfd7b607d13f` | `03951f4a39d54d172e92a6c59f73fbb23eefc0e2` |
| `tools/formal_compiled_inventory.py` | 21394 | `094ae0294bfa10f6af5a797cc7939909ab0094194c2b01b468b33e0d3aac1bab` | `c037a8c3cdab114089a37f67aeb5d109277083ea` |
| `tools/formal_gate_check.py` | 29005 | `60e33e3c768ce7bf39189184d077f810ec4408e5bfc90f0a10932d1cb2adff51` | `76720b67449a348672949e9d500bf8366d5339e5` |

## Fresh successor verification

All tests ran from `/workspace/scratch/db6428847655/cbz/worktrees/main-composition` against the frozen candidate tree above, which equals the final commit tree. No source, manifest, test or contribution document changed during/after these test runs. `composition-successor/composition-test-results.json` records exact argv, UTC starts, durations, exits, counts, skips and log names. The test runner and full raw outputs are retained there.

| Module | Normal run / skips | Optimized run / skips | Exit |
|---|---:|---:|---:|
| `test_formal_gate.py` | 51 / 1 | 51 / 1 | 0 / 0 |
| `test_formal_compiled_inventory.py` | 43 / 1 | 43 / 1 | 0 / 0 |
| `test_formal_negative_diagnostics.py` | 13 / 0 | 13 / 0 | 0 / 0 |
| `test_required_formal_check.py` | 33 / 0 | 33 / 0 | 0 / 0 |
| `test_required_formal_binding_control.py` | 7 / 0 | 7 / 0 | 0 / 0 |
| `test_formal_workflow_shell.py` | 9 / 0 | 9 / 0 | 0 / 0 |
| `test_atlas_formal_evidence_build.py` | 16 / 0 | 16 / 0 | 0 / 0 |
| `test_atlas_formal_source_bridge.py` | 20 / 0 | 20 / 0 | 0 / 0 |

Per mode: **192 run, 190 passed, two explicit Lean-dependent skips, zero failures/errors**. Fresh full normal discovery: **1613 run, 1588 passed, 25 explicit skips, zero failures/errors, exit0**, duration376.778s. Full optimized discovery was not run or claimed.

Exact test argv and source checks:

```sh
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_gate.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_compiled_inventory.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_negative_diagnostics.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_required_formal_check.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_required_formal_binding_control.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_formal_workflow_shell.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p test_atlas_formal_source_bridge.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_gate.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_compiled_inventory.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_negative_diagnostics.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_required_formal_check.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_required_formal_binding_control.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_formal_workflow_shell.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_atlas_formal_evidence_build.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -O -S -m unittest discover -s tests -p test_atlas_formal_source_bridge.py -v
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -S -m unittest discover -s tests -p 'test_*.py' -v
python3 -B -S tools/formal_gate_check.py
python3 -B -O -S tools/formal_gate_check.py
python3 -B /workspace/scratch/db6428847655/cbz/implementation/receipt/composition-successor/verify.py
```

Both source gates exit0 at the manifest above,31 targets and PENDING_INDEPENDENT_REVIEW. `git diff --cached --check da739673` exits0. Source outputs/extra argv are retained in `extra-checks.json`; preservation and precommit guards are separate raw logs. There was no unsuccessful production/test result to repair in this successor round; the expected merge conflict was resolved without weakening a control.

Explicit full-run skips (focused normal/-O each contain the real-package and real-inventory cases):

- test_kernel_enforces_cpu_memory_and_process_limits (test_architecture_sandbox.ArchitectureSandbox.test_kernel_enforces_cpu_memory_and_process_limits) ... skipped 'requires hosted restricted container'
- test_network_namespace_has_no_external_interface_or_route (test_architecture_sandbox.ArchitectureSandbox.test_network_namespace_has_no_external_interface_or_route) ... skipped 'requires hosted restricted container'
- test_nonroot_and_no_new_privileges (test_architecture_sandbox.ArchitectureSandbox.test_nonroot_and_no_new_privileges) ... skipped 'requires hosted restricted container'
- test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space (test_architecture_sandbox.ArchitectureSandbox.test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space) ... skipped 'requires hosted restricted container'
- test_source_mount_is_read_only_and_environment_has_no_publishing_secret (test_architecture_sandbox.ArchitectureSandbox.test_source_mount_is_read_only_and_environment_has_no_publishing_secret) ... skipped 'requires hosted restricted container'
- test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary (test_atlas_focus_geometry.AtlasListFocusGeometry.test_keyboard_focus_ring_fits_inside_the_list_scroll_boundary) ... skipped 'Optional Playwright and local Chromium are required'
- test_new_selection_after_a_long_read_keeps_its_heading_visible (test_atlas_reader_scroll.AtlasReaderScroll.test_new_selection_after_a_long_read_keeps_its_heading_visible) ... skipped 'Optional Playwright and a local Chromium browser are required'
- test_actual_clean_and_omitted_constants (test_formal_compiled_inventory.PinnedLeanInventoryTests.test_actual_clean_and_omitted_constants) ... skipped 'pinned Lean unavailable'
- test_real_package_executes_with_receipt (test_formal_gate.RepositoryControls.test_real_package_executes_with_receipt) ... skipped 'Lean toolchain not installed'
- test_525_composition (test_graph_binding_check.Composition525Test.test_525_composition) ... skipped '29b7d596 or Math- dbe6c0a1 unavailable'
- test_real_records_cited_artifacts_when_present (test_living_project_validation.RealRecordTests.test_real_records_cited_artifacts_when_present) ... skipped 'no real record cites repo artifact paths; missing-artifact checks are not exercised'
- test_real_records_dependencies_when_present (test_living_project_validation.RealRecordTests.test_real_records_dependencies_when_present) ... skipped 'no real record carries depends_on; dependency checks are not exercised. depends_on is not interface v0. Dependency validation is not proof verification.'
- test_real_records_evidence_id_pattern (test_living_project_validation.RealRecordTests.test_real_records_evidence_id_pattern) ... skipped 'no real record has evidence_url, released_url, or resolution_url; the evidence id pattern is not exercised on records/node_status'
- test_real_records_fresh_against_live_source_head (test_living_project_validation.RealRecordTests.test_real_records_fresh_against_live_source_head) ... skipped 'fresh records eng.d0-packet-allowlist-fix, eng.hard-gate, math.lifetime-remainder, math.side24-coefficient are not compared to a live source head; tools/validate_node_status.py performs no network fetch, so this check is skipped instead of passed'
- test_real_records_owner_released_url_consistency (test_living_project_validation.RealRecordTests.test_real_records_owner_released_url_consistency) ... skipped 'every real record has owner null; owner/released_url consistency is not exercised (released_url must stay on the owner object and must not clear it)'
- test_changed_byte_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_changed_byte_is_not_hidden) ... skipped 'installed pytest required'
- test_clean_root_run_prevents_package_bytecode (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_clean_root_run_prevents_package_bytecode) ... skipped 'installed pytest required'
- test_dirty_bytecode_is_preserved_and_rejected (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_dirty_bytecode_is_preserved_and_rejected) ... skipped 'installed pytest required'
- test_discovery_keeps_neighbours_and_same_basename (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_discovery_keeps_neighbours_and_same_basename) ... skipped 'installed pytest required'
- test_empty_cache_directory_is_accepted (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_empty_cache_directory_is_accepted) ... skipped 'installed pytest required'
- test_exclusion_is_exact (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_exclusion_is_exact) ... skipped 'installed pytest required'
- test_explicit_B_baseline_works (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_explicit_B_baseline_works) ... skipped 'installed pytest required'
- test_extra_file_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_extra_file_is_not_hidden) ... skipped 'installed pytest required'
- test_missing_file_is_not_hidden (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_missing_file_is_not_hidden) ... skipped 'installed pytest required'
- test_neighbour_failures_are_not_suppressed (test_pytest_bytecode_policy.PytestBytecodePolicyTests.test_neighbour_failures_are_not_suppressed) ... skipped 'installed pytest required'

## Limits and handoff

No local Lean launch, `--run-lean`, hosted action, remote write, foreign branch mutation, subagent or ref cleanup occurred. Local real-package and compiled-inventory specimen tests explicitly skipped because the runtime is unavailable; synthetic transcripts exercise actual Python logic but cannot establish kernel observations. The source author's standalone505 hosted result is retained as description context, not inherited as a PASS for this new composition. Exact combined-source hosted qualification remains for the parent after review.

This bounded package source includes the full compiled/IR controls that were absent from the predecessor; it does not certify their execution here, semantic alignment, other packages, whole-research proof closure, all Layer A or CBZ-0. Environmental/data skips remain open coverage. The coherent local merge retains all native inputs and source evidence, matches the tested tree exactly, has no unresolved conflict and leaves the owned worktree clean. Parent review is requested on this exact successor delta before publication.
