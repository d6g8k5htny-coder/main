# Interface graph archive readback v1 contract

Root pickup 6070431864/amendment 6070796606 owns this bounded next graph-to-SSG unit. This document is the only current write. Proposed later files are `architecture/graph_artifact_readback.py`, `tools/architecture_graph_artifact_readback.py`, and `tests/test_architecture_graph_artifact_readback.py`; root claims and assigns those separately after contract review. Standing owner authorization applies. No local project execution, implementation, tests, deployment, or private Artifact Index capture access is part of this design task.

## Observed retained data and the custody boundary

Source-only ZIP reads and hashes on 8 October 2026 inspected `work/evidence/architecture-casefold-required-main00bcd-37856787382.zip`, outside the public Git tree. Original bytes are 19001 bytes, SHA256 `7c5891445d327b88f1fb7a82966ebf681643286a64179b2389a5bcb3da41e8b9`. Root supplies the native artifact identity 11584985628, name `research-architecture-37856787382-1`, producer repository `d6g8k5htny-coder/main`, run 37856787382/attempt 1, checked commit `dcb30554f9cc9e60aeb9f6b4b8efa3b5c9287a3b`, and original receipt SHA256 `4d32e7d72b999c74c13726f665cf57632959ae797541a22da458ade3b41ca5d2`. The 19001-byte ZIP has 13 regular DEFLATED members, no archive comment, classic single-disk central directory, and signed 32-bit data descriptors. This inspection read data; it did not rerun the exporter, gate, tests, or receipt helper.

The original graph member is 48932 bytes with SHA256 `30cdcedabbdd3d9906a6cf13c838a23424ca065225d179c112db3ed7f5b3aa7f`; data inspection counted 49 dictionary node IDs and 55 edges. The original receipt is 22627 bytes. Both full logs record 89 passing qualified IDs and terminal `OK`; the sorted IDs joined with one LF after each ID have SHA256 `b51eaf6cfaede42105ad377c066c4b18789e320ccf542b6817a6850c350fc9ef`. These are original retained-data observations, not a new execution or scientific acceptance.

Independently supplied native expectations establish comparison context. The reader can prove that bytes match those declared pins and that archive contents satisfy this contract; it cannot authenticate GitHub, an artifact ID, a producer conclusion, or an archive download through caller strings. Root's separate native producer/job/artifact readback establishes that external association. A synthetically regenerated archive with supplied matching pins remains synthetic. Every output retains `custody: "unknown"`, scientific effect `NONE`, and exact false scientific-status authority. No capture-authentication override is accepted.

The producer checked commit above is distinct from the dated Math source commit below and from a later executing reader/SSG build commit. The reader never rewrites one identity into another and never requires an old producer commit to equal its own current checkout. The later immutable build manifest binds its executing commit, this output unit's SHA256, the separately governed public Artifact Index projection, and emitted assets. That build/deployment unit remains separate.

## Exact CLI, module, and output ABI

The external CLI is `python -B -S tools/architecture_graph_artifact_readback.py`. It requires exactly once, with no abbreviation, `--archive FILE` and these eight flags:

| Flag | Binding field | Exact value type |
|---|---|---|
| `--expected-commit` | checked_commit | lowercase 40-hex string |
| `--expected-repository` | repository | ASCII `[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+`, neither component `.` or `..` |
| `--expected-run-id` | run_id | ASCII positive decimal, no leading zero |
| `--expected-run-attempt` | run_attempt | ASCII positive decimal, no leading zero |
| `--expected-receipt-sha256` | receipt_sha256 | lowercase 64-hex string |
| `--expected-artifact-id` | artifact_id | ASCII positive decimal, no leading zero |
| `--expected-artifact-sha256` | artifact_sha256 | lowercase 64-hex string |
| `--expected-artifact-name` | artifact_name | exactly `research-architecture-<run_id>-<run_attempt>` |

The expectations cannot be learned from stdin, the ZIP, its manifest, receipt, log, or filename. No stdin packet, output-path flag, optional trust pin, URL input, alternate graph source, extraction, network fetch, or captured-code import is supported. Any missing/duplicate/unknown/abbreviated argument or invalid input refuses nonzero with empty stdout and exactly `ARCHITECTURE_GRAPH_READBACK_REFUSED\n` on stderr. Diagnostics never echo paths, members, archive bytes, logs, or exception text. Success exits 0, stderr is empty, and stdout is one deterministic ASCII JSON document with sorted object keys, compact separators, preserved JSON types/array order, and one terminal LF.

The pure stdlib module API is `read_graph_archive(raw_zip: bytes, expected: dict[str, str], source_graph_bytes: bytes) -> dict`. The expected dictionary has exactly the eight binding keys above, all exact strings. The source bytes must match the independently fixed source SHA below, even if callers provide another internally coherent source. The module performs no I/O. The CLI reads only the archive and the bundled fixed `docs/site/dependency-source/GRAPH.json`; it imports only trusted implementation/stdlib code and never `hard_gate.py`, the producer receipt helper, tests, or captured members. Suppress bytecode writes. Reuse the current helper's documented data rules independently, not its executable acceptance result.

Success root exact keys are `{schema_version, scientific_effect, scientific_status_authority, custody, graph, receipt}` with schema_version exact integer 1, scientific_effect `NONE`, authority exact false, custody `unknown`, and graph the original parsed seven-field export with every typed record preserved. Receipt exact keys are `{schema_version, scientific_effect, scientific_status_authority, custody, verification_scope, binding, archive, graph_member, producer_receipt_sha256, producer_receipt}`. The same exact non-scientific constants apply; verification_scope is `declared-native-pins-and-archive-content`, binding preserves the eight externally supplied strings, and producer_receipt is the validated original parsed producer receipt without added/mutated fields.

`archive` is exactly `{sha256, bytes, members}`: original raw ZIP SHA and byte count, then 13 entries sorted by canonical path, each exactly `{path, bytes, sha256}` derived from fully read original member bytes, including `ARTIFACT_SHA256SUMS`. `graph_member` is exactly `{path: "generated/graph.json", bytes, sha256}`. producer_receipt_sha256 is the original member SHA matching the external receipt pin. No raw ZIP/base64, log/console text, exporter output, inspected-image annotations, filesystem path, validation timestamp, inferred graph join, accepted badge, kernel result, or source-custody assertion is projected.

Validation completes before stdout serialization. The complete encoded output is bounded at 16 MiB. The CLI writes no files; a trusted caller retains stdout in a temporary file and promotes the one unit atomically only after exit 0 and successful complete-output validation. Refusal preserves any previously published caller-owned unit. A broken/partial output transport is discarded by that caller; it is never a valid unit merely because some JSON bytes appeared.

## File and ZIP admission

Archive raw bytes are bounded at 64 MiB. The CLI anchors every absolute parent component with `O_DIRECTORY|O_NOFOLLOW`, resolving a relative input against the trusted current directory without erasing symlink components. Reject `..` components; never call resolve/abspath in a way that collapses an unchecked component. The leaf uses `O_NOFOLLOW|O_NONBLOCK`; fstat must establish a regular file before reading. Read at most 64 MiB + 1 once through that descriptor, check byte count and pre/post descriptor identity/size/change metadata, and compare the SHA of that exact buffer to expected artifact_sha256 before archive parsing. Directory, FIFO, device, symlink leaf/ancestor, unavailable, changing, and oversized files refuse without waiting for a writer. The bundled source graph uses the same anchored regular-file policy with a 16 MiB cap and its fixed byte/SHA check.

V1 admits the observed classic single-disk ZIP subset, with STORED or DEFLATED entries and optional ordinary 32-bit data descriptors. Allowed general-purpose flag bits are 3 (data descriptor), 11 (UTF-8 names), and, for DEFLATED entries only, 1/2 (compression option); all other set bits refuse. ZIP64, multidisk, encrypted/strong-encrypted, unsupported compression/flags, archive/file comments, and extra fields refuse; these are explicit unsupported formats, not claims that all general ZIPs are unsafe. The initial local header begins at byte 0, the end-of-central-directory record ends exactly at EOF, central-directory count/size/offsets agree, and every local entry belongs to exactly one central entry. Local names/flags/methods and declared CRC/sizes or its 12/16-byte descriptor must agree with the central record. Local header/data/descriptor spans cannot overlap, share an offset, contain unreferenced gaps/entries, or extend into the central directory. No self-extracting prefix, concatenated/trailing archive, truncation, hidden duplicate, or unsigned range wrap is accepted.

Inspect every entry before reading any content. Raw/decoded/original filenames must agree and be ASCII exact canonical names; reject NUL/control characters, backslashes, absolute/drive paths, empty/dot/dot-dot components, directories, trailing separators, casefold aliases, and duplicates even if a library would return the last entry. Unix type bits must be regular when present; absent type bits are admitted only for a nondirectory DOS regular entry. Symlink, FIFO, socket, device, and other special attributes refuse. Permission timestamps/order may vary and are not projected. Both file order and manifest line order may vary; the member-name set may not vary in v1.

The exact 13 members are the 12 data members in the next section plus `ARTIFACT_SHA256SUMS`. KiB/MiB mean 1024/1048576 bytes. Total inflated member bytes are bounded at 64 MiB; the smaller member limits are stronger for this closed inventory, while independent aggregate accounting remains required. Member limits are: manifest 16 KiB; checked-commit/run-id/run-attempt/container-status each 128 bytes; exporter.stdout and exporter.stderr each 64 KiB; runtime-image.json 1 MiB; generated/graph.json, architecture-run-receipt.json and console.log each 4 MiB; normal/optimized logs each 16 MiB. Declared sizes must satisfy limits before decompression; bounded streaming must enforce each actual count and aggregate count independently. Read every member to its complete end and verify CRC; for DEFLATED data also require one complete deflate stream with no unused compressed tail. A high compression ratio alone is not rejected: valid repetitive logs remain admissible within the actual byte limits. A compression bomb that exceeds a declaration/budget, never ends, or fails structure/CRC refuses before publication. No entry is extracted, evaluated, executed, or imported.

## Manifest, producer receipt, and native consistency

`ARTIFACT_SHA256SUMS` is strict ASCII: exactly 12 nonempty lines, each lowercase 64-hex, two spaces, one canonical name from this exact set, and a final LF. No duplicate/case alias, escaped filename, binary marker, blank line, self-entry, unknown/missing member, or trailing text is allowed. Every listed SHA is compared with independently recomputed full original member bytes; never accept the manifest as its own proof.

```text
runtime-image.json
checked-commit.txt
run-id.txt
run-attempt.txt
container-exit-status.txt
console.log
tests-normal.log
tests-optimized.log
exporter.stdout
exporter.stderr
generated/graph.json
architecture-run-receipt.json
```

The receipt's eight evidence_file_sha256 keys are exactly generated/graph.json, checked-commit.txt, run-id.txt, run-attempt.txt, container-exit-status.txt, tests-normal.log, tests-optimized.log, runtime-image.json; every lowercase 64-hex value must independently match those eight raw members. The manifest additionally verifies the receipt and all three diagnostics. Manifest SHA and receipt hashes remain internal integrity until matched with the separate externally supplied raw ZIP and receipt pins; regenerating a manifest does not authorize changed context or source records.

Producer receipt exact root keys are `{schema_version, scientific_effect, scientific_status_authority, checked_commit, repository, run_id, run_attempt, runtime_image, test_modes, graph_source, evidence_file_sha256}`. Require integer 1, `NONE`, exact false, and all four native strings exactly equal external commit/repository/run/attempt. checked-commit.txt, run-id.txt and run-attempt.txt equal the corresponding ASCII string plus exactly one LF; the repository exists in the receipt, not a fabricated repository.txt member. container-exit-status.txt equals exactly `0\n`. exporter.stderr must be empty and exporter.stdout must equal exactly `/output/generated/graph.json\n`, the current fixed producer invocation. console.log may contain bounded inert diagnostics and is hash-verified but is never interpreted or projected.

runtime-image.json is strict JSON, a one-element list containing an object. Native extra inspected-image fields are admitted as inert data, but Id must be a lowercase `sha256:<64hex>` string, RepoDigests exactly `["python@sha256:83f339c1be6340ae1096010fdccf6552ac932d8f410d45d206014916bdf37e48"]`, Os `linux`, Architecture `amd64`. The receipt runtime_image is exactly `{id, repo_digest}` and must equal those checked values; do not hardcode a synthetic/current image ID. All JSON members and bundled source refuse duplicate keys at any depth, invalid UTF-8/surrogates, nonfinite constants/overflow, multiple documents, malformed types, and more than 64 container levels (root container counts; scalar leaves do not). Canonical typed comparisons distinguish false/0 and integer/float, without coercion.

## Complete test inventory and exact graph policy

Derive both inventories from the full raw normal/optimized logs independently of receipt claims. Logs must be strict UTF-8 text with LF line boundaries and no CR/NUL. Accept only complete lines `<short_name> (<qualified_id>) ... ok`, with no leading whitespace, exact delimiter, short name equal the qualified last component, and identity matching ASCII `test_architecture_[A-Za-z0-9_]+\.[A-Za-z_][A-Za-z0-9_]*\.test[A-Za-z0-9_]*`. `test1` and `testExtra` remain valid prefixes. Every ID occurs exactly once per mode, all 89 immutable IDs in the appendix must occur, and later valid unique passing IDs are allowed, up to 4096 per mode. Both discovered inventories must be identical. A count-only 89 substitute is forbidden.

Every result-looking line must match the complete passing grammar: detect case-insensitive test prefixes after optional whitespace, parenthesized qualified test markers, ellipses after a right parenthesis, and the ordinary result delimiter. Malformed/foreign/shortname-mismatched results, FAIL/ERROR/FAILED, skips, expected failures, unexpected successes, or duplicate/partial runs refuse. Exactly one `Ran <positive-decimal> tests in <finite-nonnegative-decimal>s` footer follows all test results, its count equals discovered IDs, and exactly one terminal `OK` is the last nonempty line. Diagnostic blank lines, separators and ordinary non-result text may remain inert. Receipt test_modes is exactly `[normal, optimized]`; each entry has exactly `{mode, test_count, test_ids}` with an exact integer count and sorted complete IDs equal the derived inventory. Do not discover expected IDs by executing/importing tests or learning a new floor from the input receipt.

Graph root exact keys are `{schema_version, source, nodes, edges, dimensions, scientific_effect, scientific_status_authority}` with integer 1, `NONE`, exact false. Source exact five fields are repository `d6g8k5htny-coder/Math-`, commit `7858329974e28be79f29b22644370084ff43da4f`, captured_at `2026-10-03T15:07:03Z`, graph_sha256 `8822e9618678321a342d69cd0b8ae6552de1b5d578c331de5072b2892ee9dd09`, gate_sha256 `a78f3e25f3b0cfe113e618a4c31a7a25d7f22af638c46dec1ecba221fa333ac8`. Receipt graph_source must equal those original fields exactly. Bundled GRAPH.json has 38753 original bytes and the graph_sha256 above. Verify its byte count/SHA, strict parse, then canonical typed equality of exported nodes/edges with this independent source. Preserve all 49 dictionary IDs, 55 ordered original edges, records, review qualifiers, classifications and scopes; no conversion to a node array, case aliasing, rewriting, or guessed relation to Lean or Artifact Index IDs.

Dimensions must have exactly those 49 IDs. Each value has exactly source/review/kernel/computation/alignment. Source is `recorded` iff the original node's explicit source value is truthy; review is `recorded` iff any explicit review_source, review_issue, review_basis, review_provider, review_providers or review_disposition value is truthy; otherwise each is `not_recorded`. Kernel, computation and alignment are always `not_recorded` in this dated export. Classification/proof prose and passing engineering tests cannot manufacture evidence. Raw native formal receipts and the separate evidence adapter remain distinct inputs with unknown graph-to-target joins; no actual Lean elaboration, declaration existence, per-statement proof, or accepted alignment is inferred here.

The current helper source inspected is tools/architecture_run_receipt.py SHA256 `a4bb66325f64422398ff726bc1c448afb284498275e5894a7d1af6b25fba9890`; its staged receipt validates the original eight members and 53 baseline IDs. This ZIP reader independently repeats those data invariants, strengthens the required floor to the observed complete 89 controls, verifies the producer receipt itself and all 12 manifest entries, and adds bounded archive admission. It never calls receipt() on extracted data or accepts its own newly synthesized receipt in place of the original member.

## Independent external controls and acceptance

Freeze reviewed contract before assigning tests. Target about 30 external CLI methods, with an explicit CLI-exists assertion before every child so missing implementation yields genuine hosted RED. Child interpreters retain -B/-S and active -O/-OO. Synthetic ZIP fixtures use TEST native identities and clearly labeled synthetic logs; graph source data may be read inertly from the already-public fixed source. Do not copy the private Artifact Index capture or mistake a synthetic passing log for a new native run. A separately root-supplied real-archive readback control uses the original external eight bindings above.

Meaningful control groups are: deterministic real-format positive; alternative valid native context and attempt 10; original typed graph/receipt retention; additional valid test1/testExtra controls; each eight expected pin/context substitution; all argument absence/duplicates/format errors; raw and member byte boundaries and aggregate accounting; regular/nofollow/symlink-parent/FIFO refusal; duplicate/casefold/path/NUL members; Unix special/DOS-directory types; encrypted/unsupported/multidisk/ZIP64 cases; central/local disagreement/overlap/truncation/prefix/tail; CRC and bounded expansion corruption; every missing/extra ZIP member; each 12 manifest hash and malformed/duplicate/self-row; each receipt-eight hash; receipt raw pin despite regenerated manifest; nonzero native status and each native identity mismatch; strict JSON/depth/type contrasts; all graph pins and typed node/edge/dimension mutation with coherently regenerated internal hashes; missing/substituted one of 89 IDs despite matching count; skipped/failed/expected-negative additional tests; malformed result-like extras before valid footer; unequal additional suites; receipt inventory mismatch; runtime inspect/receipt contradictions; exporter diagnostics contradictions; and inert captured strings with calibrated I/O/network observers, no extraction or writes.

Retain first exact tests-only hosted missing-CLI RED, then implementation candidate normal/optimized full discovery and original failure/success archives. Include exact-limit positives plus one-over negatives; mutations must recompute unrelated internal/external synthetic pins where needed so another mismatch does not mask the intended control. Include an independently reviewed count-only/circular-rehash mutant refusal and a meaningful positive so reject-all cannot pass. Nonauthor source review verifies the parser/privacy/IO boundary, with test-author and organizational-independence relationships disclosed. Root checks native job conclusions, original archive bytes/member identities, and executing revisions before reporting this unit implemented. No passing receipt alone promotes science, confirms custody, completes the reader, or authorizes a guessed graph mapping.

At this plan's source freeze: contract SOURCE_PREPARED; test implementation NOT_WRITTEN; hosted RED/GREEN NOT_RUN; reader/SSG/browser/deployment NOT_RUN. Existing Artifact Index tests at SHA256 `8d6fb4abc139e37f0163f344c7c53f263e2dd6a0c2f440a29643ac01c33d7736` remain unchanged.

## Immutable required 89 inventory

The literal fully qualified IDs below are independent v1 expectations, not parsed from an incoming receipt. Their sorted LF-terminated byte inventory has the b51eaf6c SHA above. Additional IDs never replace any of these 89.

```text
test_architecture_binding.ArchitectureBindingContract.test_artifact_id_requires_positive_ascii_decimal_without_leading_zero
test_architecture_binding.ArchitectureBindingContract.test_artifact_name_is_exactly_bound_to_current_run_and_attempt
test_architecture_binding.ArchitectureBindingContract.test_attempt_ten_is_a_positive_decimal_not_a_single_digit
test_architecture_binding.ArchitectureBindingContract.test_commit_format_cannot_be_authorized_by_an_equally_invalid_native_commit
test_architecture_binding.ArchitectureBindingContract.test_deeply_nested_json_is_refused_without_a_success_envelope
test_architecture_binding.ArchitectureBindingContract.test_duplicate_top_level_keys_are_refused_even_when_last_value_is_valid
test_architecture_binding.ArchitectureBindingContract.test_each_current_native_argument_is_used_for_identity_binding
test_architecture_binding.ArchitectureBindingContract.test_each_duplicate_output_key_is_refused_even_when_last_value_is_valid
test_architecture_binding.ArchitectureBindingContract.test_each_expected_native_identity_argument_is_required
test_architecture_binding.ArchitectureBindingContract.test_each_non_success_result_is_refused
test_architecture_binding.ArchitectureBindingContract.test_each_of_the_eight_outputs_is_required
test_architecture_binding.ArchitectureBindingContract.test_each_output_rejects_an_empty_string
test_architecture_binding.ArchitectureBindingContract.test_each_output_rejects_non_string_identities_without_coercion
test_architecture_binding.ArchitectureBindingContract.test_each_top_level_field_is_required
test_architecture_binding.ArchitectureBindingContract.test_extra_output_fields_are_refused
test_architecture_binding.ArchitectureBindingContract.test_extra_top_level_fields_are_refused
test_architecture_binding.ArchitectureBindingContract.test_invalid_or_multiple_json_documents_are_refused
test_architecture_binding.ArchitectureBindingContract.test_non_finite_json_numbers_are_refused
test_architecture_binding.ArchitectureBindingContract.test_non_object_top_level_values_are_refused
test_architecture_binding.ArchitectureBindingContract.test_other_valid_native_identities_are_accepted_and_preserved
test_architecture_binding.ArchitectureBindingContract.test_outputs_must_be_an_object
test_architecture_binding.ArchitectureBindingContract.test_receipt_and_artifact_sha256_require_lowercase_64_hex_digits
test_architecture_binding.ArchitectureBindingContract.test_repository_substitution_and_case_drift_are_refused
test_architecture_binding.ArchitectureBindingContract.test_run_and_attempt_decimal_format_cannot_be_authorized_by_invalid_native_args
test_architecture_binding.ArchitectureBindingContract.test_stale_attempt_is_refused_even_with_a_matching_stale_artifact_name
test_architecture_binding.ArchitectureBindingContract.test_stale_checked_commit_is_refused
test_architecture_binding.ArchitectureBindingContract.test_stale_run_is_refused_even_with_a_matching_stale_artifact_name
test_architecture_binding.ArchitectureBindingContract.test_valid_current_run_has_exact_stable_non_scientific_output
test_architecture_binding.ArchitectureBindingContract.test_valid_json_at_exact_stdin_byte_limit_is_accepted
test_architecture_binding.ArchitectureBindingContract.test_valid_json_over_stdin_byte_limit_is_refused
test_architecture_graph_export.ArchitectureGraphExportContract.test_caller_pinned_provenance_cannot_authorize_different_gate_code
test_architecture_graph_export.ArchitectureGraphExportContract.test_classification_and_proof_review_text_cannot_manufacture_evidence
test_architecture_graph_export.ArchitectureGraphExportContract.test_context_only_cycle_is_allowed_without_changing_gate_semantics
test_architecture_graph_export.ArchitectureGraphExportContract.test_default_trust_rejects_self_consistent_changed_graph_and_provenance
test_architecture_graph_export.ArchitectureGraphExportContract.test_dimensions_describe_only_explicit_recorded_metadata
test_architecture_graph_export.ArchitectureGraphExportContract.test_duplicate_graph_json_keys_are_rejected_after_identity_validation
test_architecture_graph_export.ArchitectureGraphExportContract.test_duplicate_provenance_json_keys_are_rejected_before_gate_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_exact_source_bound_hard_gate_validator_is_called
test_architecture_graph_export.ArchitectureGraphExportContract.test_failed_export_preserves_an_earlier_valid_export
test_architecture_graph_export.ArchitectureGraphExportContract.test_gate_identity_is_checked_before_untrusted_gate_executes
test_architecture_graph_export.ArchitectureGraphExportContract.test_graph_identity_is_checked_before_gate_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_missing_required_dependency_is_rejected_by_pinned_hard_gate
test_architecture_graph_export.ArchitectureGraphExportContract.test_pinned_49_node_55_edge_export_is_deterministic_and_preserves_records
test_architecture_graph_export.ArchitectureGraphExportContract.test_required_dependency_cycle_is_rejected_by_pinned_hard_gate
test_architecture_graph_export.ArchitectureGraphExportContract.test_same_size_gate_change_is_rejected_by_sha_before_import
test_architecture_graph_export.ArchitectureGraphExportContract.test_source_byte_counts_are_validated_even_when_sha_matches
test_architecture_graph_export.ArchitectureGraphExportContract.test_symlink_source_directory_is_rejected
test_architecture_graph_export.ArchitectureGraphExportContract.test_symlink_source_files_are_rejected_before_gate_import
test_architecture_receipt.ArchitectureReceiptContract.test_attempt_ten_and_other_native_repository_are_preserved
test_architecture_receipt.ArchitectureReceiptContract.test_changed_node_or_edge_is_refused_even_when_graph_source_pins_are_unchanged
test_architecture_receipt.ArchitectureReceiptContract.test_directory_or_fifo_member_is_refused_without_waiting_for_a_writer
test_architecture_receipt.ArchitectureReceiptContract.test_duplicate_json_keys_are_refused_even_when_last_value_is_valid
test_architecture_receipt.ArchitectureReceiptContract.test_duplicate_test_identity_or_multiple_runs_are_refused
test_architecture_receipt.ArchitectureReceiptContract.test_each_expected_host_identity_argument_is_used_and_required
test_architecture_receipt.ArchitectureReceiptContract.test_each_native_identity_file_must_match_current_host_arguments
test_architecture_receipt.ArchitectureReceiptContract.test_each_required_sandbox_control_rejects_skipped_failed_or_error_status
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_is_required
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_must_be_regular_and_cannot_be_a_symlink
test_architecture_receipt.ArchitectureReceiptContract.test_each_staged_member_over_16_mib_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_empty_or_zero_discovered_test_logs_are_refused_in_each_mode
test_architecture_receipt.ArchitectureReceiptContract.test_every_graph_source_pin_is_required_and_cannot_be_substituted
test_architecture_receipt.ArchitectureReceiptContract.test_every_required_identity_is_needed_even_when_53_other_tests_are_ok
test_architecture_receipt.ArchitectureReceiptContract.test_evidence_directory_or_generated_parent_symlink_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_graph_and_runtime_inspect_require_strict_json
test_architecture_receipt.ArchitectureReceiptContract.test_graph_root_schema_and_non_scientific_flags_are_exact
test_architecture_receipt.ArchitectureReceiptContract.test_invalid_matching_host_identity_does_not_authorize_bad_commit_or_decimal
test_architecture_receipt.ArchitectureReceiptContract.test_later_all_ok_test_methods_are_counted_and_bound
test_architecture_receipt.ArchitectureReceiptContract.test_malformed_extra_result_lines_cannot_be_ignored_before_the_success_footer
test_architecture_receipt.ArchitectureReceiptContract.test_malformed_foreign_or_shortname_mismatched_test_identities_are_refused
test_architecture_receipt.ArchitectureReceiptContract.test_missing_extra_or_forged_dimension_cannot_manufacture_scientific_evidence
test_architecture_receipt.ArchitectureReceiptContract.test_native_repository_argument_must_have_owner_and_repository_components
test_architecture_receipt.ArchitectureReceiptContract.test_nested_record_types_cannot_change_even_when_python_values_compare_equal
test_architecture_receipt.ArchitectureReceiptContract.test_non_finite_numbers_are_refused_even_in_additional_inspect_fields
test_architecture_receipt.ArchitectureReceiptContract.test_nonunderscore_unittest_names_are_accepted_as_additional_controls
test_architecture_receipt.ArchitectureReceiptContract.test_nonunderscore_unittest_prefix_malformed_results_are_not_diagnostics
test_architecture_receipt.ArchitectureReceiptContract.test_nonzero_or_malformed_container_exit_is_refused
test_architecture_receipt.ArchitectureReceiptContract.test_normal_and_optimized_must_discover_the_same_additional_passing_tests
test_architecture_receipt.ArchitectureReceiptContract.test_runtime_inspect_is_one_linux_amd64_image_with_the_fixed_digest
test_architecture_receipt.ArchitectureReceiptContract.test_skip_or_failure_in_additional_discovered_test_is_also_refused
test_architecture_receipt.ArchitectureReceiptContract.test_summary_must_report_exact_discovered_count_and_terminal_ok
test_architecture_receipt.ArchitectureReceiptContract.test_valid_53_test_receipt_is_exact_deterministic_and_preserves_staged_bytes
test_architecture_receipt.ArchitectureReceiptContract.test_valid_inspected_image_id_is_preserved_instead_of_hardcoded
test_architecture_receipt.ArchitectureReceiptContract.test_valid_log_at_exact_16_mib_limit_is_accepted
test_architecture_receipt.ArchitectureReceiptContract.test_valid_log_larger_than_64_kib_is_accepted_and_raw_bytes_are_hashed
test_architecture_sandbox.ArchitectureSandbox.test_kernel_enforces_cpu_memory_and_process_limits
test_architecture_sandbox.ArchitectureSandbox.test_network_namespace_has_no_external_interface_or_route
test_architecture_sandbox.ArchitectureSandbox.test_nonroot_and_no_new_privileges
test_architecture_sandbox.ArchitectureSandbox.test_output_quota_stops_disk_exhaustion_and_cleanup_recovers_space
test_architecture_sandbox.ArchitectureSandbox.test_source_mount_is_read_only_and_environment_has_no_publishing_secret
```
