# Final bounded Main composition review

Reviewer: OpenAI / Codex, agent `/root/main_composition_review`, 2026-10-29. This is a separate engineering source review, not a different-provider scientific review. The reviewer read the composition brief, both frozen PR descriptions, implementation report, exact diffs against both native parents, relevant production code/tests/workflow, and retained verification evidence. No prior input review or standalone PR692 hosted result was treated as proof of this combined source.

**Ready-to-publish on reviewed source: YES. Ready-to-merge on reviewed source: YES.** No blocking or nonblocking actionable source finding was identified. Required hosted checks for the exact combined source remain a separate prerequisite to actual integration. This verdict does not assert that those checks have run or passed.

## Exact reviewed identity

- Worktree: `/workspace/scratch/db6428847655/cbz/worktrees/main-composition`.
- Merge: `da7396732233dbd66745c13b160a9a104f91341e`.
- Tree: `507961762a1ec7c816458b712d09e27f3153b81c`.
- Native parents, in order: `5aa6fbac2c66fac0e6e2332b7e95cacc3c37350e` and `4da34569f2e6e70267d2ee9d3170916b9b598279`.
- Native merge-base: `4ef1867175dbe1072a54a02e95f80c4e2c70a4ff`; the corrected brief/report correctly distinguish this from governance ancestor `81acc060`.
- Combined manifest SHA-256: `1c7aebd2e289cee0d9f82e15052d355efaaef642ff0cb5c47332435adfea75ea`.

Direct Git inspection confirmed the merge/tree/parents, 13 changed paths against the receipt parent, and clean worktree. The reviewed diff includes all changed code, tests, manifest and contribution documentation. The second-parent diff confirms that the producer differs from frozen PR689 only by serializing the actual workflow attempt; the manifest differs only in the two combined control hashes.

## Findings and compatibility decisions

There are no P0/P1/P2/P3 actionable findings. The following file/line references identify the reviewed obligations and explain the compatibility judgment.

| Obligation | Source locations | Judgment |
|---|---|---|
| Exact current producer/consumer attempt | `tools/formal_gate_check.py:357`; `tools/required_formal_check.py:39`, `:50`, `:97` | Producer retains the actual environment value, including null local identity. Binder validates positive-decimal consumer identity and requires equality with the producer field. Missing, integer, whitespace, zero and stale producer attempts cannot equal the validated string. Binder bytes remain exact to PR692. Live CLI has only receipt/aggregate operations and no import or switch exposing historical validation. |
| Immutable historical receipt/archive route | `tools/atlas_formal_evidence_build.py:47`, `:124` | Separate validator requires the fixed archive, manifest and original receipt hashes and verifies every bound log. The fixed returned attempt is historical identity, not a missing-field default. Reconstruction verifies the archived original binding and commit and retains the original receipt bytes. |
| Fixed historical source reader | `tools/atlas_formal_evidence_build.py:69`, `:100`; `tools/atlas_formal_source_bridge.py:55`, `:113` | Only the three retained manifest hashes are allowed. Bound files pass safe-path and exact-hash checks; package symlinks and unbound files outside the excluded build area fail. Historical Python remains inert data. Both reconstruction layers use the correct distinct validators, rather than applying incompatible live source rules to old contracts. |
| All seven arithmetic controls and complete negative diagnostics | `tools/formal_gate_check.py:218`, `:265`, `:327`; `formal/manifest.json:329` | Main's default manifest requires the exact seven labels; each mutation is unambiguous and has a bound diagnostic. Execution requires arithmetic exit 1, full expected stdout including generated path/location/proposition, and empty stderr. Axiom controls require exit 0 and their full exact output. Extra, truncated, unrelated or wrong-channel diagnostics cannot count as the intended rejection. |
| Genuine axiom-auditor rejection | `tools/formal_gate_check.py:110`, `:341`; `tests/test_formal_negative_diagnostics.py:109` | Exact stdout comparison occurs before the auditor consumes the corresponding report. Therefore auditing the equal expected string is equivalent to auditing the validated observation. Only `ForbiddenAxiomError` counts; arbitrary parser `ValueError` does not. A permissive auditor causes failure and is covered in both recorded interpreter modes. |
| Prior/failed attempt custody | `tools/formal_gate_check.py:297`; `tests/test_formal_negative_diagnostics.py:88`, `:133`, `:154`; `formal/README.md:99` | Entire current evidence directory moves together into a unique history directory before executable lookup/version preflight. Symlink checks precede the move. A failed entered execution cannot leave its predecessor's successful receipt at the current path. Tests retain both receipt and log bytes and cover missing-tool failure. |
| Both test sets, proof/history and canonical custody | `tests/test_atlas_formal_evidence_build.py:68`, `:133`; `tests/test_atlas_formal_source_bridge.py:34`; contribution `DECISION_ADDENDUM.md:47` | Historical test setup choices are semantically equivalent and preserve frozen hashes. Both parents' original test bodies remain present and identical, except the authorized stronger positive log-inventory expectation. No adverse assertion was removed. Proof, historical artifact/receipt/review, fixture data and shared canonical files remain exact; timestamp correction is appended rather than rewriting old entries. |

## Independent read-only verification

No suites were rerun. I checked the existing combined-run records against their raw log count and OK/skip summaries: seven focused suites total **146 run, 145 passed, 1 explicit Lean skip in each normal and optimized mode**. The raw full-discovery log contains **1567 run, 1543 passed, 24 explicit skips, zero failures/errors**; its recorded exit is 0. The full log's unittest summary is at lines 1592–1594 rather than its final three lines, which contain expected output from adverse tests. The implementation report's summary correctly uses the actual unittest result. No full optimized discovery is claimed.

Additional review checks read actual source/Git objects without changing them:

- Recomputed all 33 contribution checksums and every current manifest-bound file hash.
- Compared the 48 preservation-record files directly with the receipt parent's Git blobs and their recorded SHA-256 values.
- Compared all non-file manifest fields and the complete file-key set with frozen PR689; only the two merged code/test hash values differ.
- Independently compared literal original test-method bodies against both parents across the six relevant test files. The sole difference is the positive runtime log-set test retaining its original expectations and adding stderr logs.
- Confirmed live binder bytes are unchanged from PR692 and `git diff --check 5aa6fbac HEAD` is clean.
- Inspected the existing workflow's ordered producer/binder steps and failure propagation; it does not turn a failed producer into a successful required binding.

The recorded combined evidence, rather than the input reviews or PR692's standalone 8/8 hosted result, supports the bounded source verdict. The inherited attempt-comparison mutation evidence remains explicitly inherited; it was not represented as a fresh combined mutation run.

## Behaviors deliberately set aside, with reasons

1. **Fresh combined Lean/kernel and hosted execution:** unavailable locally and reserved for hosted qualification. Exact diagnostic contracts can still fail on an actual compiler/environment discrepancy; such a failure must block integration, not be waived because Python tests are green. This review does not substitute synthetic subprocess observations for that run.
2. **Complete compiled declaration inventory, later PR689 extensions, semantic alignment, all Layer A and CBZ-0:** excluded from these frozen inputs and explicitly unclaimed. The 31 selected arithmetic targets and existing proof sources are unchanged. Accepting this repair makes no broader completeness judgment.
3. **Source-only failure before entering `execute`:** it does not archive a previous receipt. This is an explicit documented boundary at `formal/README.md:99`, not a newly hidden success path. The actual command must succeed before the unchanged workflow reaches binding. Artifact existence alone remains insufficient evidence. The broader report sentence about failed reruns is read with this source-documented execution boundary.
4. **Historical bridge metadata naming the old `source_check`:** `tools/atlas_formal_source_bridge.py:136` retains the published historical `source_validators` string to reproduce immutable bridge bytes. Current execution actually uses the fixed identity reader; comments and new documentation state this accurately. Rewriting the historical record to describe today's reconstruction would violate byte-preservation requirements. That field is not treated as evidence that today's live validator ran on an old manifest.
5. **Archive retention beyond the local attempt directory:** this change preserves complete previous attempts on disk. The unchanged workflow uploads its current evidence directory, not the entire history tree; this review does not certify remote retention of every local prior attempt or PR347's separate retention work. The bounded implementation requirement is the intact preflight move, which is satisfied.
6. **The remaining 24 full-suite skips and full optimized regression:** skips remain open environmental/data coverage, and full optimized discovery was not required or claimed here. Focused optimized coverage includes all changed verifier and historical-consumer paths.

These boundaries were assessed for concrete interference with the requested repair; none supplies permission to ignore a known broken live attempt check, historical identity check or diagnostic rejection path. No such unresolved defect was found.

This review wrote only this report outside the checkout. It performed no source/index/HEAD/remote mutation, suite execution, Lean launch or subagent delegation.
