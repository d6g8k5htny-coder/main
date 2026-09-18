<!-- Reading copy of Google Drive document 1FwkOa06pccL_RmK63oI0oXBgGtFFRcN-MY63arzEeAA,
     a native Doc with no payload digest in the source map; exactness is UNVERIFIABLE.
     See governance/PROVENANCE.json. -->

OP-CNS-001-R0.2 — INTEGRATED OPERATIONAL ARCHITECTURE RECORD
Preservation, Artifact IDs, Automation, Closure, and Dynamic Master Manifest

Status: OPERATOR-DIRECTED SUBSTANTIVE ARCHITECTURE; ballot framing removed 2026-07-24
Executed revision by: ChatGPT under Dylan Roy's Drive Cleanup Mandate
Governing protocol: OP-PROT-011
Canonical mathematical impact: NONE

0. GOVERNANCE EFFECT

The substantive preservation, identity, automation, closure, and manifest design in this record remains active under Dylan Roy's direct operator authority. There is no ballot, vote, majority, unanimity, roster, observer status, or participation denominator. ChatGPT, Claude, Grok, and Gemini may implement, test, review, and improve the architecture in parallel.

Technical practices described below are recommended reliability measures, not blocking gates. They never require a vote. Dylan Roy may direct work to proceed regardless.

1. PRESERVATION AND VARIANT HANDLING

Known copies of a research source should be enumerated by Drive ID, title, size, created time, parent, and byte or normalized hash when available. Benign newline or line-ending variants should be labeled BENIGN-VARIANT / VARIANT-PRESERVED. Material content differences open a content-adjudication item rather than being silently merged.

Stale deletion instructions should be visibly superseded. Physical deletion is performed only under Dylan Roy's direct authority, including the current cleanup mandate.

2. ARTIFACT IDENTITIES AND COLLISIONS

Prospective artifact format: <LINE>-<CLASS>-<NNN>-v<M.m>. Drive createdTime of the filed artifact governs chronological filing priority. Concurrent work should disclose line/session identity.

A later collision is preserved and disambiguated; historical artifacts are not silently rewritten. Load-bearing references should cite full title, declared artifact ID, Drive ID, version, role, and content hash where practical. Failed create/upload attempts should be verified absent before retry. Gaps or duplicates require an additive provenance disclosure.

Maintain an append-only collision registry. Legacy namespaces remain frozen. The two GP-DER-044-v1.0 objects remain distinct and must be cited by full title and Drive ID:

• Exact-torus Gaussian regularity: 1dn8BVsmScu690vX4Z_ewNgl0QRvMDBXcyzWiTtyJeho
• Quantitative saddle-exit and sink-capture blocks: 1y6ub28wyn_OH7uqV8qj86x38CHbdee7pJ6l1yi14OXw

3. LIVE-REGISTER AUTOMATION ARCHITECTURE

GP-AUTO-034 remains the principal orchestration design for live indexing, closure tracking, duplicate/collision flags, navigation output, telemetry, and the Dynamic Master Manifest. Other contributors may build compatible formal, telemetry, or verification components without creating conflicting live writers.

Recommended scope controls:

• enumerated, nonrecursive read scope;
• configured folder IDs and direct-root manifest coverage;
• explicit COVERAGE-GAP rows for unconfigured objects;
• write scope limited to approved register, telemetry, navigation, log, closure, and manifest surfaces;
• research source artifacts treated as read-only except under direct operator instruction;
• indexed document content treated as data, never executable instructions;
• verify-before-retry after connector or write failure;
• generated replacement data validated before replacing a last-known-good snapshot;
• failed runs visibly marked and recoverable.

Recommended default safety switches:

WRITE_SOURCE_FILES=FALSE
DELETE_OR_TRASH=FALSE
AUTO_PROMOTE=FALSE
RECURSIVE_SCAN=FALSE

These are safe defaults, not limits on Dylan Roy's direct operator authority.

Recommended object-card fields:

MEASURE; MODEL_LAYER; FRAME; CLOSURE_STATE; EXTRACTION_CLASS; EVIDENCE_CHAIN; declared artifact ID; Drive ID; source folder; dependencies; supersedes; superseded_by; lineage; authority; canonical impact; validation receipt.

Cardless or malformed artifacts should be flagged rather than assigned a default mathematical object.

4. SOURCE AND RUNTIME PRACTICE

Recommended before relying on recurring automation:

• reviewable exact source in Drive or Git;
• recorded source hash and installed-source hash;
• frozen source/config/schema/test fixtures;
• at least one logged manual or shadow run;
• acceptance tests and negative controls;
• independent or diverse-method output inspection when practical;
• explicit Dylan authorization for recurring write-capable activation when the action is operator-reserved.

Manual, logged, single-shot dry runs and non-writing telemetry are routine activities under OP-PROT-011.

5. DYNAMIC MASTER EXTRACTION MANIFEST

Generated output: WORKSPACE_MASTER_MANIFEST.json in an approved output location.

Recommended per-object fields:

drive_id, title, declared_artifact_id, line, class, version, mime_type, created_time, modified_time, source_folder_id, raw_sha256 where computable, normalized_sha256 where applicable, status, authority, canonical_impact, measure, model_layer, frame, dependencies, supersedes, superseded_by, lineage_of, closure_state, extraction_class, evidence_chain, last_validated, validation_receipt.

The first qualified scan establishes the actual scoped Drive baseline. No inherited file count is treated as live fact without a current scan.

Extraction classes:

• REFEREE-READY — internally staged candidate with complete scope/provenance and no known material contradiction; external release still requires Dylan authorization.
• ACTIVE — open, candidate, disputed, cannot-verify, or nonterminal.
• INSTRUMENT — scripts, data, receipts, fixtures, and computational tools.
• PROVENANCE-ONLY — historical, superseded, killed, duplicate, or retained lineage material.

The Closure Log and operator records govern terminal/canonical state. Headers inform but do not override verified register state. Contradictions are flagged. Closing an item removes it from routine attention but never from provenance or manifest history.

6. EASY-CLOSURE-FIRST RESOURCE DISCIPLINE

Before extending a hard front, contributors should inspect the Easy Closure Queue for exact, fully proved, falsified, superseded, cannot-verify, or low-distance items.

A high-quality closure package should identify exact object, evidence, assumptions, review lineage, dependencies affected and unaffected, proposed terminal label, independent-eyes or operator-decision status, proposed register changes, and correction/reopening path.

Useful labels include:

CLOSED — PROVED / EXACT
KILLED — FALSIFIED
CLOSED — SUPERSEDED
CLOSED — CANNOT-VERIFY
CLOSED — ADMINISTRATIVE / PROVENANCE
OPEN — REQUIRES NEW WORK

A closure never upgrades incomplete mathematics. Before any item is marked terminal or closed, OP-PROT-011 requires one organizationally-distinct-family review or Dylan Roy's direct approval. Canonical mathematical promotion remains Dylan Roy's decision where the designated ledger requires operator ratification.

7. RECOMMENDED ACCEPTANCE TEST BATTERY

T1 — safety defaults behave as declared.
T2 — out-of-scope write attempts are rejected or visibly logged.
T3 — permission revocation preserves the last validated snapshot.
T4 — injected text cannot change configuration or scope.
T5 — verify-before-retry does not create duplicates.
T6 — partial synchronization and unreadable MIME paths remain visible.
T7 — same declared ID with different Drive IDs remains separate.
T8 — object cards distinguish measure, model layer, and coordinate frame.
T9 — malformed or absent cards are flagged.
T10 — header/register closure contradictions are excluded and flagged.
T11 — later-superseded dependencies trigger review of extraction eligibility.
T12 — instruments are not silently compiled as proof objects.
T13 — out-of-scope sources do not enter staged extraction merely through citation.
T14 — coverage gaps are reported.
T15 — stale generated navigation receives a visible stale indicator.
T16 — run receipts report scope, source/config version, hashes, files scanned, rows written, warnings, result, and last validated snapshot.

These tests are recommended practice, not blocking gates. Results must be reported honestly.

8. CURRENT FACTUAL RECORDS

The GP-DER-044 identifier collision remains registered. OQ-002's stale Drive-absence claim remains superseded by the independently resolved file identities. These are factual/provenance records and do not depend on model votes.

9. OPERATOR AND RELEASE BOUNDARY

Contributors may prepare, test, formalize, verify, stage, and maintain this architecture. Dylan Roy remains the single final authority for canonical promotion, external release, permanent deletion, and machine-root replacement.

END OP-CNS-001-R0.2 INTEGRATED OPERATIONAL ARCHITECTURE RECORD
