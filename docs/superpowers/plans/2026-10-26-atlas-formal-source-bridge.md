# Atlas formal source connection Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: use superpowers:executing-plans.

Goal: expose one source-verified, explicitly limited arithmetic companion.
Architecture: deterministic offline evidence assembler, conservative pure
projection, native inspector renderer. Existing admission/state/GPU untouched.
Tech stack: Python stdlib and current native JS modules/brand CSS; no new library.
Spec: docs/superpowers/specs/2026-10-26-atlas-formal-source-bridge-design.md.

## Global constraints

- Claim math.side24-coefficient; target UniversalLaw.Side24.pairing_sum_six.
- Formal source3b39c42d6edcc98ae165270be346f0e31c57f81d; manifest8dfcfe4e…
  and original run40246431034/artifact12618189537/receipt1ae8341c….
- Reuse formal_gate_check.source_check and required_formal_check.bind_receipt.
- Preserve original receipt/type bytes, all records and missing mappings.
- Alignment pending until an actual qualified immutable record is validated;
  scientific effect NONE. No workflow/validator/proof/source-admission changes.

## Review focus

Other ID with equal name/fingerprint, changed scope or extra record fields,
wrong source blob, changed manifest, forged/stale receipt/logs, and refused
selection must never expose this connection. Test each at the appropriate layer.
Long/untrusted text stays inert and wraps. Distinguish observer/source identities.

### Task 1: pinned assembly and source guard

Create tools/atlas_formal_evidence_build.py and
tests/test_atlas_formal_evidence_build.py. build_connection(root,snapshot,proof,
receipt_dir) returns a JSON-compatible record or raises ValueError. Validate
all current formal sources through the existing source gate; bind the original
receipt/logs to its real3b39/run/attempt; independently match proof SHA/blob/size
to the claim inventory and manifest source. Capture exact declaration/type and
scope, with no mathematical generalization. Write generated data module and
unchanged receipt/type assets under docs/site/evidence/side24-main-3b39/.

- [ ] Test successful real inputs and wrong proof/commit/log/target/manifest.
- [ ] Observe intended failures; implement and run normal/optimized tests.
- [ ] Reproduce generated output byte-for-byte; preserve qualification inputs.

Create atlas-formal-evidence-model.mjs and its focused Node tests.
formalConnection(snapshot,selectedId,evidence) returns the source connection or
null, requiring exact ID/record/source and manifest identity.

- [ ] Test exact positive identity, reordered fields, wrong ID/bytes/blob/scope,
  extra record fields, absent inventory and stale formal manifest.
- [ ] Observe RED, implement minimal guard, verify PASS.

### Task 2: inspector and real reader journeys

Create atlas-formal-evidence.mjs; modify atlas-claim-summary.mjs to delegate only
formal rendering. Pass selected ID in atlas.mjs; update CSS/README as needed.
Keep existing formal IDs/disclosures and the complete seven-stage fallback.

- [ ] Add real-controller or browser RED for the mapped declaration and receipt,
  with all other claims/unavailable identities still explicitly unmapped.
- [ ] Render exact closed declaration, type, hypotheses boundary, source/run and
  separate alignment; no coefficient proof badge.
- [ ] Test keyboard, Graph/List/3D, refusal/reset, links/share/history and mobile.
- [ ] Compare sealed baseline with the same fixture; inspect captures and timing.

### Task 3: bounded release

- [ ] Stage new assets, run canonical release/check and prescribed site commands.
- [ ] Run meaningful builder/model/controller/browser, actual 3D/refusal checks.
- [ ] Record source identities, visual evidence, performance and limitations.
- [ ] Separate reviewer and integrator, actual Ready/current checks, expected-head
  merge under repository rules; preserve exact source/head/live preimages first.
- [ ] Verify actual deployment bytes/journeys, archive evidence privately with
  independent readback, and release only this author's finite claim.
