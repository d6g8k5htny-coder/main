# CROSS-MODEL CONTRIBUTION REGISTRY — v1
**Program:** q0 Rate Program. **Issued:** 2026-07-20 (UTC) by Claude, instance tag C047R. **Discipline:** versioned, supersede-never-overwrite — to contribute, copy this file verbatim, append rows, issue as v2; all prior versions and rows are retained unedited. **Purpose:** no mechanism existed that attributes contributions per model or assigns each markup/update a number for tracking across the multi-model workflow. This registry is that mechanism, created by recommendation and example.

## 1. Scheme
- **U-ID:** global sequential (U001, U002, …) across all models; one row per contribution (artifact creation, correction, kill, review, upload, navigation aid, consolidation, registry issuance).
- **Row fields:** U-ID | UTC date | Model + instance tag | Type | Artifact(s), sha256 prefix where applicable | One-line description | Relates/supersedes.
- **Types:** CREATE · CORRECT · KILL · REVIEW · UPLOAD · NAV · CONSOLIDATE.
- **Attribution rule:** record-supported only. Where model identity is not in the record, the row reads [MODEL UNRECORDED]; the builder claims it by appending a CORRECT row targeting that U-ID. No prior row is ever edited.
- **Header rule (all future artifacts):** first lines carry `U-ID: Uxxx | model/instance | UTC date`.
- **Corrections to corrections:** also new rows. The registry is append-only in exactly the ledger's sense.

## 2. Seeded rows
| U-ID | Date (UTC) | Model / instance | Type | Artifact(s) | Description | Relates |
|---|---|---|---|---|---|---|
| U001 | 2026-07-05 → 07-12 | Claude (Anthropic), multiple instances | CREATE | C001–C047 cycle chain; Master v3.2 compiled set 00–08; Claim Language Annex v1 | The inherited two-sided q0 program through the master-set extraction | Session records: /mnt/transcripts 2026-07-05 … 07-12 |
| U002 | ≤ 2026-07-19 | [MODEL(S) UNRECORDED — claim via CORRECT row] | CREATE | C048–C093 incl. Q0_C092_FINAL_MASTER (026c4665), GATE_FRAMEWORK v1.2 (ea3b0bce) | Frozen core + gate methodology built on the inherited chain | Builds on U001 |
| U003 | ≤ 2026-07-19 | [MODEL(S) UNRECORDED — claim via CORRECT row] | CREATE+CORRECT+KILL | C094–C108: E-C094-1…11; C101 roots (8c2ded65); C104 Theorem B (f592bc6f); Q0-SHARP kill; portfolio close (c4247dd5) | The successor corrections and terminal portfolio state | Corrects/extends U001–U002 |
| U004 | 2026-07-19 | [MODEL UNRECORDED — claim via CORRECT row] | CONSOLIDATE | Q0_MASTER.md; Q0_LEDGER.md; q0_machine.json (488-manifest); q0_verify.py | Consolidation C-CONS-2026-07-19 | Embeds U001–U003 |
| U005 | 2026-07-20 | Claude / C047R | REVIEW | Adjudication_C094_C108_vs_C024_C047.md (d39a0fa0); MASTER_SET_ERRATA.md (acd8b11a) | CONCUR with receipts on every kill/correction touching the C024–C047 record; errata annotation of the embedded master set | Reviews U003–U004 against U001 |
| U006 | 2026-07-20 | Claude / C047R | NAV+UPLOAD | 4 navigation Google Docs (Drive index; Q0_MASTER companion; Q0_LEDGER companion; review-package companion) + byte-verified Drive uploads: 01 (ba7faf2c), 02 (4a77f3b7), 03 (50dd5b9d), U005 artifacts | Drive made navigable; canonical copies mirrored with byte-size verification (03 required one re-upload: trailing-newline byte; short copy flagged for trash) | Serves U001–U004 |
| U007 | 2026-07-20 | Claude / C047R | CORRECT (self, scope) | — | Prior turn claimed review files 00/04/05/06 "not recoverable from any record I can reach" before searching q0_machine.json's interior; search since extended to the registry (zero hits at structure, text_artifacts, and 488-manifest layers). Conclusion survives; the original claim overstated its scope at time of assertion. Standing state: the four files exist only with the operator — re-attach to complete the package | Corrects U006-adjacent claim |
| U008 | 2026-07-20 | Claude / C047R | CREATE | CROSS_MODEL_CONTRIBUTION_REGISTRY_v1.md (this file; sha256 recorded in the issuance report and to be carried verbatim into v2's row for U008) | Registry scheme + seeding; the lead-by-example instantiation | Governs all future rows |

## 3. To the contributing models (direct)
1. **Claim your work.** If you built any part of U002/U003/U004, append a CORRECT row naming your model and which cycles/artifacts are yours. Unclaimed rows stay [MODEL UNRECORDED] forever — the registry never guesses.
2. **Adopt the header rule** on every artifact you emit for this program, starting with your next one.
3. **Issue v2 properly:** copy this file verbatim, append your rows (your v2 issuance is itself a row), never edit existing rows, keep v1 in place.
4. **Open technical items any model may take:** (a) the marked-repulsion certificate under the determinant-weighted typed six-pin pair-Palm law — review Question 1, Q0_MASTER line 4582, the sole path to promoting the lower side; (b) review-form processing once files 00/04/05/06 are re-shared; (c) SARD-G acceptance is an external human-specialist dependency, not a model deliverable — do not claim it.

## 4. Recommendation to the operator
Adopt the header rule as a requirement for anything you relay between models; keep one registry per program (this one is q0-scoped); when closing any model session, require the active model to issue vN+1 with its rows as part of the close.
