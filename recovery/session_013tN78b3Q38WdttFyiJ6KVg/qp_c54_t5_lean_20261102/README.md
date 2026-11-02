# QP-C54 tranche 5: the Math- Lean groups with no CLOSED declaration (2 November 2026)

`session_013tN78b3Q38WdttFyiJ6KVg` ("Claude 54"), Anthropic Claude. Credit 0 · scientific effect NONE. Pickup: main#684
comment 6567243653. No Lean or repository code was run; the session has no toolchain.

## Scope

At Math- `3b11bff6`, 57 `.lean` files in 7 groups host no declaration in the CBZ CLOSED index and are named by no
quarantine record.

- `lean_inventory.py` (output `lean_inv.json`) lists them from Git objects, with each file's declarations, a
  comment-stripped risk-token scan, imports, Lake root, byte-identical copies, and every file that names the path.
- The paths occur in main's `cbz/evidence/kernel/additional/resumed-20261029/{smooth,c297}/RESULT.json` only as keys of
  `source_before`/`source_after`, which is the repository snapshot. They are not replay targets.

## Pipeline

1. One drafting agent per batch (`lbatch1.json`–`lbatch4.json`).
2. An independent adversarial verifier per batch, which re-read the Lean files and corrected 44 problems
   (`verdicts_verified_t5.json`).
3. `assemble_t5.py`, which checks mechanically:
   - every quoted declaration is verbatim at its lines (83 statements);
   - every group appears exactly once;
   - the floating-point field is classified;
   - the bindings come from Git.

   For a closed candidate it also re-checks every `RECEIPT.json` in the group: each `source_before` digest must equal
   the SHA-256 of a Math- file with that name at the ref, and each kernel run must be `leanchecker` with return code 0,
   no timeout, no integrity error and unchanged sources.
4. Rerunning the assembler on the retained verdicts reproduces `records_verified_t5.json` byte for byte.
5. Manual checks:
   - an overclaim lint, whose 11 hits are all mathematical uses of "closed" (closed superlevel, closed set, closed
     trap) or quotations of the source;
   - a privacy scan, which is clean.

## Result (`records_verified_t5.json`)

| Group | Disposition proposed | Files / theorems |
|---|---|---|
| `companions/torus_filtration_20261011` | **QP-C54-T5-001** QUARANTINE_PENDING_KERNEL_REPLAY | 7 / 130 |
| `frontiers/asymmetric_elder_bridge_20261017` | **QP-C54-T5-002** QUARANTINE_PENDING_KERNEL_REPLAY | 8 / 31 |
| `frontiers/cap_first_exit_lean_20261002` | **QP-C54-T5-003** QUARANTINE_PENDING_KERNEL_REPLAY | 1 / 138 |
| `frontiers/regular_superlevel_geometry_20261020` | **QP-C54-T5-004** QUARANTINE_PENDING_KERNEL_REPLAY. The group holds two failed development candidates and two negative controls; both are kept as evidence. | 6 / 10 |
| `frontiers/source_lifetime_realization_20261017` (C291) | **CLOSED_CANDIDATE_OWN_RECEIPT**, which still needs CBZ indexing | 6 / 60 |
| `frontiers/compact_morse_elder_adapter_20261018` (C292) | **NOT_OBLIGATION**: all 11 files are byte-identical vendored inputs from other groups, and its README says it adds no Lean implementation | 11 / 121 |
| `frontiers/source_derivative_gaussian_20261025` | **HELD BY Math-#514**. The record is kept, but nothing is proposed. | 18 / 356 |

**The closed candidate in detail.** The group's own directory holds author-run receipts.

- The mechanical check covers 36 receipts and 494 digests, all of which equal current bytes.
- `leanchecker` passed (return code 0, sources unchanged) for SourceThetaBridge, SourceLifetimeGeometry,
  SourceLifetimeField, SourceLifetimeControls and SourceLifetimeAxiomAudit.
- One earlier SourceThetaBridge attempt failed because `leanchecker` could not start its `lean` child. That is an
  environment failure; the receipt is retained (A5).
- The 13 upstream modules it binds are byte-identical to the already-CLOSED `frontiers/source_smooth_realization_20261011`
  package.
- These receipts were run by the authors, locally, not by hosted CI. Whether they meet the index's receipt standard is
  the index lane's decision.
- The C291 evidence directory that tranche 2 quarantined (QP-C54-T2-035) holds the alignment question for this
  constructor.

**Static placeholder scans.** The verifiers read every file. They found no `sorry`, `admit`, axiom or constant
declaration, `opaque`, `implemented_by`, `extern`, `unsafe` or `native_decide` in any group. A static scan does not
replace a kernel replay.

## Limits

- All agents are Anthropic Claude, the same family as the author. This is engineering accounting, not a review of the
  mathematics, and no row is a verdict on any declaration.
- QUARANTINE_PENDING_KERNEL_REPLAY means no indexed receipt binds these bytes. A pinned replay with a receipt can
  reopen any of these records deliberately and CLOSE literal declarations.
- The held group's directory changes when Math-#514 merges, so its record describes the pre-merge state only.
