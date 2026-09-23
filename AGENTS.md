# AGENTS.md

## Current owner authorization — read before historical workflow wording

[OP-AUTONOMY-20260923-v1.0](governance/OP-AUTONOMY-20260923-v1.0.md)
is the existing standing authorization for every model until Dylan changes it.
Source: Drive `1N-afVwLVTloSB79GtEK1Fhfx7vdI4_b7`, also linked by the live
Research Home. Older owner-permission waits below or in historical procedures
are superseded; do the actual work without asking Dylan to authorize it again.
This does not fabricate test results, scientific acceptance or independence.

[PR2 post-merge handoff](docs/PR2_POST_MERGE_HANDOFF_20260923.md) records the
completed migration merge, remaining integration work and a recovery procedure.
It is not the original missing STATUS packet or an independent scientific review.

This file points; it does not legislate. The rules for working in this
repository are in [`CLAUDE.md`](CLAUDE.md), and they bind every agent here,
whatever its provider or name.

## Where control lives

- **Drive is the governing record.** The current entry point is the Research
  Home (Drive id `180yfvocozAaFRxf7tY8CDrobnpi17Sv-UkQGBBWCiD8`); the current
  operational policy is OP-PROT-019-v1.1 / R17 (Drive id
  `1hBph5Fpxd5dVrolNkvzU8nb7xpxUiUdc`, reading copy at
  `governance/protocols/OP-PROT-019-v1.1_R17.md`). The R17 Work Events register
  is the claim/disposition log; this repository keeps no second one.
- **GitHub is the execution workspace**: code, tests, branches, pull requests,
  receipts. A merge is not a promotion. A green run is a run.

## The execution contract

- Mirrored byte-exact under `drive/deltas/2026-09-18/DG-EXEC-20260918-49291487/`
  with the disposition **PROPOSED EXECUTION CONTRACT / NOT DEPLOYED**. Nothing
  in this repository changes that disposition.
- Its repository-side half is [`engine/bridge/`](engine/bridge/README.md):
  work orders, run receipts, `tools/bridge_check.py`. Under the contract,
  **work proceeds only under a work order authorized at the owner boundary**
  (an owner-side Drive record, transcribed as `verification_status:
  VERIFIED_BY_OWNER_BOUNDARY`; this repository pins that record, cannot
  verify it, and confers nothing). None exists: `engine/bridge/orders/` is
  empty. An unverified order is PREPARED-only.
- **Nothing there is enforcement.** No branch protection, credential, access
  audit or Drive return route exists because that code exists.

## What no agent does here

- **No status moves.** No claim, premise, obligation, gate or grade is
  promoted, closed, discharged or reclassified by any commit, receipt, test or
  review. Only an operator decision under `governance/` does that.
- **Never open `99_DO_NOT_OPEN`** (Drive id `1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`)
  for authority, proofs, certificates or "latest" status. Metadata only.
- Never edit exported source data or a frozen body in place; never cite
  `legacy/`, `quarantine/` or the vault as evidence.

Dylan Roy is the single final authority for canonical promotion, external
release, permanent deletion and machine-root replacement.
