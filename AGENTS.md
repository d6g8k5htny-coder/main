# AGENTS.md

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

## Support withdrawal and scheduling — 2026-09-21 operational addendum

Read [OP-WITHDRAWAL-20260921-v1.0](governance/withdrawal/PROTOCOL.md) before
proposing abandonment, retiring a proof route, changing admissibility or resuming
affected work. Revalidate current R17 Work Events and exact source/head identities;
the historical bridge descriptions above are not a substitute for live records.
Drive delivery folder: `1VfuXdkyIi0VIsmz42WS6ALEhlc2qneM_`.
Closing a workstream is not refuting a claim. Withdrawing a warrant cannot erase
its parent's premises. Preserve alternative proofs and require exact survivor
warrants. The executable checker is an unreviewed loss-only pilot, not a verdict
engine. Existing scientific records are NOT_MIGRATED. Apply the safeguards to new
work; do not bulk-convert historical statuses. Use the existing R17 event log.

## Governance rollout compatibility — 2026-09-21

Drive governance is being re-audited for premature or overly broad activation. Read the
Governance Rollout Audit (Drive id `1QQOu8r9FBnYZL3XxURSHqsu0mkqcK5P466dXUtpklu8`) before
interpreting a newly installed rule as universally active. Default adoption is
`PROPOSED -> SHADOW -> PILOT -> REVIEWED -> ACTIVE_SCOPED -> ACTIVE_GENERAL`.
Immediate containment is reserved for narrow integrity/safety hazards. Words such as
`mandatory`, `automatic`, `superseded`, `complete`, `abandoned`, `retired`, and `HOLD`
do not cross object types: an operational state is not a scientific verdict. Preserve
predecessor rules as compatibility evidence while affected consumers are audited. Do not
pause unrelated research solely because the governance audit is open.
