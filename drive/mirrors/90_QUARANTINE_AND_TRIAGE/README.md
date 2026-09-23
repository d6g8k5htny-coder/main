# Mirror of `90_QUARANTINE_AND_TRIAGE — NONAUTHORITATIVE` (three notices, nine reading copies, one index)

Drive root lane, folder id `1VvBQrv1M0E51socyGY-fjzlypEkc9IPz` (the Drive-wide triage inbox
`00_ACTIVE_GOVERNANCE_AND_DIRECTIVES / _TRIAGE_INBOX — DRIVE-WIDE`, as GP-MIG-156 names it),
191 items in the 2026-09-17 inventory: 51 files, 140 folders, 22 of the files carrying a digest.
Sub-lane folder ids named by the audit: `1zm0bScV66jPsfU8MymIsNRRn7CBqJRTG`
(`01_CONFIRMED_DUPLICATES`), `1cKcmxCj9jcFFCUBgUqpifDiOfJJyh17f`
(`02_DEFECTIVE_SCOPE_AND_REPLAY_HOLDS`), `14QpTsBRSsWUpmqGAOTSkmamsGGqzUg3X`
(`03_UNVERIFIED_AND_TRANSPORT`).

Added 2026-09-20.

## The controlling banners, verbatim

From `00_QUARANTINE_README_R17.md` (id `1EPaxMOccu0c7x8TFFL9OtStN7JiBEerC`, mirrored here
byte-exact):

> "Current operational rule: [OP-PROT-019-v1.1 / R17](…)."
>
> "- **01_CONFIRMED_DUPLICATES:** surplus copies with verified identity. Keep the selected
> original; links to the moved copy still resolve."
>
> "- **02_DEFECTIVE_SCOPE_AND_REPLAY_HOLDS:** exclusion records for defective or invalid
> certification chains. Frozen archive members remain in their original carrier and are excluded
> by path plus hash."
>
> "- **03_UNVERIFIED_AND_TRANSPORT:** uncertain or obsolete transport material. Presence here
> is not a mathematical falsity verdict."
>
> "- Older FROM_* collections and package-local quarantines retain historical evidence. Read
> their exact disposition before restoring anything."
>
> "Wrong work is quarantined only to the extent established by evidence. Distinct derivations,
> same-name scripts with different contents, and deliberate self-contained package dependencies
> must not be deleted as duplicates. Superseded valid work belongs in history."
>
> "Restoration requires a named correction, an exact new identity where content changed, scoped
> re-review and a recorded routing update. … Never silently remove a defect notice or overwrite a
> frozen source. The R17 release bundle contains the full reversible move log."
>
> "No file was permanently deleted in this overhaul."

From `00_H5_SCOPE_HOLD_R17.md` (id `1jvUHuHwgWqkw5klqthsfqfnsX5R5SSpG`, mirrored here byte-exact
under `02_DEFECTIVE_SCOPE_AND_REPLAY_HOLDS/`):

> "The 11 archive members listed in the central [Quarantine Index](…), keys Q-R17-H5-01 through
> Q-R17-H5-11, remain at their original immutable locations."
>
> "This is a hold on reliance on the affected certification chain, not a claim that every
> numerical conclusion is false. A repaired scalar kernel and one dominant-box replay do not
> certify the entire spatial cover."
>
> "Restore eligibility only after full affected-consumer replay, exact replacement identities and
> scoped review. Keep the frozen source bytes and historical manifests unchanged. Review Queue
> RV-H5-REPAIR is the next work item."

"Review Queue RV-H5-REPAIR is the next work item" is a work pointer in the source, not a status.
No hold is lifted, no exclusion discharged and no queue entered by this directory existing.

From the triage-inbox index (id `1zCAlbdkQ0YwySc8RT9UL4huJuZc0T8Kp`, mirrored here byte-exact):

> "The **one** place for any item that can't yet be confidently classified. Replaces the 10
> scattered "UNKNOWN ACTION — PEER REVIEW" trees."
>
> "| — | *(empty — add a row per item as it arrives)* | | | | |"

Its MASTER INDEX is still empty in the source.

## What was ported

| file | identity |
|---|---|
| `00_QUARANTINE_README_R17.md` (1,659 B) | **byte-exact**: SHA-256 `e63238b9…` and byte count equal the `drive/inventory.jsonl` row |
| `90_HISTORICAL_TRIAGE_MIGRATION_INDEX_2026-07-22.md` (1,127 B) | **byte-exact**: SHA-256 `999c1400…` equals the inventory row; stored under the **live** name (see the delta note below) |
| `02_DEFECTIVE_SCOPE_AND_REPLAY_HOLDS/00_H5_SCOPE_HOLD_R17.md` (1,100 B) | **byte-exact**: SHA-256 `988c324d…` equals the inventory row |
| nine `*.export.txt` files at this level | **reading copies (`exact:false`)** of the nine routing/receipt native Docs at the lane root: GP-MIG-154/155/156/157, LS-COR-012/013/015, LS-CTL-002, CL-AUD-229 |
| `_MANIFEST.jsonl` | a stored row or a `tree-only` index row for **every one of the 191 inventory items** in the lane: id, title, snapshot path, inventory bytes and digest where one exists, access status, and the `quarantine_index` / `file_catalog` rows that name the object |

Delta check: `drive/deltas/2026-09-18/PATH_CHANGES.jsonl` records that
`1zCAlbdkQ0YwySc8RT9UL4huJuZc0T8Kp` was renamed after the snapshot — inventory title
`00_READ ME + MASTER INDEX — single triage inbox (CL, 2026-07-22).md`, live title
`90_HISTORICAL_TRIAGE_MIGRATION_INDEX_2026-07-22.md`. It is stored under the live name and
`CHANGED_SINCE_SNAPSHOT.jsonl` marks it `IN_INVENTORY_MODIFIED` for the title change only: the
bytes fetched today still hash to the inventory digest, so the row is `exact:true`. No other
object ported here has a delta row. Every other lane item's row carries whatever the delta files
say about it.

The nine reading copies keep the Drive title plus `.export.txt`. In CL-AUD-229's file name a
`:` was replaced by ` —` so the path is safe in a checkout on any platform; the manifest row
carries the untouched title.

## What was deliberately not ported

* **The bytes of the quarantined objects themselves.** `1Y_3zFonLsFIAHP5KSkUfsJXqHZXIUAL2`
  (EXACT_DUPLICATE surplus copy), `1CHErajW7G2y8y7Mr0bKYMy2lJbA2HIAt` (SUPERSEDED_A1_ERRATUM,
  "False A1 finding is not restored as authority"), `1LooOZdyTyMMA_-kuCVifR3RNjWepkyfxbsH8ATP1lPQ`
  (TEMP TRANSPORT, "not a verified raw carrier") and the archive members named by
  Q-R17-H5-01..11 and Q-RN5-MOMENT-001..003. Storing an excluded payload is exactly what
  `tools/quarantine_check.py` invariant 3 fails on. Index rows only.
* **GP-DER-118-v1.7** (`1CLIFexNr-2IvM8S0sWx3WxkXC667vmc75tKRC9k8fhY`), a "NONOPERATIVE EMPTY
  SHELL — DO NOT USE" whose title mimics a theorem document and whose body says "Do not cite,
  review, hash, promote, or attach evidence to this shell." Never a reading copy; index row only.
* **HISTORICAL PREDECESSOR — CL-AUD-228-v1.0** (`1vzPLA5x5h2yTlzr4OBXyionAFD3uITIPfK_BQJPjUFc`),
  whose title says "CURRENT HOLD DISCHARGED". Index row only, and not as a status source in any
  form.
* **The July-2026 triage inbox items in the thirteen `FROM_*` buckets** (34 files): the stored
  index calls the inbox "The **one** place for any item that can't yet be confidently
  classified.", and its MASTER INDEX is still empty. Index rows only.
* The vault manifest transcription proposed as the audit's rank 5 was **not** done here:
  `drive/vault_tree.txt` is owned by another lane. `99_DO_NOT_OPEN`
  (`1VTiBaRlBvHGptiXqoEli4E5630nO7mNb`) was not opened, listed or queried in any way.

## Titles are not statuses

Two of the reading copies carry status words in their titles. CL-AUD-229's title says "CONFIRMED
EXACTLY" and LS-CTL-002's first line is its own supersession banner:

> "SUPERSEDED FOR ACTIVE EVALUATION — TARGET BODY AMEND REQUIRED — LS-COR-015-v1.0 — 2026-07-26"

What this repository carries about EC-019, EC-020, AD-009 and the P0.1 predicates is what the
register tabs under `registers/` carry, transcribed, and nothing else. A title, a reading copy or
a sentence inside one of these documents does not move a status here. The documents' own limits
are worth reading in their own words: CL-AUD-229 says "It corroborates ONE number… It is not a
review of GP-DATA-178"; LS-CTL-002 records "ELIGIBLE_P01_V18 = FALSE" and "HOLD / NOT YET
ELIGIBLE".

Three of the four GP-MIG receipts disclaim adjudication in so many words; they do not all use the
same sentence, the fourth does not use one at all, and none of the four closes on it. GP-MIG-154
and GP-MIG-155 carry, each under its own `SCOPE LIMIT` heading:

> "This receipt does not adjudicate, merge, delete, promote, close, or canonize any contained
> research or governance object. It only reconciles Drive routing, naming, and navigation."

GP-MIG-156's `SCOPE LIMIT` names a different set of objects:

> "This receipt does not adjudicate, merge, delete, promote, close, or canonize any contained
> mathematical, computational, or governance object. It changes only Drive routing and
> navigation."

GP-MIG-157 has no scope-limit section and no such sentence anywhere in it; the nearest thing it
carries is, under `GOVERNANCE AND MATHEMATICAL STATE`, "No ballot, Easy Closure, theorem, machine,
release, or mathematical status changed." All four end on their own end-marker line instead —
`END GP-MIG-154-v1.0` and its three counterparts — so none of these sentences is a receipt's last
word.

## What this does not establish

Mirroring is not review, replay, endorsement, restoration or promotion. A digest match proves
that the bytes on disk are the bytes the 2026-09-17 inventory declared for that Drive id — it
proves nothing about whether the document is correct, current or complete. A reading copy is not
the object: a native Google Doc has no inventory digest, its `.export.txt` is a text rendering
fetched today, its SHA-256 here is of that export as first computed in this repository, and a PDF
rendering of anything is not a frozen body either. An index row is metadata, not a mirror, and
presence in the index is not a mathematical falsity verdict. Nothing under this lane may be cited
as evidence (`CLAUDE.md` rule 10, and the lane's own `NONAUTHORITATIVE` banner). The 22 register
exclusions stand exactly as `registers/json/quarantine_index.json` writes them, the 11 H5 members
stay held, the RN5 DEFECTIVE_SCOPE holds stay, the A1 erratum is not restored as authority, and no
gate, hold, review or independence credit moves. The five validity premises of Theorem D1
v2.2(2) stay OPEN, `D3-LEMMA-RN-UNIF` is not closed, nothing here composes the 2D tracks with the
3D lifetime track, and no original prize problem is solved. Nothing stored here is imported or
executed, and nothing here feeds a claim, a bound, a carrier or a grade. It is read, though:
`tools/verify_manifests.py` runs from the repository root with no path argument, so every CI run
opens each stored file in this directory and re-hashes it against its manifest, and
`tools/mirror_quotes_check.py` reads these same bytes as the corpus the quotations above are
checked against. Both only read, hash and compare.
