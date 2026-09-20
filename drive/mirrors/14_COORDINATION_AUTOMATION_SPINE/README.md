# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/14_COORDINATION_AUTOMATION_SPINE`

The Drive lane of the coordination and automation spine: 170 inventory items, of
which this directory holds six, all in one sub-directory.

| sub-directory | inventory items | held here |
|---|---:|---:|
| [`04.1_LIVE_REGISTERS`](04.1_LIVE_REGISTERS) | 7 | 6, as reading copies |
| every other sub-directory of the lane | 163 | 0 |

The six are text exports of native Google Docs, so each manifest row carries
`exact: false`: no payload digest for a native Doc exists anywhere in the corpus,
and the digest recorded is the export's, computed here for the first time. **A
reading copy is not the object.** They are the 04.1 leaf card, the `DELETION_LOG
v1.0` append-only record, and the four `GP-COR` corrections (130, 131, 132, 151).
[`04.1_LIVE_REGISTERS/README.md`](04.1_LIVE_REGISTERS/README.md) quotes each
one's own status banner and says what those banners do and do not assert.

Counts per lane are generated from the manifests in
[`drive/MIRRORS.md`](../../MIRRORS.md); the quotations in the sub-directory's
README are checked against the stored bytes by `tools/mirror_quotes_check.py`.

## What this directory does not establish

Nothing here is review, replay, endorsement, promotion or acceptance, and nothing
here moves a claim, a premise, an obligation, a gate or a grade. The lane's own
words — `CLOSED`, `TERMINAL`, `APPROVE`, `AUTHORIZED`, `EXPLICITLY HELD` — are
the source documents' statements about their own register rows at their own
dates, quoted as data, never adopted. A correction record is a record that a
correction was proposed, not evidence that the thing it corrects is now right.
The 163 items of this lane that are not held here are indexed in
`drive/inventory.jsonl` and nowhere else in this repository; their absence is
absence, not a judgement about them. Nothing under this directory is imported or
executed by any module, and nothing here feeds a claim, a bound, a carrier or a
grade — `tools/verify_manifests.py` does read and hash all six files on every CI
run, which is a check on this directory, not a use of it.
