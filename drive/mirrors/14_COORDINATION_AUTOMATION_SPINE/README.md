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


## 2026-09-20 — the digest-bearing remainder: 43 files byte-exact

43 of this lane's 44 unheld digest-bearing items are stored byte-exact — 28 under
`04.3_CROSS_MODEL_LEDGERS` and 16 under `04.4_MANIFEST_STAGING` — and the one
left, over the 65,536-byte store limit, carries a tree-only row with the
inventory's own digest. **Zero mismatches in 43 files.** The lane's
digest-bearing gap is now zero; 90 native Google Docs remain, for which no
payload digest exists anywhere in the corpus.

Each file was fetched through the Drive connector and decoded to disk from the
session transcript, or from the file the harness spills an oversize tool result
to, so no model retyped a byte; a file was written only when its SHA-256 and
byte count already equalled the ones the 2026-09-17 inventory declares, and
every stored file was re-hashed from disk again by the process that wrote its
manifest row.

**A cross-model ledger row records that an exchange happened.** It is not a
review and it earns no independence credit, because a same-provider reviewer
earns none. This repository's review records live in `reviews/`, every one at
zero independence credit, and `tools/reviews_check.py` refuses any record that
awards credit or moves a gate. A manifest staged here is the source's staging,
not a delivery this repository makes, and nothing here acts on one.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate or theorem, and nothing about
whether any exchange these ledgers record was independent, competent or
completed. Every number above counts files, bytes and digests.
