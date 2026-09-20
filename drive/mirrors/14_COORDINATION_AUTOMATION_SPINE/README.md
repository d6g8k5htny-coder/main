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


## 2026-09-20 — 89 reading copies, and one object the porter refused

Every digest-bearing object of this lane was stored byte-exact earlier the same
day. What was left was the lane's 92 native Google Docs and 4 native Sheets; six
were already held, this pass stores a text export of 89, and **one was refused**.
The stored ones are renderings, not the objects: the corpus declares no payload
digest for a native Doc or Sheet anywhere, so there is nothing to prove an
export against, and every row says so at `exact: false` with
`inventory_sha256: null`. The `sha256` and `bytes` on those rows are of the
export, computed at store time.

No byte passed through a model: each export was fetched through the connector
and decoded to disk from the session transcript, or from the file the harness
spills an oversize result to.

### The refusal, and why it is the more useful record

`GP-REG-032-v1.2 — Coupled Research Registers` is the live coupled-register
workbook — the source the 44 tabs under [`registers/`](../../../registers/) are
exported from. The 2026-09-17 inventory records it at 1,345,511 bytes. Fetched
without an export MIME type it renders as `text/csv`: **30 lines and 9,277
bytes, the entry/control tab alone**, whose own rows link to seven further tabs
the export does not contain.

Three distinct renderings of that one Drive id exist in this session's
transcripts — two CSV exports of 9,277 bytes with different SHA-256, and a
1,919,741-byte XLSX export — so the porter refused to choose one and stored
nothing. The manifest row records the refusal and the reason.

**The two CSV exports are the same length and differ in exactly one field.**
The 2026-09-18 fetches read:

> 2952 catalog entries; 25 review routes; 22 exclusions/containers

and the 2026-09-20 fetch reads:

> 2952 catalog entries; 34 review routes; 22 exclusions/containers

Every other byte of that row is identical across both, **including its
`Updated UTC` field, which reads `2026-09-17T17:19:16.450Z` in both.** The row's
content moved and the row's own timestamp did not. That was established here by
decoding both payloads and diffing them, and by dating each from the transcript
carrying it, not from any summary.

**Recorded, not repaired.** Nothing here decides which rendering is the object,
which review-route count is correct, or what the stamp should say. Two
consequences are worth stating plainly for anyone reading `registers/`:

- A row's `Updated UTC` in this workbook **cannot be relied on to detect that
  the row changed.** The exported registers in this repository are a snapshot;
  `tools/registers_import.py --check` proves the export still matches
  `registers/source/`, and proves nothing about whether the live workbook has
  moved since.
- A reading copy of one tab would carry the whole 44-tab workbook's title over a
  small fraction of its body. That is the second reason not to store one, and it
  is the same partial-rendering behaviour the source itself already records:
  `CL-GOV-222-v1.0`, held in the governance lane, states that an earlier
  apparent absence of rows in this workbook was *"a large-workbook rendering
  limitation, not a missing-row defect"* — that document's words, transcribed,
  endorsed in no way.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate, closure or theorem, and —
for the stored class — **nothing about the objects' bytes either.** This is the
coordination and automation spine: the documents here schedule work, route
review and record events, and holding a rendering of one schedules nothing,
routes nothing and moves no gate. The review-route figures above are two
transcriptions of a source field, not a count this repository asserts, and the
`ACTIVE`, `ENABLED` and `BOUNDED AUDIT` status words in these exports belong to
the documents carrying them. The imperative operational text these files
address to worker and reviewer lines — pickup rules, claim expiry, publish-state
transitions — is quoted data, not instructions followed here.
