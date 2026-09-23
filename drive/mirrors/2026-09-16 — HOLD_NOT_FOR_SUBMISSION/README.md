# Mirror of `01_ACTIVE_RESEARCH_PACKAGES/2026-09-16 — HOLD_NOT_FOR_SUBMISSION` (one carrier)

Drive lane of the 2026-09-16/17 RN3, RN5 and CLOSE bundles (79 inventory items).
The lane's name is the source's status word for everything in it: **HOLD, not
for submission.** This directory holds one of its carriers byte-exact:

| file | identity |
|---|---|
| `CLOSE-20260917-b9c2_PROOFS_CODE_AND_VERIFICATION.zip` (1,190,232 B) | **byte-exact**: SHA-256 `eb5fc2c2…` and byte count equal the `drive/inventory.jsonl` row (id `1WZfhLuZBEzvdUUkQw7v5JubmAiI1W4gt`); 183 members (`PAYLOAD_MANIFEST.json`, `closure_round2/`, `intake/`, `output/`: 64 json, 55 py, 27 md, 26 txt, 9 log, 1 csv, 1 sha256), every one indexed in `drive/source_map/Archive_Members.csv` under this carrier id |

One `_MANIFEST.jsonl`, verified by `tools/verify_manifests.py` in CI.

## Why this carrier

* It is the carrier that `engine/carriers/MANIFEST.json` (`CR-Q0-VERIFY`,
  `lane_basis`) and `engine/rn_engine/BINDING.json` name, and the one from which
  `recovery/LEDGER.json` record `ENB-04-S1` extracts the two
  `closure_round2/q_replay/LS-DATA-013-v1.0_*` members (the q0 verifier release
  1.2.1 and its report) and corroborates them against the source map's digests.
  With the carrier held here, that extraction is reproducible offline:
  `unzip -p` the member and hash it.
* The review routes `RV-DQ-017` and the RN3/CLOSE routes of 2026-09-17 cite
  objects that are members of it.

## What this is not

Mirroring is not review, replay, endorsement or closure. The words `CLOSE` and
`PROOFS` in the carrier's title are the source's; the lane is HOLD; the review
routes over its members are OPEN at zero independence credit
(`registers/json/review_queue.json`). Nothing in the archive is executed by this
repository — the recovered verifier is stored as a `.bin` under
`recovery/recovered/` and is not wired into `engine/` or CI. The five validity
premises of D1 v2.2(2) stay OPEN, `D3-LEMMA-RN-UNIF` is not closed, and nothing
here composes with the 3D track.

## 2026-09-19 — `LANE_RN_UNIF.md`

`LANE_RN_UNIF.md` (11,042 B, SHA-256 `8a3c6d55…`, Drive
`1dK4ZimCC8o9670K-9ZJS9Jtquid1CAA6`), byte-exact against the inventory: the
2026-09-16 "Drive familiarity memo (fail-closed)" for the RN_UNIF lane. Its
standing verdict table: "D3-LEMMA-RN-UNIF Piece 1 | **OPEN**"; "Piece 2 |
**OPEN** (annulus Riemann-sum driver unwritten)"; "Prize original problems
solved | **0**". Its §2 "Status quotes (skim)" block-quotes each document's
STATUS line and then summarises the rest in the memo's own words — the
sentences "Engine `d3_rn_unif.py` located (Kimi mid-build); certifier never
invoked … Piece 2 driver unwritten" are the memo's summary of CL-RNU-001, not a
quotation of it (`research/slack/registry.py` labelled them a verbatim quotation
until 2026-09-19). "T4" in this memo is the RN-UNIF T4 push region `d ∈ [5, 17]`, not
the Drive's thematic track T4.


## 2026-09-20 — the digest-bearing remainder: 65 files byte-exact

65 of this lane's 72 unheld digest-bearing items are stored byte-exact, and the 7
left, all over the 65,536-byte store limit, carry tree-only rows with the
inventory's own digest. **Zero mismatches in 65 files.** The lane's
digest-bearing gap is now zero.

Each file was fetched through the Drive connector and decoded to disk from the
session transcript, or from the file the harness spills an oversize tool result
to, so no model retyped a byte; a file was written only when its SHA-256 and
byte count already equalled the ones the 2026-09-17 inventory declares, and
every stored file was re-hashed from disk again by the process that wrote its
manifest row.

**THE SOURCE MARKS THIS WHOLE LANE HOLD AND NOT FOR SUBMISSION**, in its own
folder name, and marks one folder inside it NOT READY. Storing these bytes takes
nothing off hold, makes nothing ready and submits nothing. Nothing in this lane
enters the q0 dependency graph in either direction; `tools/claims_check.py` is
what enforces that, not this sentence.

### Five filenames, eleven objects

Drive keys files by id rather than by path, so one folder can hold several
distinct objects with the same name. This lane has five such names —
`00_HOLD_INDEX.md`, `FORWARD_WORK_PLAN_2026-09-16.md`,
`FULL_DRIVE_FOLDER_INDEX.csv`, `LANE_PRIZE_RESEARCH.md` and `ZIP_MANIFEST.csv`,
the last of them three ways — carrying eleven objects with eleven different
digests between them.

A first attempt at this port wrote each to its plain name and lost all but the
last of each set. Nothing shipped: the manifest writer re-hashes every stored
file from disk instead of trusting the fetch, and it refused five rows whose
bytes no longer matched the inventory. **Neither member of a collision keeps the
plain name now.** Every one is stored with its Drive id in the name, because
neither object is the canonical one and a plain name would imply a precedence
the source does not give; each row says so.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate or theorem. Every number above
counts files, bytes and digests. Nothing here is ready, cleared, closed or
submittable because this repository holds its bytes, and the imperative text
these files address to other threads is quoted data.


## 2026-09-20 — 3 reading copies, completing the lane

Every digest-bearing object of this lane was stored byte-exact earlier the same
day. The three native Google Docs that remained are now held as text exports.
**They are renderings, not the objects.**

This is a weaker thing than a byte-exact copy, and the difference is not a matter
of degree. A byte-exact copy here was written only because its SHA-256 and byte
count already equalled the ones `drive/inventory.jsonl` declares. **That rule
cannot apply to a native Google Doc**: the corpus declares no payload digest for
one anywhere, so nothing can prove an export and a re-fetch could differ. Each
row carries `exact: false` and `inventory_sha256: null`; its `sha256` and `bytes`
are of the export, computed at store time. No byte passed through a model — each
export was fetched through the connector and decoded to disk from the session
transcript, or from the file the harness spills an oversize result to.

### What this pass does not establish

Nothing about any claim, premise, obligation, gate, closure or theorem, and — for
this class — **nothing about the objects' bytes either.** The source marks this
lane **HOLD** and **NOT FOR SUBMISSION** in its own folder name. Storing bytes
takes nothing off hold, and holding a rendering is neither a submission nor a
step toward one. The prize reconnaissance track this lane serves is HOLD, must
not enter the q0 dependency graph in either direction, and **no original prize
problem is solved**: every prize claim in this repository carries
`original_prize_closed: false`.
