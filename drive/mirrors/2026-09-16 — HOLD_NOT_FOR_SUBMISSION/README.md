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
quotation of it (`research/slack/registry.py` said "quoted verbatim" until
2026-09-19). "T4" in this memo is the RN-UNIF T4 push region `d ∈ [5, 17]`, not
the Drive's thematic track T4.
