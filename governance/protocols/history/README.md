# Historical operator protocols and the cross-model ledger — provenance copies

`governance/README.md` places historical protocols (OP-PROT-001..011, -013..018) here "as provenance;
they apply only where R17 says they are retained". As of 2026-09-19 this directory holds three objects
from the Drive canon lane's `Cross-Model Exchange` folder — the operator-protocol relays that exist
on the Drive as digest-bearing markdown, stored byte-exact against `drive/inventory.jsonl` — and a
reading copy (`exact: false`, not the object) of a fourth, the Cross-Model Review Ledger v1.0 Doc. None of the other historical protocols has
been mirrored. The current protocols are the files one level up (`OP-PROT-019-v1.1_R17.md`,
`OP-PROT-012.md`, `OP-GDN-002.md`, `OP-CNS-001-R0.2.md`); nothing here supersedes them.
`_MANIFEST.jsonl` is checked by `tools/verify_manifests.py`. Ported 2026-09-19.

## What is here

| Drive id | title | stored as (relative to this README) | bytes | SHA-256 | exact |
|---|---|---|---:|---|---|
| `1v4Cv5xBvZRl8Ing9y34xN_xgFg-_K2q5` | OP-PROT-001 — OPERATOR-ISSUED Updated Collaboration Protocol (verbatim relay, 2026-07-20) | `OP-PROT-001 — OPERATOR-ISSUED Updated Collaboration Protocol (verbatim relay, 2026-07-20)` | 9,796 | `1d6b851030741cba5c1c542dc0a7e6decf414cf78f6c2510d252d36f38d46b65` | true |
| `1pEGZoTdCUe9hu5tE_O8nY2N84STQNXhQ` | OP-PROT-003 — OPERATOR OPERATING PHILOSOPHY: close easiest items first, retire settled facts permanently (verbatim relay) | `OP-PROT-003 — OPERATOR OPERATING PHILOSOPHY: close easiest items first, retire settled facts permanently (verbatim relay)` | 4,702 | `63c69551f2bb64df3d39dd5a4bc7f20336f5f1ff6cd55bb1b95c2649c1e1200d` | true |
| `1x1JzVd5CMpP6Iqus6FSMHO6ncDrxHxUo` | OP-PROT-005 — OPERATOR PROTOCOL: Coupled Mathematical–Architectural Advancement and Required Safeguards (verbatim relay) | `OP-PROT-005 — OPERATOR PROTOCOL: Coupled Mathematical–Architectural Advancement and Required Safeguards (verbatim relay)` | 14,720 | `4867b40c46a81e3cb4d9b83034f58ead298de5686cbb19c62580e7e92fff69d1` | true |
| `1rwfvL3sJUmRVpwf3AHx2FnduUrtoIi-9pA8C9Rbku7o` | Cross-Model Review Ledger v1.0 — 2026-07-20 | `Cross-Model Review Ledger v1.0 — 2026-07-20.export.txt` | 182,476 | `e282d09a0bc0a0115acfc0fcdc5332ee01fa779b509582cc771703659df9caa7` | false (reading copy) |

## What they are, in their own words

* `OP-PROT-001`: "CLASS: OPERATOR-ISSUED PROTOCOL (verbatim relay — this instance authored only this header block)" / "PROVENANCE: received from the operator (Dylan M. Roy) in this instance's session on 2026-07-20, stated to be drafted by another AI at the operator's direction and endorsed by the operator … Hash of the verbatim body: PENDING — … this relay is single-pass transcription, flagged for one cross-model fidelity read. STANDING: this protocol updates the collaboration layer … It does NOT alter the mathematical authority order — its own text keeps Q0_MASTER.md, Q0_LEDGER.md, q0_machine.json, q0_verify.py governing, and all promotions provisional and reversible pending human review."
* `OP-PROT-003`: "CLASS: OPERATOR-ISSUED OPERATING PHILOSOPHY (verbatim relay — this instance authored only this header and the reading notes)" / "PROVENANCE: received from the operator (Dylan M. Roy) in this instance's session, 2026-07-21 … Single-pass transcription; one cross-model fidelity read invited; cross-session corroboration welcome if other sessions received it."
* `OP-PROT-005`: "CLASS: OPERATOR-ISSUED PROTOCOL (verbatim relay — this instance authored only this header)" / "PROVENANCE: received from the operator (Dylan M. Roy) in this instance's session, 2026-07-21. Single-pass transcription; fidelity read invited; cross-session corroboration welcome." Its operator text opens: "The mathematics and the research architecture are not separate workstreams. They are a coupled system that must advance together."
* `Cross-Model Review Ledger v1.0 — 2026-07-20`: "Status: Independent audit and collaboration record" / "Canonical impact: NONE unless Dylan M. Roy explicitly approves promotion and the change is entered into the governing project ledger." / "Record policy: Append-only." (each a line of the source; until 2026-09-19 this README joined the header lines of all four files with inserted periods) — "It is not itself a theorem source. The live q0 authority remains Q0_MASTER.md, Q0_LEDGER.md, q0_machine.json, and q0_verify.py." Its contribution labels include "CANDIDATE THEOREM — not canonical; requires independent verification and formal promotion."

The relays are what a model instance wrote down of what the operator said, in a single pass, and each
says so of itself; the relayed body hash `OP-PROT-001` calls PENDING is not computed or bound here.
The ledger is a text export of a native Doc (a reading copy, `exact: false`) whose entries are the
contributions of several model instances; none of them is evidence for anything in `claims/`.

## How these bytes got here

Every object was fetched through the Drive connector (`download_file_content`, base64) and the
base64 was decoded from the session transcript straight to disk, so no model retyped any byte.
For a raw file the SHA-256 and byte count were recomputed from disk and had to equal the
`drive/inventory.jsonl` row (2026-09-17 snapshot) or the bytes were not stored; every stored raw
file here passed. A native Google Doc has no payload digest anywhere in the corpus: its text export
is stored as `<title>.export.txt` with `exact: false` (a reading copy), and where the export carries
full-line `BEGIN_*BODY` / `END_*BODY` markers the marker-delimited body digest is recorded and
compared with the digest the registers declare for that Drive id. `_MANIFEST.jsonl` holds one row
per object and `tools/verify_manifests.py` re-checks every digest in CI. Archives are not extracted
and nothing was executed; `<name>.zip.members.txt` is a derived member listing, not a Drive object.

## What this directory does not establish

Nothing here changes which protocol governs (R17 does), grants any autonomy class, promotes, closes or
reclassifies anything, or turns a ledger entry into a theorem, review or independence credit. A digest
match is identity of bytes with the 2026-09-17 inventory, not fidelity of the relay to what the
operator said. Dylan Roy remains the single final authority for canonical promotion, external release,
permanent deletion and machine-root replacement.
