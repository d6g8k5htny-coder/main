# `02.1_MACHINE_ROOT_SCHEMA/01_PRIMARY_MACHINE_ROOT`

Drive folder id `1Rps49-V2b0Qc5rDq5ZANQ-QZCbXhE7wy`. The 2026-09-17 inventory gives this
folder **2 items**, and both are held byte-exact. Between them they are 6,868,024 of this
lane's 8,644,343 stored bytes.

## What is held, and at what exactness

| stored file | exactness | bytes | SHA-256 (first 16) |
|---|---|---:|---|
| `q0_machine.json` | byte-exact | 5,910,703 | `c3a93bd250c4…` |
| `q0_verify.py` | byte-exact | 957,321 | `eca1755d2537…` |

Both digests and both byte counts equal what `drive/inventory.jsonl` declares for their
Drive ids. Both were downloaded through the Drive connector, spooled to a file by the
harness because they exceed the inline limit, and decoded from that file — no part of
either was retyped.

## This contradicts a carrier record elsewhere in the repository

`engine/carriers/MANIFEST.json` carries `CR-Q0-VERIFY` for `q0_verify.py` with
`blob_stored: false`, `blob_path: null` and `not_stored_reason: SIZE`, and a `computes`
field saying the file was not read and was recorded from the inventory only. Those bytes
are now on disk beside this README. The carrier record is stale in that one respect. This
port does not edit `engine/carriers/`; the disagreement is reported, not resolved here.

Nothing else about that carrier changes. It remains non-certifying, its arithmetic remains
unexamined by this repository, and storing the file neither reads nor runs it.

## The status banners, verbatim

`q0_verify.py` opens

> q0_verify.py — CONSOLIDATED EXECUTABLE VERIFICATION LAYER (C-CONS-2026-07-19)

and describes its own contents as

> Contains, byte-verbatim, all 48 Python instruments of the 488-file q0 archive, plus the reconstructed q0_llm_verifier_v3.py shim (grade DERIVED; ledger E-CONS-2; behaviorally certified by the v4 battery and the v5 22/22 battery with zero case-level divergence from the archived validation record).

`q0_machine.json` declares its schema as `q0-consolidated/1.0` and states its consolidation
policy as

> Append-only discipline preserved. C092/C093 archives remain immutable at their recorded hashes; this set is a successor consolidation, not an edit.

Its `authority_order` lists, in the source's own order,

> Q0_C108_PORTFOLIO_CLOSE (portfolio terminal)

> C104 Theorem B package / C105 IV freeze

> C102 referee package (FREEZE_v2)

> Q0_C101_QUALITATIVE_RATE_THEOREM (live rate root)

> C094-C098 successor corrections/contracts

> Q0_C092_FINAL_MASTER (frozen core)

`registers/json/artifact_index.json` describes both files, in the row for the P1.1 priority
brief, as

> No canonical impact; active q0_machine.json and q0_verify.py remain authoritative and untouched.

## What this does not establish

Neither file was executed, imported, extracted or self-tested here, and nothing in this
repository reads either of them. `q0_verify.py` describes itself as an executable
verification layer carrying 48 embedded instruments and a self-test; none of that was run,
and the word *verification* in its own name is the source's, not a statement by this
repository. `q0_machine.json` names live roots and an authority order; those are the
source's declarations about itself, and mirroring them neither adopts nor checks them. The
digests above establish an identity of bytes and nothing further: not correctness, not
currency, not authority, and no mathematical claim whatsoever.
