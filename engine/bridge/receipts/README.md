# `engine/bridge/receipts/` — run receipts (empty)

One `<idempotency_key>.json` per delivery, schema `q0.bridge.run_receipt/v1`,
written only by `engine.bridge.run_receipt.store_receipt` with exclusive
creation and validated by `python3 tools/bridge_check.py`. A conflicting second
delivery for the same key is held beside the first as
`<key>.NEEDS_RECONCILIATION.<held-digest-prefix>.json` and never replaces it.
Files here are append-only: the checker fails on any receipt that any commit
reachable from HEAD rewrote or removed, and on any working-tree difference
from HEAD. The bytes on disk are the record (canonical serialisation, no
duplicate keys); nothing here is skipped silently.

A receipt is a record that something ran. It is not evidence, not a verdict,
not an acceptance (`ACCEPTED` is owner-side; a receipt claiming it must name a
Drive-id-shaped record, and the claim is transcribed unverified) and not a
status: `scientific_status_change` is always
`UNCHANGED`. This directory is empty because the contract it belongs to is
PROPOSED / NOT DEPLOYED; the README exists so the directory does.
