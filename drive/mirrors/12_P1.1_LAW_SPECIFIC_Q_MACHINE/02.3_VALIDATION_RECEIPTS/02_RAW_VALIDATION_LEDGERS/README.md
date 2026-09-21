# `02.3_VALIDATION_RECEIPTS/02_RAW_VALIDATION_LEDGERS`

Drive folder id `1EvN5tJS20TBFE1SQFKu98MwLPRQo-vTR`. The 2026-09-17 inventory gives this
folder **3 items**, all native Google Docs, all held here as text exports — reading copies,
not the objects. They are the three parts of one ledger.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `C047R-VAL-005 — Raw per-artifact validation ledger (all 91 records) — Part 1 of 3 …` | reading copy | 6,040 | 2,505 |
| `C047R-VAL-005 — Raw per-artifact validation ledger (all 91 records) — Part 2 of 3 …` | reading copy | 5,507 | 2,710 |
| `C047R-VAL-005 — Raw per-artifact validation ledger (all 91 records) — Part 3 of 3 (END) …` | reading copy | 1,751 | 1,179 |

## The status banners, verbatim

Part 1 states the ledger's format and its own standing:

> Condensed from val_results.json. Format: [class|integrity|safety|ran] TITLE  then "-> result".

> integrity: sha-OK / sha-MISMATCH / no-hash / n/a. Additive; canonical impact NONE.

Part 3 closes the ledger and counts it:

> END C047R-VAL-005 — full per-artifact validation ledger complete. Additive; canonical impact NONE.

> Summary: 5 real capsules (3 sha-OK+ran, 2 corrupt); 35 candidate docs not byte-exact capsules; 51 scripts (31 ran, 20 need co-located inputs incl. 5 shell-out FLAG). The other big file (drive_manifest.csv) is in 04.4_MANIFEST_STAGING + CL-NAV-100.

## What this does not establish

The ledger is a condensation of a machine-readable result file — `val_results.json` — that
is **not** in this lane and is not held anywhere in this repository. What is stored here is
three text exports of a human-readable condensation of that file. Many of its lines are cut mid-word in the source itself — title strings especially, and
some result strings — though most are not: of the 191 non-blank lines across the three
parts, 128 are under eighty characters and plainly complete. The ledger's own format line
does not describe the cutting; what it says is that the file is "Condensed from
val_results.json", and the word "truncated" appears nowhere in any of the three parts. No row was re-derived, no script
was re-run, and no `sha-OK` or `sha-MISMATCH` verdict in it was recomputed here. The
integrity verdicts inside are that session's, about objects in other lanes, and they are
unrelated to the digest checks this repository performs on its own stored bytes.
