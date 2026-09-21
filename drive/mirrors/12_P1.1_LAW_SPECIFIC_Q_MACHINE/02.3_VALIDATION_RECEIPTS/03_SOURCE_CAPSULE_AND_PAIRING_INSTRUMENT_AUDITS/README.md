# `02.3_VALIDATION_RECEIPTS/03_SOURCE_CAPSULE_AND_PAIRING_INSTRUMENT_AUDITS`

Drive folder id `1mjJ9gfvK9qd6Yfk3yCGhLmRM4bUe6JxH`. The 2026-09-17 inventory gives this
folder **6 items**, all native Google Docs, all held here as text exports — reading copies,
not the objects.

## What is held, and at what exactness

| stored file | exactness | bytes on disk | Drive-reported size |
|---|---|---:|---:|
| `GP-AUD-055-v1.0 — Hardened Pairing Instrument Self-Test Receipt.export.txt` | reading copy | 1,690 | 2,133 |
| `GP-AUD-056-v1.0 — Static source audit of GP-DATA-054 …` | reading copy | 12,861 | 7,766 |
| `GP-AUD-057-v1.0 — Drive Source Capsule Reconstruction Receipt.export.txt` | reading copy | 3,421 | 3,285 |
| `GP-AUD-059-v1.0 — Amended Pairing Instrument 21-Control Receipt.export.txt` | reading copy | 3,166 | 3,394 |
| `GP-AUD-061-v1.0 — Stale-export incident, current-source revalidation …` | reading copy | 12,440 | 7,269 |
| `GP-AUD-063-v1.0 — GP-DATA-054-v1.1 source-publication and reconstruction gate audit.export.txt` | reading copy | 7,541 | 5,378 |

Two of these documents claim the same artifact id at different times; `GP-AUD-061` records
the collision and the reidentification in its own opening notice. Both Drive objects are
held, under their own titles, exactly as the inventory names them.

## The status banners, verbatim

`GP-AUD-055-v1.0` records

> STATUS:            13/13 SELF-TESTS PASS; PRODUCTION BARGMANN–FOCK EXECUTION NOT PERFORMED

and bounds itself:

> The receipt verifies only the control logic, analytic fixtures, fail-closed gates, source hashing, and exact-field adaptive endpoint control. It does not verify a full Bargmann–Fock field census, full torus separatrix pairing, the historical six-field output, a_hat≈0.57, or any asymptotic exponent.

`GP-AUD-056-v1.0` records

> Status: MATERIAL TECHNICAL BLOCKERS IDENTIFIED — NO PRODUCTION EXECUTION — NOT INDEPENDENT REVIEW

and states what it is not:

> This audit does not claim a hash match, byte-exact reconstruction, installed-source identity, independent execution, production-field validity, or theorem evidence.

`GP-AUD-057-v1.0` records

> STATUS:            PASS — FRESH DRIVE EXPORT RECONSTRUCTS EXACT FROZEN SOURCE

and then limits it:

> This receipt establishes source reconstructability and identity only. It does not establish:

`GP-AUD-059-v1.0` records

> STATUS:            PASS — 21 OF 21 AUTHOR-SIDE CONTROLS; OUTSIDE QUALIFICATION PENDING

`GP-AUD-061-v1.0` records the incident and the correction, classifying its own independence:

> Independence class: LOW / SAME-LINE RECONSTRUCTION.

and

> Current disposition:

> SOURCE RECONSTRUCTABILITY PASSED / 13 OF 13 LEGACY SELF-TESTS PASSED / 3 OF 3 ADVERSARIAL QUALIFICATION DEFECTS CONFIRMED / AMENDED SOURCE AND OUTSIDE-LINE EXECUTION REQUIRED BEFORE PRODUCTION.

`GP-AUD-063-v1.0` records

> Status: SOURCE-PUBLICATION GATE FAILED — 21/21 AUTHOR RECEIPT READABLE / EXACT v1.1 BYTES NOT RECONSTRUCTABLE FROM DRIVE

and

> Source-publication gate: FAIL.

> Exact-source reconstruction gate: FAIL.

> Author receipt readability: PASS.

> Independent execution: NOT POSSIBLE FROM CURRENT DRIVE ARTIFACTS.

## What this does not establish

This folder is the clearest illustration in the lane of why a mirror is not a review. Six
audits by one provider's line audit that same line's instrument; `GP-AUD-061` labels its own
independence LOW, and `GP-AUD-056` says in its own words that it is not independent review.
**A same-provider reviewer earns zero independence credit, and nothing here changes that.**

Nothing was re-executed. No hash claimed inside these documents was recomputed here: in
particular, the `7f061dd7…` and `dbff5918…` digests these receipts argue about belong to an
object in another lane which this repository does not hold. The two audits that disagree
about whether a capsule reconstructs are both stored, unreconciled, because reconciling them
would be a judgement this port has no standing to make.
