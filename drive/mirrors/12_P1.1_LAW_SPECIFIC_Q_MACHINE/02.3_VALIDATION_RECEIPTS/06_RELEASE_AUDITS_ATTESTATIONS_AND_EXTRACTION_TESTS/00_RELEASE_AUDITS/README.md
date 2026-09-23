# `…/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/00_RELEASE_AUDITS`

Drive folder id `1e62SsLicbRMCEPXOWyElY65kR9qLvaDM`. The 2026-09-17 inventory gives this
folder **5 items**, all `application/json`, and all 5 are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_RELEASE_AUDIT.json` | byte-exact | 1,422 |
| `C096_RELEASE_AUDIT.json` | byte-exact | 7,174 |
| `C101_RELEASE_AUDIT.json` | byte-exact | 2,698 |
| `C108_RELEASE_AUDIT.json` | byte-exact | 2,626 |
| `Q0_C093_AUDIT_REPORT.json` | byte-exact | 11,778 |

## The status banners, verbatim

All five record `valid` true with an empty `issues` list, for release ids
`Q0-C095-GATE-AND-FREEZE`, `Q0-C096-RECONCILED-GATE`, `Q0-C101-QUALITATIVE-RATE`,
`Q0-C108-PORTFOLIO-CLOSE` and `Q0-C093-CLOSEOUT` respectively.

`C101_RELEASE_AUDIT.json` records, inside its own claim-language block,

> "4p35_not_claimed": true,

> "finite_lower_not_claimed": true,

> "external_review_flag": true

`C108_RELEASE_AUDIT.json` carries the portfolio contract's own metadata inside its checks
block, recording `terminal_state` as `RELEASE-CLOSED`, `unowned_items` zero,
`C092_core_reopened` false, and the external dependencies

> "independent SARD-G specialist acceptance",

> "real LLM endpoint, trace, latent/logit, token-accounting, and benchmark capabilities"

with

> "future_numerical_sharpening": "new freeze required"

The per-project terminal states are not in this file; they are in `C108_ATTESTATION.json`,
one folder along.

`Q0_C093_AUDIT_REPORT.json` records its semantic core as valid with roots
`RATE_PROGRAM_GRADE` and `Q0_LIMIT`, zero unowned items, and

> "deep_execution_requested": true

## What this does not establish

A release audit that reports itself valid has checked a manifest, a file list and a set of
hashes **inside its own bundle**, at the time it ran. That bundle is not in this lane and is
not held in this repository; what is held is the audit's report about it. Nothing was
re-run, no bundle was re-hashed, and no `valid` flag here was recomputed. The word *audit*
in these filenames is the source's; none of these files is an independent review, and the
same objects' own attestations record that outside review is still required.
