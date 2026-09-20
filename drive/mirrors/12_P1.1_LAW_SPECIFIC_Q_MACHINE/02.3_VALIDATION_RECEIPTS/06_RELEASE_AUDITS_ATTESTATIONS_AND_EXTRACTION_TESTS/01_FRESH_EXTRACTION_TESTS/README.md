# `…/06_RELEASE_AUDITS_ATTESTATIONS_AND_EXTRACTION_TESTS/01_FRESH_EXTRACTION_TESTS`

Drive folder id `1gJoHwuiIeNGbDMaTlPUhQpygrsCt8PU1`. The 2026-09-17 inventory gives this
folder **5 items**, all `application/json`, and all 5 are held byte-exact.

## What is held, and at what exactness

| stored file | exactness | bytes |
|---|---|---:|
| `C095_FRESH_EXTRACTION_TEST.json` | byte-exact | 857 |
| `C096_FRESH_EXTRACTION_TEST.json` | byte-exact | 10,679 |
| `C101_FRESH_EXTRACTION_TEST.json` | byte-exact | 5,355 |
| `C108_FRESH_EXTRACTION_TEST.json` | byte-exact | 9,647 |
| `Q0_C093_FRESH_EXTRACTION_TEST.json` | byte-exact | 6,802 |

## The status banners, verbatim

`C096_FRESH_EXTRACTION_TEST.json` and `C101_FRESH_EXTRACTION_TEST.json` both record the
protocol as

> immutable verification root plus disposable execution copy

with `immutable_root_unchanged` true. Each gives its own reason for the second directory.
`C096_FRESH_EXTRACTION_TEST.json`:

> Disposable; generated reports may change there without modifying the immutable verification root.

`C101_FRESH_EXTRACTION_TEST.json`:

> Disposable; writer outputs do not mutate immutable root.

`C108_FRESH_EXTRACTION_TEST.json` records `immutable_tree_unchanged` true across three tree
digests, and its captured stderr preserves an unrelated environment warning from the machine
it ran on:

> Spreadsheet runtime warmup failed during python startup

`Q0_C093_FRESH_EXTRACTION_TEST.json` records

> "test": "fresh extraction cold start",

with `all_pass` true.

## What this does not establish

A fresh-extraction test establishes that a ZIP the author built extracts to files whose
hashes match a manifest the author wrote, on the author's machine, at one moment. That is a
packaging property. **It is not premise discharge, it is not review, and it is not
independence.** None of these tests was re-run here, none of the bundles they extract is in
this lane, and the captured stdout and stderr inside them — including the environment
warning quoted above — are records of that machine, not of anything in this repository.
