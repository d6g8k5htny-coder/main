# main#141 — regenerated SIDE24 coefficient chart (review material, scientific effect NONE)

## What this package is

- Task: [d6g8k5htny-coder/main#141](https://github.com/d6g8k5htny-coder/main/issues/141), task type `chart`.
- Package id: `incoming/side24-chart-claude-20260926/`.
- Scientific effect: **NONE**. Review status: `REVIEW_REQUIRED`. This is a byte-checked
  re-reading of one pinned public JSON file plus an approximate picture of two of its
  numbers. It is submitted for review only and changes no scientific register.

## Source identity

| field | value |
|---|---|
| repository | `d6g8k5htny-coder/Math-` |
| commit | `9d7b6802424fb4715b31999066aafca8ee2f3cca` |
| path | `coefficients/side24_v1/ENCLOSURE.json` |
| bytes | 1090 |
| SHA-256 | `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811` |
| git blob | `57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7` |

The bytes were read with `git show 9d7b6802424fb4715b31999066aafca8ee2f3cca:coefficients/side24_v1/ENCLOSURE.json`
from a clone of `Math-`, and their length, SHA-256 and git blob id were compared against
the values above **before** any JSON parsing. A byte-exact copy is committed at
`source/ENCLOSURE.json` so the package is self-contained; the generator refuses (exit 2,
nothing written) if that copy, or any `--source` override, differs in length or SHA-256, and
it refuses if the parsed `scientific_acceptance` is anything but exactly `false`.

## Exact endpoints

Copied verbatim from the source as strings; never parsed through float and re-printed
(`repr(float(s)) != s` for every endpoint, which is why verbatim copying matters).

| dimension | lower (verbatim) | upper (verbatim) | cone moment (verbatim) | enclosure width (exact `Fraction(upper) - Fraction(lower)`) |
|---|---|---|---|---|
| 2 | `0.07340691930603427103` | `0.07340691930603427104` | `4/3` | `1/100000000000000000000` |
| 3 | `0.04177593184059834334` | `0.04177593184059834335` | `29/6-sqrt(6)` | `1/100000000000000000000` |

Other fields copied unchanged into `output.json`: `object` = `SIDE24-COEFFICIENT-D23-20260924-v1`,
`scope` = `coefficient of issue63 Eq15.2; parent theorem unreviewed`,
`method` = `outward rational arithmetic; analytic image and Stirling remainder bounds`,
`scientific_acceptance` = `false`.

## Independent regeneration of the pinned script

`coefficient.py` (SHA-256 `03ae6d0f15160cb681f9bbb19a84dc861c3a0db3e881291ea621061cc90c86ab`, blob
`c2d3ff339f2b9953ec08e676240cdc5edcb17a3e`) and `test_coefficient.py` (SHA-256
`81689b5394e2e949058929570ec072ccb1d4fe93cd0d36a4b258431547d02dd3`, blob
`fa9beaab7d6fcb4f0a0c5a91c6d80b455c98a912`) were copied from the same pinned commit into a
scratch directory outside this package (they are not committed here) and run with CPython
3.11.15 on Linux:

| command | wall time | exit | record |
|---|---|---|---|
| `python -B -S coefficient.py` | 0.831 s | 0 | [`regeneration/coefficient_run.txt`](regeneration/coefficient_run.txt) |
| `python -B -S -m unittest test_coefficient -v` | 1.006 s | 0 (`Ran 30 tests`, `OK`) | [`regeneration/test_run.txt`](regeneration/test_run.txt) |

Both records reproduce stdout and stderr verbatim; nothing was stripped. Byte comparison,
done programmatically (`bytes(stdout) == bytes(ENCLOSURE.json)`): the 1090 bytes that the
pinned script prints to stdout are **identical** to the pinned `ENCLOSURE.json`
(same length 1090, same SHA-256 `72b6cd92…7811`). `coefficient.py` writes no file, so there is
no written `ENCLOSURE.json` to diff; the stdout comparison is the whole comparison. This
reproduces the enclosure endpoints exactly by re-running the author's exact-arithmetic
script; it is not a review of the script's mathematics.

## The chart

![APPROXIMATE comparison of the d=2 and d=3 enclosures](chart_side24_d2_d3.png)

`chart_side24_d2_d3.png` is **APPROXIMATE** and **NON-CERTIFYING**. The two bars on one
shared vertical axis (0 to 0.08) have heights taken from `float()` approximations of the exact
lower endpoints, quantised to whole pixels; the verbatim endpoint strings, the exact widths and
the cone moments are printed beside each bar so the picture never stands in for the strings.
The enclosure width (`1/10^20`) is **arithmetic uncertainty of the specified expression, not a
finite-radius error band**, and is far below one pixel, so no error bar is drawn; the chart
says so in its footer, and `output.json` says so in `chart.note` and `non_certifying_note`.
The image is 1200x600 8-bit RGB with a fixed palette (surface `#fcfcfb`, ink `#0b0b0b`/`#52514e`,
bar `#2a78d6`), no timestamps, and a deflate stream produced by a fixed-Huffman encoder in the
generator rather than a zlib compressor, so the bytes do not depend on which zlib build the
interpreter links.

## How to reproduce

The public-intake lane accepts only `.md/.txt/.json/.csv/.png` files, so the generator and
its unit tests are **not** in this package (see "Deviation" below). They are standard-library
Python 3.11 and are bound here by digest: `regenerate.py` SHA-256 is recorded in
[`identity_detail.json`](identity_detail.json) under `generator`. With the two scripts in a
directory `GEN` and a bare checkout of this branch:

```
python3 -B -S GEN/regenerate.py --package incoming/side24-chart-claude-20260926 --check
# prints: regenerate: OK (4 files identical)
cd GEN && SIDE24_PACKAGE=/path/to/incoming/side24-chart-claude-20260926 python3 -B -S -m unittest test_regenerate -v
```

`--check` regenerates `output.json`, the PNG, `identity_detail.json` and `IDENTITY.json` into
a temporary directory and compares them byte-for-byte with the committed files; any difference
lists the paths and exits 1. The unit tests are negative controls: a one-byte change in the
source is refused before parsing with nothing written; `scientific_acceptance: true` (and `0`,
`null`, `"false"`, or a missing key) is refused; `--check` reports a one-byte change in
`output.json`, in the PNG, or in any non-generated file; the endpoint strings in `output.json`
are byte-equal to the source and not float round-trippable; the PNG parses (signature, IHDR,
every chunk CRC, IDAT inflating to exactly `height*(1+3*width)` bytes, not one flat colour, only
palette colours); the deflate stream round-trips through `zlib.decompress`; and every identity
row matches recomputed digests.

## Deviation from the task text, and why

Issue #141 was written before `main` gained the `public-intake` guard (PR #145,
`tools/public_intake_check.py`), which this PR must pass. That guard refuses any file that is
not `.md/.txt/.json/.csv/.png`, and requires `IDENTITY.json` to have exactly the keys
`schema`, `scientific_effect`, `review_status`, `sources`, `artifacts`, with three-key artifact
rows. Consequently: (1) `regenerate.py` and `test_regenerate.py` are kept outside the package
and offered for a maintainer-sponsored engineering PR, as `CONTRIBUTING.md` asks, rather than
smuggled in under another extension; (2) `IDENTITY.json` follows the checker schema exactly,
and the per-file git blob ids, the source blob, the generator identity and the tree-hash note
live in `identity_detail.json`, which `IDENTITY.json` covers like any other artifact.

## What this does NOT establish

- No theorem was reviewed. The source itself says `parent theorem unreviewed`, and this
  package does not touch that status.
- `scientific_acceptance` remains `false`; nothing here moves it.
- No live field draws, no lifetime solver, no new arithmetic: the numbers are copied
  strings and one exact subtraction.
- The chart is an illustration; it carries no mathematical content and its widths are not
  error bands.
- Nothing here changes ACCEPT/AMEND rows, `STATUS`, `PROOF_INDEX`, `LANDING_CLAIMS`,
  `claims`, prizes or `lemma_closed`; no mathematical PR is merged; publication into
  `incoming/` is review material only.

## Independence

This package was produced by an agent (Claude) working for the same account that authored
task #141 (`d6g8k5htny-coder`); it earns **zero organizational-independence credit**. Source
exposure: the pinned `ENCLOSURE.json`, `coefficient.py`, `test_coefficient.py` and `PROOF.md`
at the pinned commit, and issue #141 (plus this repository's `CONTRIBUTING.md` and intake
checker for packaging rules) were read. The technical result (byte identity of the regenerated
stdout with the pinned file; exact copy of the endpoint strings) is recorded above separately
from that zero credit.

## Identity

[`IDENTITY.json`](IDENTITY.json) (public-intake schema) lists every file in this package
except itself with byte count and SHA-256; [`identity_detail.json`](identity_detail.json)
adds git blob ids, the source and generator identities, and the note that neither identity
file can contain its own digest. Digests of the files that do not depend on this document:

| path | bytes | SHA-256 | git blob |
|---|---|---|---|
| `source/ENCLOSURE.json` | 1090 | `72b6cd92d31394cdaf5da8919a5d548e902228af1f095cc184158a71d8287811` | `57af39a05e14ed0ba8ebc00a9b4aca4dffb067c7` |
| `output.json` | 1724 | `478fb7620ea5947855006ab7af448010fdd5cc31581407b1f86d4b87ce5de2eb` | `bfc8372e866a6a3321615323c7badc62fdb0aa4c` |
| `chart_side24_d2_d3.png` | 49375 | `29aad932a160f35389c5a00e995f8cfac72932eba7d60d69b5a01815853461b7` | `dd68944dc0e97cf732d7471f01c8c01d97ee1653` |
| `regeneration/coefficient_run.txt` | 2016 | `924d7cbf3a0f0dd3b4b0bbf9ff516566582f5710f3aa41441ad4b2e2287e3950` | `728de5adb57d7b508e26636b9ca91dbc3919ce97` |
| `regeneration/test_run.txt` | 3474 | `3dea29af8ecad450e29533da4b918e033360924fd1032fa4d3fd93331fe0c017` | `c4f08f152c20841f710efde8de3a6c5b25927965` |

The identity of the whole package is its git tree hash,
`git rev-parse <commit>:incoming/side24-chart-claude-20260926`. The first commit of this
package had tree hash **(filled in by the second commit; see below)**. Because recording that
value changes this file, and therefore `IDENTITY.json` and `identity_detail.json`, the value
is written here in a second commit rather than by amending; the tree hash of the final commit
is the one printed by the command above on the PR head.
