# Federation reference crawler: evidence and proposed registration

Built by Anthropic Claude, session 01SHCrxv (`session_01SHCrxvZuVwzVNC7SDC9TAj`), under the
owner's CBZ-0 addendum. Credit 0. Scientific effect NONE.

- **Claim history.** This work started as pickup `CBZ0-DRIFT-CRAWLER-01SHCRXV`
  (main#684/6510540816, 23:01:47Z). Claude017's claim (main#684/6510531192, 23:01:14Z)
  was earlier, and main#712 delivers it. That claim is released, and this tool is offered
  as a complementary component under master §7.4 ("merge, don't choose").
- **What is written here (B-003).** Only `tools/cbz/reference_crawler*.py`,
  `tests/test_cbz_reference_crawler.py` and this directory. `GOVERNANCE.md`,
  `TOOLING.md` and the gate matrix belong to Root. The proposed text for them is below.

## What it checks, next to the drift crawler of main#712

- **main#712** (`tools/cbz_drift/crawl.py`) checks that records claiming one identity
  agree. That covers pin sources (gitlinks, `.gitmodules`, `pins.json`,
  `workspace/repositories.json`), the 592 registry artifacts, canonical inputs, and Lean
  declarations shared across locations.
- **This tool** checks every cross-repository reference recorded anywhere in the
  federation, against the Git objects of the repository it names. It reads:
  - JSON objects that pair a repository with a path and an identity;
  - commit-pinned GitHub URLs in text;
  - line excerpts;
  - claims that a file is byte-identical at another commit.

  It also reports how a frozen reference's file has drifted since its commit:
  - Lean statements;
  - Python signatures;
  - JSON shapes.
- **No overlap.** The two tools share no paths and no checks.

## Observed run

| | |
|---|---|
| When | 2026-10-30, 10:49:37Z to 10:51:35Z |
| Environment | CPython 3.11.17, git 2.43.0 |
| Exit status | 1 |
| Logs | `crawl.stdout`, `crawl.exit` |

```
python3 -B -S tools/cbz/reference_crawler.py --workspace <Universal-Law-Workspace clone> \
  --workspace-rev origin/main --cache <mirror directory> --json report.json --markdown report.md
```

- **Mirrors.** The `--cache` mirrors are bare clones of each pinned repository. Their
  branch heads were refreshed from GitHub just before the run. A mirror that is
  missing is cloned from its `pins.json` URL.
- **Outputs.**
  - `report.md` is the complete Markdown output.
  - `summary.json` holds the full report's topology, counts and every finding except
    the two `FROZEN_*` classes, which it gives as counts by source repository.
  - The full JSON report (2,978,144 bytes, SHA-256
    `db2c11d335809f3292ec3a0c5c0465e8a70afc2db56db84b3f67bafe9d2731d9`) is reproduced by
    the command above, against mirrors at the same heads.

**Topology.** It was discovered from Universal-Law-Workspace `a4b016820329afc1787d9c191572b05cd88236ac`, with no list in the tool:

| Repository | Crawled commit | Pinned |
|---|---|---|
| Universal-Law-Workspace | `a4b016820329` | no; this is the workspace |
| main | `81acc060341e` | yes |
| Math- | `8151f592918a` | yes |
| meta-framework | `c70e1984477a` | yes |
| query- | `fb4c457b5861` | yes |
| google-drive | `ad08584b3ea9` | yes |
| trial | `3738c859d519` | yes |
| governance- | `e396002aa191` | yes |
| sandbox | `dce04a417483` | yes |

**References checked: 17,578.** These counts are from `reference_stats.py`; its output is
in `refstats.txt`.

| Kind | Count | Detail |
|---|---|---|
| JSON records | 13,396 | 13,379 commit-pinned and 17 live |
| GitHub URLs | 4,182 | commit and path checked |

The JSON records carry these identity claims:

| Claim | Count |
|---|---|
| Git blob ids | 12,163 |
| SHA-256 values | 3,651 |
| Byte counts | 12,726 |
| Line excerpts | 324 |

By source repository: main 15,954, Math- 772, meta-framework 600, Universal-Law-Workspace 170,
query- 38, governance- 27, google-drive 16, trial 1.

**Findings.**

| Class | Count | Blocking |
|---|---|---|
| LIVE_DRIFT | 1 | **yes** |
| FOREIGN_REPOSITORY | 1 | no |
| FROZEN_CHANGED | 1,294 | no |
| FROZEN_REMOVED | 4,322 | no |
| PIN_SKEW | 4 | no |
| UNPARSEABLE | 1 | no |

- **Skipped fixture files.** 66 in main, 6 in Math- and 5 in meta-framework, all under
  `tests/`, `fixtures/` or `examples/`.
- **Pin skew** is in main, Math-, meta-framework and governance-, the same four repositories
  that #712 reports as `PIN-STALE`.
- **Not blocking.**
  - The `UNPARSEABLE` file is trial `.cursor/environment.json`.
  - The foreign reference is Math- `reviews/cap_i4_d3_spectral_adapter_20261005/SOURCES.json`,
    citing `leanprover-community/mathlib4`.

### The one blocking finding

Math- `frontiers/three_fronts_20260924/SOURCES.json`, key `rn_status`:

```
{"repository": "d6g8k5htny-coder/main", "path": "docs/math_status/STATUS_RN_UNIF.md",
 "git_blob": "ccd542fb174e5a242ee343fad213aa40b35b733d", "meaning": "observed OPEN old lane; unchanged"}
```

- **What is wrong.** The record has no commit, so it is checked at main's pin. The path is
  absent on main's default line, both at the pin `81acc060` and at the tip `11e28815`.
- **Where the blob is.** It is on main's branch `chatgpt/drive-github-hardening-20260919`,
  at `fbb43601369b19ecb12447d4cc02ed44340dce60` (and `64c3c176…`).
- **Why it blocks.** The record is honest about being an open-lane observation. Without a
  commit, though, nobody can verify it against the pinned federation.
- **The fix.** It belongs to Math-'s lane and is not made here: add
  `"commit": "fbb43601369b19ecb12447d4cc02ed44340dce60"` to the record. It then verifies
  (path and blob match at that commit) and reports `FROZEN_REMOVED`, which is INFO.

## False positives found on the live federation, and fixed

Each fix has a test and a mutant (see `mutation.txt`).

The first two runs used earlier versions of the tool. Their reports stay in the author's
scratch directory and are not committed, so their numbers are observations rather than
reproducible evidence.

| Run | Blocking | Cause | Fix |
|---|---|---|---|
| 29 Oct | 2,633 | 370 `MISSING_COMMIT`: synthetic commits in `tests/fixtures` (main 352, Math- 17) and meta-framework `examples/` (1) | Files under `tests`/`test`/`testdata`/`fixtures`/`examples` are skipped and counted; `--include-fixtures` checks them |
| | | 1,937 `MISSING_PATH`, 1,895 of them percent-encoded URLs (`%20`, `%E2%80%94`) from main `docs/public-math` | URL paths are percent-decoded |
| | | 324 `IDENTITY_MISMATCH`: main `docs/site` excerpt records, whose SHA-256 covers lines `line_start..line_end` | Excerpt records are checked against exactly those lines (SHA-256, bytes and `text`). The final run checks all 324 with no mismatch. |
| 30 Oct | 15 | 12 `MISSING_PATH`: URLs read from raw JSON text kept the escape `\n` | URLs in `.json` files are matched in the decoded strings, and a backslash ends a URL path |
| | | 1 `MISSING_PATH`: `…12%20(1)/…` was cut at `)` | URL paths keep balanced parentheses |
| | | 1 `LIVE_DRIFT`: a record pinned its commit under `cut` | Key census, below |
| final | 1 | the record above | none in this tool: the record needs a commit |

**Key census.** Every key holding 40 hex characters in a reference-shaped object was
counted across the eight pinned repositories. Before each key was adopted, its values in
main, Math-, meta-framework and query- (outside fixtures) were checked against the Git
objects, without fetching:

| Key | Values matching the Git object |
|---|---|
| `blob_sha` | 93 resolvable, all equal the Git blob id |
| `blob_sha1` | 22 of 22 |
| `git_blob_sha1` | 11 of 11 resolvable without a fetch |
| `git_blob_sha` | 2 of 2 |
| `raw_sha256` | 2 of 2 raw-byte SHA-256 values |
| `byte_length` | 2 of 2 byte counts |

- **New commit keys:** `cut`, `head` and `at_commit`.
- **Not a commit key:** `math_pin`. It pins another repository.
- **`byte_identical_at_*`.** Its 15 claims, in main `docs/research-translation/20260930/`, are now
  checked at their own commit, and all 15 match.
- **Every value checked.** Each recorded value under any of these keys is checked, not
  just the first.
- **Foreign repositories.** JSON references to repositories of an owner outside the
  federation were previously neither checked nor counted. They are now an INFO class,
  `FOREIGN_REPOSITORY`. A foreign repository that shares a short name with a federation
  repository is never resolved to it.

## Self-verification (A8, B-006)

**Tests.** `tests/test_cbz_reference_crawler.py` has 29 tests, run on local multi-repository
Git fixtures with no network. The output is in `tests-normal.txt` and `tests-optimized.txt`.

```
python3 -B -S -m unittest -v tests.test_cbz_reference_crawler
python3 -B -S -O -m unittest -v tests.test_cbz_reference_crawler
```

**Mutation check.** `tools/cbz/reference_crawler_mutation_check.py` applies 46 mutants, and
each must apply exactly once. A mutant counts as killed only if the suite ran at least one
test and reported `FAILED`. There are two controls: the unmutated copy must pass, and a
syntax-broken copy must come out invalid, never killed. The output is in `mutation.txt`.

```
python3 -B -S tools/cbz/reference_crawler_mutation_check.py
```

The mutation check runs the suite in normal mode. Neither the tool nor its tests contain an
`assert` statement or a `__debug__` branch, so `-O` cannot change what the tool decides. The
`-O` test run above confirms that the suite itself also passes in that mode.

**Domain (B-006).** The tool's docstring states it. In short:
- the checks are on Git objects and file text;
- Lean statement drift compares comment-stripped declaration text, not elaborated terms;
- free-text fields such as `meaning` are not interpreted;
- a nested identity sub-object is not attached to its parent.

A pass means every reference that was read resolved and matched. It is not a mathematical
judgement.

## Proposed text for Root (B-003)

**GOVERNANCE.md, tool registry.** Add this entry:

> `tools/cbz/reference_crawler.py`: federation reference crawler (G7 component, with
> `tools/cbz_drift/crawl.py`). It discovers the federation from the workspace's pins and
> registries, and resolves every cross-repository reference record and commit-pinned URL it
> finds, by commit, path, Git blob, SHA-256, byte count or line excerpt. It blocks on a
> reference that does not resolve or does not match, and reports frozen-reference interface
> drift as INFO. Its domain is in its docstring. Self-verification: 29 tests in both
> interpreter modes, and 46 mutants with 100% required kills.

**TOOLING.md.** Add this row, in the columns of the #695 candidate:

| Tool | Usage/purpose | Actual verification/boundary | Owner/criterion |
|---|---|---|---|
| Main tools/cbz/reference_crawler.py | `python3 -B -S tools/cbz/reference_crawler.py --workspace ULW --cache DIR --json OUT`. Resolves every cross-repository reference record and commit-pinned URL in the federation. | 29 tests per mode; 46 of 46 intended source mutants killed (`reference_crawler_mutation_check.py`). Live run at ULW `a4b01682`: 17,578 references, 1 blocking. Boundary: Git-object and text identity only; no Lean elaboration or mathematical certification. | none, self-contained; 8 (G7), 10 |

**Gate matrix §10, G7.** Proposed wording: "PASS when, in the final run, both
`tools/cbz_drift/crawl.py --at pins` and `tools/cbz/reference_crawler.py` exit 0."

At this run the reference crawler exits 1 on the Math- record above. G7 is therefore RED
until that record carries its commit. That is the federation's actual state.
