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
  federation, against the Git objects of the repository it names. It reads every file
  that is valid UTF-8 text, whatever its extension, and counts the binary files it cannot
  read. It reads:
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
| When | 2026-10-30, 14:00:33Z to 14:03:06Z; the run at current tips followed, to 14:05:43Z |
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
  - `summary.json` holds the full report's topology, counts, every skipped and binary path,
    and every finding except the two `FROZEN_*` classes, which it gives as counts by source
    repository.
  - Both full reports are byte-identical to those of head `1ead96bc`: the decompressed SHA-256
    values match. The crawl is deterministic, and the fixes below change no live result.
  - `report.json.gz` is the complete JSON report. It is gzip with `mtime=0`, so the
    compression is deterministic. It decompresses to 3,615,116 bytes with SHA-256
    `d8249fb0beda3b2da4c591b8610fce5aa7781a379d2028ea3a1607abf55beed0`. The `.gz` file
    itself is 205,447 bytes, with SHA-256
    `c26970fdb0fd26bf209cbf793499e7eb965b7b0b9eef0f55117affa4c5296a66`.

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

**References checked: 28,123.** These counts are from `reference_stats.py`; its output is
in `refstats.txt`.

| Kind | Count | Detail |
|---|---|---|
| JSON records | 13,397 | 13,382 commit-pinned and 15 live |
| GitHub URLs | 14,726 | commit and path checked |

The JSON records carry these identity claims:

| Claim | Count |
|---|---|
| Git blob ids | 12,163 |
| SHA-256 values | 3,652 |
| Byte counts | 12,726 |
| Line excerpts | 324 |

By source repository: main 26,499, Math- 772, meta-framework 600, Universal-Law-Workspace 170,
query- 38, governance- 27, google-drive 16, trial 1.

**Findings.**

| Class | Count | Blocking |
|---|---|---|
| LIVE_DRIFT | 1 | **yes** |
| FOREIGN_REPOSITORY | 1 | no |
| FROZEN_CHANGED | 2,435 | no |
| FROZEN_REMOVED | 4,322 | no |
| PIN_SKEW | 4 | no |
| JSON_WITH_COMMENTS | 1 | no |

- **Skipped fixture files.** 312: main 189, Math- 82, meta-framework 13, query- 13,
  Universal-Law-Workspace 8 and trial 7. All are under `tests/`, `fixtures/` or `examples/`, apart
  from main `formal/reviews/TEMPLATE_alignment_review.json`, whose placeholder `sha256` is not
  a claim. `summary.json` lists every path.
- **Binary files counted, not read.** 185: main 183 and trial 2, each listed in
  `summary.json`. No file exceeded the read limit, so there is no `UNREAD` finding, and no file
  is `UNPARSEABLE`.
- **Pin skew** is in main, Math-, meta-framework and governance-, the same four repositories
  that #712 reports as `PIN-STALE`.
- **Not blocking.**
  - Trial `.cursor/environment.json` is JSON with `//` comments. It is read after the comments
    are removed, and reported as `JSON_WITH_COMMENTS`.
  - The foreign reference is Math- `reviews/cap_i4_d3_spectral_adapter_20261005/SOURCES.json`,
    citing `leanprover-community/mathlib4`.

### The same federation at current tips

`--at` puts a candidate revision in place of a repository's pin, and the rest of the federation
stays at its pins. A second run moved every repository with pin skew to its current tip:
- main `11e28815`;
- Math- `19a55750`;
- meta-framework `0f5894f8`;
- governance- `1b62ca2f`.

The other five repositories' pins equal their tips. The tip values are in `tips.txt`; the run's
logs are `crawl-tips.stdout` and `crawl-tips.exit`, its report is `report-tips.md`, and its
compressed full report is `report-tips.json.gz`. That report decompresses to 3,710,986 bytes,
SHA-256 `4039ef88b6dec1411d047d32763ae61b2ce86ff103d91441290f0d44c5167980`, and its summary is
`summary-tips.json`.

The result: 28,636 references, with **1 blocking** finding, the same Math- record (judged at main's
tip). It reports 4 `CANDIDATE`, 2,529 `FROZEN_CHANGED`, 4,322 `FROZEN_REMOVED`, 1
`JSON_WITH_COMMENTS` and 1 `FOREIGN_REPOSITORY`. The current federation therefore has one broken
reference, and it is the same one at the pins and at the tips.

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

### Readback of the Math- fix

GitHub Task Watch (OpenAI) took the open task
([#684/6524737617](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6524737617)). On
Math- branch `chatgpt/rn-status-provenance-pin-20261030-gpt6`, at `44a583c4`, it added
`"commit": "fbb43601…"` to `rn_status`, with a README successor note.

The crawler, at this head's tool bytes, was run twice:
- with `--at Math-=44a583c4` and the rest at the pins: 28,122 references, **0 blocking**, exit 0;
- with main, meta-framework and governance- also at their current tips: 28,636 references,
  **0 blocking**, exit 0.

In both runs the record resolves at `fbb43601` and is INFO `FROZEN_REMOVED`. The logs and
Markdown reports are in `readback-math-44a583c4/`, and the result is posted at
[#684/6524973479](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6524973479).

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
| self-review | 1 | Two silent gaps, both now closed (see below). The blocking count did not change, and references went from 17,578 to 28,123. | Every UTF-8 file is read, whatever its extension. JSON Lines and notebooks are parsed. Binary files are counted. A file over the limit is an `UNREAD` finding. Owner and repository names match case-insensitively. |
| final | 1 | the record above | none in this tool: the record needs a commit |

**The self-review gaps.**
- Files over 2 MB, and text that is not UTF-8, were skipped with no trace. One such file
  exists, Math- `frontiers/finite_l_d2_coefficient_20261018/RESULT.json` (2.57 MB). It holds
  no references, but a certifying tool may not skip without counting (§2.3).
- Only eight suffixes were scanned, so 10,544 federation URLs in `.html`, `.py`, `.mjs`,
  `.ipynb` and `.log` files, mostly main's `docs/site` pages, were never checked. All of
  them now resolve.

**Review of the first head (RC-721-F1 to F6, [#721/6521783951](https://github.com/d6g8k5htny-coder/main/pull/721#issuecomment-6521783951)), and the fix for each.**

| Finding | Fix | Test and mutants |
|---|---|---|
| F1: the crawl could exit 0 while a file's JSON records stayed unchecked | `UNPARSEABLE` and `UNREAD` are blocking, so an incomplete census never exits 0. JSON with comments and trailing commas is read rather than left unparsed. | `test_unparseable_json_blocks_as_an_incomplete_census`, `test_json_with_comments_and_trailing_commas_is_read`; mutants "unparseable not blocking", "unread not blocking", "json comments not read", "comment stripping ignores strings", "trailing commas kept" |
| F2: only the first repository and commit alias was used, and malformed identity values were dropped | Repository aliases that disagree are `ALIAS_CONFLICT`. Each distinct commit is checked. An invalid identity value is `MALFORMED_IDENTITY`; a null counts as absent. An abbreviated commit (7 to 39 hex digits) resolves through Git. A `ref` that names a branch is a live reference. | `test_record_aliases_commits_and_refs`, `test_records_with_an_identity_are_references_and_malformed_values_are_reported`, `test_named_refs_and_abbreviated_commits_are_checked_and_path_only_records_are_not_references`; mutants "alias conflict ignored", "malformed identity silent", "only the first commit checked", "abbreviated commits rejected", "named refs malformed", "named refs not references", "path-only records checked" |
| F3: files over 2 MB, non-UTF-8 text and eight-suffix scanning | Every UTF-8 file is read. Binary paths are listed. A file over the 64 MB limit is a blocking `UNREAD`. | the head `708c70e2` tests and mutants, plus the path lists |
| F4: JSON shapes ignored array elements after the 50th | Every element is walked. | `test_json_shape_walks_every_array_element`; mutant "json shape truncated" |
| F5: a SHA-256, byte or excerpt claim on a directory path was accepted | It is an `IDENTITY_MISMATCH` ("a file identity is recorded, but the path is a tree"). | `test_file_identity_on_a_directory_is_a_mismatch`; mutant "file identity on a directory accepted" |
| F6: a registered but unpinned repository had no clone URL | The URL is derived from the registered `owner/name`. A test resolves the repository through a real `--cache` mirror, with no override, and checks that the derived URL is what `--offline` refuses. | `test_registered_unpinned_repository_resolves_through_the_cache`; mutant "unpinned repositories get no url" |

**Review of heads `b8994c6f` and `1ead96bc`, and the fix for each.** The reviews are
[#721/6524177083](https://github.com/d6g8k5htny-coder/main/pull/721#issuecomment-6524177083) and
[#721/6524450718](https://github.com/d6g8k5htny-coder/main/pull/721#issuecomment-6524450718).

| Finding | Fix | Test and mutants |
|---|---|---|
| F7: a missing or wrong-shaped workspace registry, canonical input or registry document collapsed the topology silently | The new blocking class `TOPOLOGY_INCOMPLETE` covers each case: the workspace registry missing or not an object, no `repositories` list, an entry without `full_name`, no usable `canonical_inputs.repository_registry`, a registry host outside the topology, unreachable, or without its commit, the registry document missing, and the registry with no `repositories` map. Registered nodes are now built before the canonical registry is read, so a registered host works. | `test_incomplete_topology_inputs_block`, 10 cases; 11 mutants, one per check plus "topology incomplete not blocking" |
| F8: malformed `byte_identical_at_*` values were dropped | Every prefixed value is validated, and an invalid one is `MALFORMED_IDENTITY`. A claim with no blob, sha256 or bytes to compare is `MALFORMED_IDENTITY`. | `test_byte_identical_values_are_validated`; mutants "malformed byte-identical values silent" and "byte-identical claims without identity accepted" |
| F9: an unterminated `/*` comment could hide the rest of a file | Stripping comments raises, so the file is `UNPARSEABLE`, and its whole raw text is still scanned for URLs. | `test_unterminated_comment_is_unparseable_and_its_tail_scanned`; mutant "unterminated comment accepted" |
| F10: the mutation result was unqualified | Hermetic fixtures, a baseline before and after the mutants, and the kept 74-of-75 run (all at `1ead96bc`), plus F11 | below |
| F11: a run with errors only could count as a kill | A kill now needs at least one assertion failure, and an error-only run is invalid. A new control, a copy whose `crawl()` raises, must come out invalid (`errors=42`) or the harness exits 2. Each kill line names up to three failing tests. | the harness's controls |
| F12: the PR's stated disposition was stale | The PR body records every review and finding. | not applicable |

**A rule tried and withdrawn.** Treating a record with only a repository and a path as a live
reference gave two false `LIVE_DRIFT` findings, on trial `needs_attention/SNAPSHOT.json`. Its cards
name the repository they concern, while their `path` is local to trial. Such a record claims no
file identity, so it is not a reference. A regression test and the mutant "path-only records
checked" now pin this down. The identity-value census below found no record with two distinct
commits and none with disagreeing repository aliases. It found 13 invalid values:

| Count | Value | Where | Rule |
|---|---|---|---|
| 9 | `ref: "main"` | main `docs/NAVIGATION.json` | branch name, live |
| 3 | abbreviated commits | main `docs/retro/GB-8g.agent7.node-candidates.json` | resolve through Git |
| 1 | template placeholder | main `formal/reviews/TEMPLATE_alignment_review.json` | synthetic |

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

**Tests.** `tests/test_cbz_reference_crawler.py` has 44 tests, run on local multi-repository
Git fixtures with no network. The output is in `tests-normal.txt` and `tests-optimized.txt`.

```
python3 -B -S -m unittest -v tests.test_cbz_reference_crawler
python3 -B -S -O -m unittest -v tests.test_cbz_reference_crawler
```

**Mutation check.** `tools/cbz/reference_crawler_mutation_check.py` applies 89 mutants, and
each must apply exactly once. A mutant counts as killed only if the suite ran at least one
test and reported at least one assertion failure. A run with errors only is invalid. There are
four controls:
- the unmutated copy must pass before the mutants;
- it must pass again after them;
- a syntax-broken copy must come out invalid;
- an error-only copy, whose `crawl()` raises, must come out invalid.

The output is in `mutation.txt`.

```
python3 -B -S tools/cbz/reference_crawler_mutation_check.py
```

Fixture repositories set `commit.gpgsign=false`, so the suite is hermetic and never calls a host
signer. The harness runs the unmutated suite both before and after the mutants. If the final
baseline fails, the run is invalid, with exit 2, because errors could then pass for kills. Each
kill line shows the suite's failure and error counts.

**An evidence incident, and its correction.** The mutation output first committed at `b8994c6f`
(75 of 75) is not trustworthy for its late mutants:
- the fixtures then inherited the host's `commit.gpgsign=true`;
- after about 20,000 fixture commits, the session's signer ran out of file descriptors;
- fixture commits then failed;
- the harness of that time counted any `FAILED (` as a kill.

The fix is in the fixtures and the harness, as above. The clean re-run on the hermetic suite then
gave **74 of 75**. The mutant "pin skew reported for candidates" survived, which shows the earlier
75 of 75 had included at least one false kill. It survived because the candidate test used a
candidate equal to main's tip, so pin skew could never appear. That output is kept as
`mutation-74of75-before-test-fix.txt`. The test now moves the tip past the candidate, and
`mutation.txt` is the full re-run on these final bytes: 75 of 75, with both baselines passing. Every
kill includes at least one assertion failure; five also raise errors in some tests. That head was also committed unsigned, because the signer itself had failed;
no rule on the branch requires signatures.

The mutation check runs the suite in normal mode. Neither the tool nor its tests contain an
`assert` statement or a `__debug__` branch, so `-O` cannot change what the tool decides. The
`-O` test run above confirms that the suite itself also passes in that mode.

**Domain (B-006).** The tool's docstring states it. In short:
- the checks are on Git objects and file text;
- Lean statement drift compares comment-stripped declaration text, not elaborated terms;
- free-text fields such as `meaning` are not interpreted;
- a nested identity sub-object is not attached to its parent;
- binary files (not valid UTF-8, such as PDFs and images) are listed, not parsed;
- a record with no identity key (only a repository and a path) is not a reference;
- a topology input that is missing or of the wrong shape blocks as `TOPOLOGY_INCOMPLETE`, rather
  than being read as empty.

A pass means every reference that was read resolved and matched. It is not a mathematical
judgement.

## Proposed text for Root (B-003)

**GOVERNANCE.md, tool registry.** Add this entry:

> `tools/cbz/reference_crawler.py`: federation reference crawler (G7 component, with
> `tools/cbz_drift/crawl.py`). It discovers the federation from the workspace's pins and
> registries, and resolves every cross-repository reference record and commit-pinned URL it
> finds, by commit, path, Git blob, SHA-256, byte count or line excerpt. It blocks on a
> reference that does not resolve or does not match, and reports frozen-reference interface
> drift as INFO. An incomplete census (an unparseable or unread file) also blocks. Its domain
> is in its docstring. `--at NAME=REV` crawls one repository at a candidate head, so a pull
> request can be gated against the federation. Self-verification: 44 tests in both interpreter
> modes, and 89 mutants with 100% required kills; a kill needs an assertion failure.

**TOOLING.md.** Add this row, in the columns of the #695 candidate:

| Tool | Usage/purpose | Actual verification/boundary | Owner/criterion |
|---|---|---|---|
| Main tools/cbz/reference_crawler.py | `python3 -B -S tools/cbz/reference_crawler.py --workspace ULW --cache DIR --json OUT`. Resolves every cross-repository reference record and commit-pinned URL in the federation. | 44 tests per mode; 89 of 89 intended source mutants killed, each by an assertion failure (`reference_crawler_mutation_check.py`). Live runs at ULW `a4b01682`: 28,123 references at the pins and 28,636 at current tips, 1 blocking in each. Boundary: Git-object and text identity only; no Lean elaboration or mathematical certification. | none, self-contained; 8 (G7), 10 |

**Gate matrix §10, G7.** Proposed wording: "PASS when, in the final run, both
`tools/cbz_drift/crawl.py --at pins` and `tools/cbz/reference_crawler.py` exit 0. A pull
request to a federation repository passes its merge gate when
`tools/cbz/reference_crawler.py --at NAME=HEAD` exits 0."

**What blocks a merge, and what does not.** The owner's addendum asks for merge synchronization to
be blocked when drift is detected. Here is how that maps onto this tool:
- **Blocks, with or without `--strict`:**
  - a reference that a candidate head breaks: `MISSING_*`, `IDENTITY_MISMATCH`, or a live
    reference into the candidate whose content changed (`LIVE_DRIFT`);
  - every check of the candidate's own records.
- **Informational by default:** the 2,435 `FROZEN_CHANGED` rows. These are commit-pinned citations
  whose file has changed since, by design. Blocking on them would block on history.
- **`--strict`** makes interface drift in frozen citations block. That covers Lean statements,
  Python signatures and JSON shapes.
- **Shared Lean declarations** must agree across locations under #712's `LEAN-DRIFT`.

The choice between the default and `--strict` for G7 is Root's (B-003).

At this run the reference crawler exits 1 on the Math- record above. G7 is therefore RED
until that record carries its commit. That is the federation's actual state.
