# Census of mathematics delivered in comments — 31 October 2026

Task: [main#684](https://github.com/d6g8k5htny-coder/main/issues/684), claim [684/6563844530](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6563844530).
Performer: Claude S (Anthropic Claude), session `session_01S16677s1SGeAJmN3QMz4ip`.

**This is a discovery inventory, not a result.** It lists mathematical artifacts that were delivered inside GitHub comments, PR reviews and issue or PR descriptions. For each one it records where its bytes are committed, what implements it, and which records mention it. It makes no correctness claim and no CLOSED or QUARANTINED disposition. Credit 0 · sci NONE · OBL OPEN.

## Why it exists

The existing inventories start from committed files: the Workspace `INVENTORY.md`, the coverage ledger `data/hypergraph/math_coverage_ledger_20261023.md`, and Claude 54's packet ledger. If mathematics was delivered only as comment text, none of them can see it. The Workspace inventory says so itself: at `a4b01682` its "Remaining discovery" section lists "all comments/handoffs" as not yet enumerated. This census covers the mathematical part of that gap, working from the comments themselves.

## Files

| File | What it is |
|---|---|
| `CENSUS.json` | The data: source cuts, coverage, limits, one row per artifact, and the summary. |
| `CENSUS.md` | Rendered from `CENSUS.json` by `verify_census.py`. Do not edit it by hand. |
| `verify_census.py` | Read-only verifier and renderer. Python standard library and `git` only. |
| `README.md` | This file. |
| `SHA256SUMS` | Digests of the four files above. |

## Row fields

- `key`, `kind`, `statement`, `status_in_post`, `stage`, `depends_on`: what was delivered and the status the post gives itself. Self-labels are quoted, not adopted.
- `performer`: who delivered it, as the classifying subagent read it from the post.
- `family`: one of Anthropic, OpenAI, Grok, Google, Owner or Unknown. It is the first family that `performer` names, so a label like "OpenAI review of a Claude packet" counts as OpenAI. Grok covers both the xAI Grok lanes and the Cursor-hosted Grok Bot agents, because their posts describe themselves inconsistently. The exact wording stays in `performer`. The case-insensitive patterns are:
  - Anthropic: `Anthropic|\bClaude`
  - OpenAI: `OpenAI|Codex|ChatGPT|\bGPT-?\d|Astra Pro|\bdot\b`
  - Grok: `Grok|xAI|Cursor`
  - Google: `Google|Gemini`
  - Owner: `\bowner\b|Dylan|d6g8k5htny-coder`
  - Unknown: no match.
- `versions`: every comment carrying the artifact. Each records its repository and number, comment id, URL, REST path, created and updated times, and body length and SHA-256 at read time.
- `custody.class`:
  - `COMMITTED_ON_MAIN`: at least one cited blob is in the main or Math- tree at the cut.
  - `COMMITTED_ON_BRANCH_ONLY`: committed, but not in the main or Math- tree at the cut: on an unmerged branch or PR head. One row, `PAL-01`, is here because its only copies are on the default branches of meta-framework and the archived google-drive repository.
  - `DRIVE_OR_DROPBOX_ONLY_CITED`: the post cites only Drive or Dropbox locations.
  - `INLINE_ONLY`: no committed copy was found.
  - `UNKNOWN`: custody could not be determined.
- `custody.copy_kind`:
  - `BYTE_IDENTICAL`: the evidence describes an exact copy.
  - `INCORPORATED`: the content is restated or credited in a committed file.
  - `UNSPECIFIED`: neither of the above.
- `custody.carriers` and `implementation.carriers`: `{repo, commit, path, blob, role}` plus an optional `sha256`. Every carrier is checked mechanically.
- `implementation.class`:
  - `FOUND`: a script, test, Lean file or certificate that computes or checks the content is committed.
  - `CITED_NOT_FOUND`: the post cites an implementation that was not found.
  - `NONE_NEEDED`: purely deductive content.
  - `NONE_FOUND`: computational content with no implementation located.
- `records`: mentions of the comment id, the artifact key or a committed carrier path. They are found by `git grep` at the cuts over `docs/retro/`, the coverage ledger, `cbz/`, `docs/cbz/`, `STATUS.md` and the Workspace inventory files.
- `probe`: the distinctive comment lines searched at the cuts. See Method, step 4.
- `claude54_packets`: cross-reference to Claude 54's packet ledger at Math- `8151f592`, for artifacts whose carrier is a Math- packet.
- `audit` (sampled rows only): the outcome of the audit for each of statement, custody and implementation. The values are `AGREE`, `CORRECTED` (with the old value), `DISPUTED_RECORDED_STANDS` or `DISPUTED_UNRESOLVED`. See Method, step 7.
- `incorporation_check` (rows checked by the citation pass): `PROMOTED` (with the old class), `CITES_ONLY` or `SPLIT_KEPT`, with both agents' evidence. See Method, step 8.

## Gap lists

`CENSUS.md` lists each gap separately:

- **A.** Inline only.
- **B.** Committed only on an unmerged branch.
- **C.** Only Drive or Dropbox cited.
- **D.** Computational content without an implementation (`CITED_NOT_FOUND` or `NONE_FOUND`).
- **E.** Not mentioned by any record.
- **U.** Custody unknown.

## Verify

Clone main, Math-, Universal-Law-Workspace, meta-framework and google-drive with full history and all branches; branch-only carriers name branch commits. Then run, from this directory:

```
python3 -I verify_census.py --check \
  --repo main=PATH --repo Math-=PATH --repo Universal-Law-Workspace=PATH \
  --repo meta-framework=PATH --repo google-drive=PATH
sha256sum -c SHA256SUMS
```

A carrier in a repository with no `--repo` clone makes the run fail, unless `--allow-missing-repos` is given; the output then lists it as unchecked.

The verifier checks the following:

- Every carrier: `git rev-parse <commit>:<path>` must equal the recorded blob, and the bytes' SHA-256 must match when one is recorded.
- Every record: the needle must occur in that file at that commit.
- Each custody class must agree with blob membership in the main and Math- trees at the cuts.
- Classes without carriers must list none, and `FOUND` must list some.
- The inline probe is re-run on every row that is not `COMMITTED_ON_MAIN`. The run fails if two or more probe lines occur in one committed file, or if the recorded maximum differs.
- The summary and gap lists must equal a fresh recomputation.
- With `--check`, `CENSUS.md` must equal a fresh rendering.

`--live [--token-env GITHUB_TOKEN]` re-fetches every cited comment and reports body drift. It only reports, because comments may be edited after the cut.

The verifier does not check whether a model judgment is right: the classification, `kind`, `statement`, or the custody of rows with no carrier. It cannot prove absence either, and it does not establish that the census is complete. Coverage and its measured limits are stated in `CENSUS.json`, under `coverage` and `limits`.

## Method

1. **Corpus.**
   - Every issue and PR comment (REST `issues/comments`, all pages), every PR review body and every issue or PR description, in the ten repositories this session can read.
   - Classification ran in three stages, recorded in `coverage`:
     - Stage A: a lexical prefilter.
     - Stage B: reviews and descriptions.
     - Stage C: triage of every comment the prefilter excluded that is over 800 characters and carries a math marker. It was added after a random audit showed the prefilter missed 4.4%.
   - A uniform sample of the residual bounds what remains unlisted.
2. **Custody.**
   - For each artifact, a subagent searched the pinned main, Math- and Workspace trees and all refs for its bytes, file names, hashes and distinctive lines. Each row keeps the subagent's evidence note.
   - Carriers were then extracted deterministically. A blob id is kept only if it resolves at the cited commit and path, and its prefix appears in the evidence text.
   - A small normalization pass handled the remaining carriers.
   - Any carrier that still could not be verified was dropped. Its row was downgraded to `UNKNOWN` (custody) or `CITED_NOT_FOUND` (implementation) rather than kept unverified.
3. **Implementation:** the same search, for scripts, tests, Lean files and certificates that compute or check the content.
4. **Inline probe.**
   - For every row that is not `COMMITTED_ON_MAIN`, up to five distinctive literal lines of the comment are searched with `git grep -F` at the cuts.
   - Two or more lines in one committed file reclassify the row as `COMMITTED_ON_MAIN`, with that file as carrier.
   - Single hits stay as recorded.
5. **Records:** recomputed by `git grep` as described under Row fields.
6. **Deduplication:** by artifact key. Mirrors and later versions become extra `versions` on one row.
7. **Audit.**
   - A stratified random sample of 50 rows was drawn with seed 20261101, from the 827 rows complete at 23:30Z on 1 November. The strata are custody classes, plus two implementation classes.
   - Subagents told to refute each statement, custody class and implementation class re-checked the sample.
   - Every dispute went to an adjudicator. A value changed only when a second adjudicator independently reached the same replacement.
   - The outcome counts are the second-to-last entry of `limits`.
   - All custody disputes ran one way: a row listed in a gap had a committed copy, usually a credited restatement. No sampled `COMMITTED_ON_MAIN` row was disputed.
8. **Citation pass.**
   - Prompted by step 7: for every gap-list row outside the sample, the comment ids were searched with `git grep -F` in the main and Math- trees at the cuts.
   - Each row with a citing file was re-checked by a subagent that opened those files and decided whether one carries the mathematics or only cites it.
   - A row moved to `COMMITTED_ON_MAIN` (`copy_kind` `INCORPORATED` or `BYTE_IDENTICAL`) only when a second agent independently agreed.
   - The counts are the last entry of `limits`.
   - A restatement that credits only an issue number or a key is not found this way.

## Use for retention and compaction

For each row in gap lists A, B and C, the only copy these searches located is a comment, an external Drive or Dropbox file, or an unmerged branch. The errors the audit found all ran toward over-listing: a listed row turned out to have a committed restatement, and no row marked committed lacked a carrier. Treat the lists as a superset of what is at risk.

Promote those bytes into a committed packet, or into a recorded bundle with blob and SHA-256, before any of the following removes a carrier:

- a branch prune;
- a thread or comment cleanup;
- a post-gate sweep.

This census promotes nothing itself.

## Proposed Workspace inventory addendum (not made here)

The inventory owner could add a line under "Remaining discovery" in `INVENTORY.md`, with a matching entry under `limitations` in `INVENTORY.json`:

> Mathematics delivered in comments, reviews and descriptions to 31 October 2026 17:20Z is censused in main `cbz/evidence/comment-math-census-20261031/` (custody, implementation and record coverage per artifact, with stated recall limits). Non-mathematical handoffs remain unenumerated.

Whether and when to adopt it is the inventory owner's call.

## Boundaries

- Classification and custody are judgments by Anthropic Claude subagents of this session, with independence credit 0.
- Anthropic-authored artifacts, including this performer's, are listed structurally like every other row and are not judged.
- This contribution makes no change to any canonical surface (`STATUS`, `PROOF_INDEX`, `TOOLING`, `BASELINE`, `DECISIONS`) and no proof disposition.
- Scientific review of anything listed belongs to a different family. An engineering read of the verifier may come from any nonauthor.
