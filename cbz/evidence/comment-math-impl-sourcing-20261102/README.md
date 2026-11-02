# Implementation sourcing for comment-delivered mathematics — 2 November 2026

Task: [main#684](https://github.com/d6g8k5htny-coder/main/issues/684), claim [684/6568102222](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6568102222).
Performer: Claude S (Anthropic Claude), session `session_01S16677s1SGeAJmN3QMz4ip`.
This addendum follows the comment-math census in [main#745](https://github.com/d6g8k5htny-coder/main/pull/745), which is pinned at `683fdfa2` under `cbz/evidence/comment-math-census-20261031/`. The census itself is not modified.

**This is custody and discovery evidence, not a result.** Nothing in `extracts/` was executed. Nothing here makes a correctness claim or a disposition. Credit 0 · sci NONE · OBL OPEN.

## Why it exists

Census gap list D holds the artifacts whose posts make computational claims (scripts, runs, certificates, test counts) but have no committed implementation. This addendum asks of each one: does the implementation exist anywhere? It can turn up in three places.

1. **Inside the comment itself.** Many posts paste the complete checker as a fenced code block. Those bytes exist only as comment text, so this addendum commits each such block byte-exact under `extracts/`.
2. **In a committed file the census missed.** The comment cites a SHA-256 that equals the SHA-256 of a blob committed somewhere in some clone's history, for example on a branch or under another path.
3. **In Drive or Dropbox.** A read-only metadata search by cited SHA-256 and file name returns candidates only; no bytes were fetched.

## Files

| File | What it is |
|---|---|
| `SOURCING.json` | One row per gap-D artifact, with its outcome, extract ids, git carriers and store candidates, plus the summary and limits. |
| `SOURCING.md` | Rendered from the JSON files by `verify_extracts.py`. Do not edit it by hand. |
| `MANIFEST.json` | One entry per extracted block. It records the source comment, REST path, body SHA-256, character offsets of the block, SHA-256, git blob id, role, dependencies, whether a SHA-256 cited in the comment matches, and any committed file with the same bytes (`committed_elsewhere`). |
| `extracts/` | The extracted code. Each file holds the UTF-8 bytes of the text between the opening and closing fence, unchanged. A file is named `<comment>__b<block index>` with an extension taken from the fence's language tag, else from the file name the comment gives, else `.py` if it compiles as Python. A block shared by two census artifacts is one file with two manifest entries. |
| `verify_extracts.py` | A read-only verifier and renderer, using the standard library and `git` only. |
| `SHA256SUMS` | Digests of every file above. |

Outcomes:

- `INLINE_EXTRACTED`: the comment contains implementation code, now committed in `extracts/`.
- `COMMITTED_ELSEWHERE`: a committed file whose SHA-256 the comment cites implements the computation.
- `DRIVE_OR_DROPBOX_CANDIDATE`: a Drive or Dropbox file's title or name matches a cited SHA-256 or file name, and the title indicates code, a checker or a certificate.
- `NOT_LOCATED`: none of the above. Store matches on data files, archives or reviews are still listed on the row.

Rows take the first outcome that applies, in the order above.

## Verify

```
python3 -I verify_extracts.py --check --repo main=PATH --repo Math-=PATH [--repo NAME=PATH ...]
python3 -I verify_extracts.py --live --token-env GITHUB_TOKEN
sha256sum -c SHA256SUMS
```

The offline check covers four things:
- each extract's SHA-256 against `MANIFEST.json`;
- that `extracts/` holds no unlisted file;
- that every git carrier and every `committed_elsewhere` entry resolves with `git rev-parse COMMIT:PATH` to the recorded blob, and that the blob matches the extract;
- that the summary and `SOURCING.md` equal a fresh render.

The live check re-fetches every source comment and checks its body SHA-256. It then re-extracts each block from the recorded character offsets and compares it byte for byte with the committed file.

## Method

1. **Blocks.**
   - Candidates are fenced blocks of five or more lines, in every comment version of every gap-D artifact. A fence opens with three or more backticks or tildes and closes with at least as many of the same character. A block counts if its language tag names code (Python, Lean, Sage, Julia) or its text looks like code.
   - Each body's SHA-256 had to equal the value the census recorded.
   - Each candidate was hashed and compared with every 64-hex value the same comment cites. Python candidates were compiled but never run, and each block's git blob id was looked up in all ten clones.
   - Subagents read each block with its context and classified it:
     - `IMPLEMENTATION`: runnable given the dependencies it names;
     - `PARTIAL`: real code, but it needs project modules or sibling files;
     - `OUTPUT_OR_DATA` or `OTHER`.
   - Only `IMPLEMENTATION` and `PARTIAL` blocks are extracted.
2. **Committed elsewhere.**
   - Every blob, in the full history of all ten clones, whose base name matches a file cited by a gap-D comment was hashed.
   - Each match against a cited SHA-256 went to a subagent, which decided whether the blob implements the computation or is only the reviewed subject or an input.
3. **Drive and Dropbox.**
   - Subagents used only `search_files`, with content snippets excluded, and Dropbox title-only search, querying by cited SHA-256 prefix and by exact file name.
   - Substring coincidences were dropped. Owner names are not recorded.

## Results

- 274 gap-D rows: 75 `INLINE_EXTRACTED`, 0 `COMMITTED_ELSEWHERE`, 4 `DRIVE_OR_DROPBOX_CANDIDATE`, 195 `NOT_LOCATED`.
- 135 extracted blocks in 132 files: 63 `IMPLEMENTATION` and 72 `PARTIAL`; 70 Lean and 65 Python. 118 match a SHA-256 cited in the same comment.
- Committed elsewhere: all 41 committed blobs whose SHA-256 a gap-D comment cites are the reviewed subject or an input, not the implementation.
- Drive: the four implementation candidates are the 24 September folders cited by main#56 (two artifacts), main#57 and main#58. They hold the cited script names at the recorded sizes. They are candidates for cold recovery into git by a packet owner; nothing here fetched them.

## Limits

The authoritative list is `limits` in `SOURCING.json`, rendered in `SOURCING.md`. In short:

- Classification and committed-elsewhere judgments are by Anthropic Claude subagents of this session (independence credit 0).
- Compiling Python proves only that it parses. Running, adopting or replaying any extract is a packet owner's decision, and a `PARTIAL` extract needs the modules or sibling files its `dependencies` field names.
- Store candidates are metadata matches only; no bytes were fetched or compared.
- `NOT_LOCATED` does not prove absence.
- Comments edited after 31 October may differ from the bytes here; `--live` reports drift (none on 2 November).
- Census correction, census unchanged: four extracted Lean blocks (three for C212, one for I2-OrdinaryH0Split) match files committed at Math- `06bac48e` under `frontiers/compact_source_cutoff_baseline_20261009/`, on `codex/compact-cutoff-*` branches only. Three are byte-identical; the I2 block has one extra final newline.

## Use

An extract is a committed carrier for an implementation that was previously only comment text. Adoption works like this:
- A packet owner copies the file into the packet's path. Its git blob, recorded in `MANIFEST.json`, stays the same.
- The owner then wires it into the packet's replay and tests.

Until adoption, the extracts sit outside every workflow and test.
