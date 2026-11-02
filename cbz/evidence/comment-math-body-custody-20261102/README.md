# Comment-body custody for the comment-math census — 2 November 2026

Task: [main#684](https://github.com/d6g8k5htny-coder/main/issues/684), claim [684/6568483719](https://github.com/d6g8k5htny-coder/main/issues/684#issuecomment-6568483719).
Performer: Claude S (Anthropic Claude), session `session_01S16677s1SGeAJmN3QMz4ip`.
This addendum follows the comment-math census in [main#745](https://github.com/d6g8k5htny-coder/main/pull/745), which is pinned at `683fdfa2` under `cbz/evidence/comment-math-census-20261031/`. The census itself is not modified.

**This is custody evidence, not a result.** Nothing in `sources/` was executed, reviewed or disposed. Credit 0 · sci NONE · OBL OPEN.

## Why it exists

The census classifies 389 artifacts as `INLINE_ONLY`: their mathematics exists only as the text of a GitHub issue comment or pull-request review. No file, branch or ref holds those bytes, so a comment edit or deletion would lose them, and a disposition record has no committed location to cite.

This bundle commits the 362 comment bodies that carry those 389 artifacts. Some comments carry more than one artifact. The bodies are exactly as the census recorded them on 31 October. The original comments remain canonical; these files are carriers. The approach follows the precedent in [`docs/superpowers/plans/2026-10-24-exact-sample-comment-custody.md`](../../../docs/superpowers/plans/2026-10-24-exact-sample-comment-custody.md): raw bodies, no appended newline and no edits.

## Files

| File | What it is |
|---|---|
| `sources/<repo>_<number>_<comment>.body.txt` | One comment body: the UTF-8 bytes of the REST API `body` field, unchanged. Line endings are as GitHub stored them, with no terminal newline added. |
| `MANIFEST.json` | One entry per body: comment id, URL, REST path, created and updated times, byte count, SHA-256, git blob id, and the census artifact keys it carries. Also the summary. |
| `INDEX.md` | A table rendered from `MANIFEST.json` by `verify_bodies.py`. Do not edit it by hand. |
| `verify_bodies.py` | A read-only verifier and renderer that uses only the standard library. |
| `SHA256SUMS` | Digests of every file above. |

## Verify

```
python3 -I verify_bodies.py --check --census PATH/TO/comment-math-census-20261031/CENSUS.json
python3 -I verify_bodies.py --live --token-env GITHUB_TOKEN
sha256sum -c SHA256SUMS
```

The offline check covers:
- each file's byte count, SHA-256 and git blob against `MANIFEST.json`;
- that `sources/` holds no unlisted file;
- that the summary and `INDEX.md` equal a fresh render.

With `--census`, it also checks that every SHA-256 equals the census `body_sha256` for that comment, and that every `INLINE_ONLY` artifact is covered.

The live check re-fetches every comment and compares its current body byte for byte. An edit made after 31 October is reported as drift, and the committed copy keeps the census version.

## Screen and disclosure

- **Key material.** I screened the bodies for PEM private-key armour, GitHub, OpenAI-style and AWS tokens, Slack tokens, signed-URL parameters and e-mail addresses. None were found.
- **Local paths.** 60 bodies contain the owner's local workstation paths (`/Users/<owner>/…`).
- **Drive file ids.** 3 bodies contain Drive file ids.
- All of this text is already public in those comments. The same kind of path also occurs in Math- committed files and in the census excerpts on #745.
- Main's mainline has no such path yet. The reviewer or the owner should decide before integration whether these copies stay byte-exact, which is the point of custody, or go elsewhere.

## Limits

- The copies are the 31 October versions the census recorded. `--live` reports later edits; it does not update the copies.
- Custody only. It adds no disposition, review, statement or dependency edge for any artifact.
- The census classification of `INLINE_ONLY` is used as given; see #745 for its method and audit.
- Comment authorship on this board is by agent lanes posting through the owner's account. The manifest records the comment, not the performer. The census `performer` field has that.
