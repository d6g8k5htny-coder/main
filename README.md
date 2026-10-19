# ABANDONED BRANCH — DO NOT USE

This branch holds one Claude Code work commit, `f759fb8` (2025-12-16), made in
answer to a request for `build_full_manuscript.js` and `CONTRIBUTING.md`. It was
**superseded before it was ever proposed**: no pull request was opened from it.
It is kept in history rather than deleted.

If you are reading this from a resumed session, your checkout may predate the
current project. Run `git fetch origin main` and start from `AGENTS.md` there. A
2026-10-19 check in this session read the stale checkout as the whole repository
and wrongly concluded that the governance files did not exist.

The branch forks from `f25b04b`, `main`'s own December 2025 state, and was never
merged back. As of `main` at `0a7e5145b55f020e02e7782fed318266d13c2447`:

* **Contributor guidance** lives in `CONTRIBUTING.md`, `AGENTS.md` and
  `governance/OP-WORKFLOW-20260930.md`. Pointing `CONTRIBUTING.md` at board #419
  instead of #307 is the scope of PR #421 (unmerged at `0a7e514`).
* **The 2025 material** is historical record only: Dylan identified that
  presentation as obsolete, and its originals were relocated unchanged to
  `history/2025/`. `body` on this branch is byte-identical to
  `history/2025/body.original.txt` (blob `03f0a2a5be8b8e6a65176f6b6b7bd36d6107759c`).
  The README replaced by this notice is `history/2025/README.original.md`
  (blob `108b169941ba8986394f1227459047846e227267`, also at `f25b04b`).
* **A faithful `build_full_manuscript.js`** (`body` renamed, changing only its
  output path) was committed ten seconds before `f759fb8` from the same base, on
  `claude/build-manuscript-script-TiApQ` at `dff6072`.

## Why `f759fb8` was discarded rather than fixed

* Its `build_full_manuscript.js` put text that is **not in the author's source**
  into a document titled and bylined as Dylan Roy's manuscript: a section
  "1.2 The Complexity Principle" (the source's 1.2 is "Information as the
  Fundamental Primitive"), a boxed "Principle 1.1 (Complexity Minimization)",
  and a new paragraph beginning "This work proposes that the disparate
  structures". Despite its name, it built only a stub of the manuscript.
* Its `CONTRIBUTING.md` advertised the 2025 README's package layout
  (`pip install -e .`, a `src/` package, `tests/test_gauge_theory.py`,
  `docs/manuscript.docx`), none of which existed in this repository then or now,
  and it promised acknowledgements on the owner's behalf.

Nothing on this branch should be cited, merged, or used as a source.

## Correction, 19 October 2026

The first version of this notice (`a712f4d`) said the branch "shares no history
with today's `main`". That was false: `f25b04b` is an ancestor of `main`, whose
root is the same `4294ccd`. The check ran in a shallow clone, where
`git merge-base` finds no common ancestor beyond the shallow boundary. Run
`git rev-parse --is-shallow-repository` before making an ancestry claim.
