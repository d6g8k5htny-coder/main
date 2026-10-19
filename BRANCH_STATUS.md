# Branch status: superseded — do not merge or extend

Checkpoint of 2026-10-19 by Claude Opus 5.5 (Anthropic), continuing Claude Code
session https://claude.ai/code/session_01V3DF6ms7UTpj27UqDG4u5h. Claude Opus 4.5
did this branch's December 2025 work in the same session. The transcript is
shared, but the model and execution are different.

## Disposition

- This branch turns the December 2025 manuscript `body` (Git blob
  `03f0a2a5be8b8e6a65176f6b6b7bd36d6107759c`) into `build_full_manuscript.js`,
  plus `package.json` and `.gitignore`.
- Dylan has since retired that material. Current `main` keeps it unchanged as a
  historical record in
  [`history/2025/`](https://github.com/d6g8k5htny-coder/main/blob/0a7e5145b55f020e02e7782fed318266d13c2447/history/2025/README.md).
  That record says the 2025 package, installation, and validation claims are not
  current repository facts or verified research results.
- This branch shares no history with `main`: `f25b04b` is not an ancestor of
  `0a7e5145`. Do not open a pull request from it, merge it, or extend the script.
- This session never posted a claim for this work on any coordination board, so
  there is nothing to release there.

## Push blocker (19 October, Claude Fable 5.1, same session)

- Pushing this branch from the session container failed four times with
  `could not read Username for 'https://github.com'`: no credential helper, no
  `gh` login, and the only local service (127.0.0.1:42827) answers 401. The
  December 2025 push went through a session git proxy that is no longer wired.
- This is an environment gap, not a denial to route around. Fix it by
  reconnecting GitHub at https://claude.ai/connect-github (with the Claude
  GitHub App installed on the repository) or by starting a session with this
  repository attached, then `git push -u origin claude/build-manuscript-script-TiApQ`.

## Findings worth keeping

- The script was never run, and no `.docx` was built or checked.
- The source text is double-encoded UTF-8. For example, `δ` is stored as
  `c3 8e c2 b4` and displays as `Î´`, so any build would print garbled symbols.
  Reading it as cp1252 bytes and re-decoding them as UTF-8 repairs 1,029 of
  its 1,035 non-ASCII runs; the other 6 need manual repair. Leave the archived
  bytes unchanged.
- The December 2025 follow-up wrongly concluded that the governance files did
  not exist, because it read only this stale local branch. Before acting, fetch
  `origin/main` and read
  [AGENTS.md](https://github.com/d6g8k5htny-coder/main/blob/main/AGENTS.md)
  and the coordination board, [#419](https://github.com/d6g8k5htny-coder/main/issues/419).
