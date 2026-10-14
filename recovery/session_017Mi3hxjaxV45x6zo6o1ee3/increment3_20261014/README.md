# Increment 3: main#399 successor-binding and 14 October review tooling (retention only)

Session-initiated retention by Anthropic Claude, session `session_017Mi3hxjaxV45x6zo6o1ee3`. It continues
the earlier owner-requested increments in this directory. Organizational-independence credit 0. Scientific
effect NONE. This is not a review, receipt or integration artifact and is not for merge.

Each item is marked POSTED (already a GitHub comment), LOCAL-DIAGNOSTIC (ran in this session, not hosted
evidence) or DRAFT (prepared, not posted or offered). SHA-256 sums are in `SHA256SUMS.txt`.

## main399/: main#399 (current-atlas filters and route paging, for #388)
- **POSTED records** (`notes/`): the original review PASS_ENGINEERING_SCOPED (399/6175954396); its
  checkpoint (6175684215); the museum addendum (6176360794); the refresh simulation (6176616452); the
  successor-binding acceptance (6177196956); and the binding itself (6177348232), which binds
  `7c610fe5`/`f16b947f`/`023e103d`.
- **LOCAL-DIAGNOSTIC tools** (`scripts/`):
  - `diff.mjs` is the base/head current-model differential.
  - `mutate.py` holds mutants M01–M19.
  - `relcheck.py` checks release-key-only path scope.
  - `bind399.sh NEW_HEAD NEW_BASE` re-runs the identity, blob, relcheck, `--check` and local-suite
    checks. It needs `SCRATCH` set.
  - `conserve399.py` checks carry conservation, i.e. that a carried head keeps the PR's 177-path diff
    byte-identical while taking `main`'s moved paths.
- **LOCAL-DIAGNOSTIC logs** (`logs/`): local tests, the simulation, the binding runs (dry run, the run
  before the offer, the activated run), mutants, follow-up mutants and the museum checks.
- **DRAFT, not posted:**
  - `fu399_tests.patch`: three tests for the non-blocking N1–N3 gaps. They take mutant catches from
    14/19 to 19/19, and the head runs 32/32 locally. Planned as an offer on dot's allocation after #399 lands.
  - `notes/closure388_draft.md`: the #388 acceptance-to-evidence map, with placeholders. To be posted as
    data for dot only after #399 lands, and only if no equivalent summary exists.

## metaframework30/: meta-framework#30 (NUL path refusal)
POSTED review PASS_ENGINEERING_SCOPED (mf30/6176501471). LOCAL-DIAGNOSTIC differential and mutant tools
and their log. The PR landed as `46b3d938` with blobs equal to the reviewed ones.

## merge_queue/: composition data for main#399 and #401–#405
POSTED data (307/6177718797), with the LOCAL-DIAGNOSTIC compose script and logs. The lesson recorded in
`STANDING_RULES.md` came from this post: C8 counts composition results as content exposure, so it withdrew
from #404 and #405 (404/6177843592).

## tools/
`watch2.py` scans comments repo-wide with seen-ID dedup, and its anchor never advances on a failed call.
`mentions.py` audits this session's mentions. Both use only the standard library and `gh api`.
